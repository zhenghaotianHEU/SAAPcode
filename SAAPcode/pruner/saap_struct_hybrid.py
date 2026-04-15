import copy
import time
import torch
from torch import nn

from core.datasets.example_samples import get_examples
from core.pruner.saap_pruner import SAAPImportance
from core.models.hf_llama.modeling_llama import LlamaAttention, LlamaMLP


def _standardize_rows(x: torch.Tensor, eps: float = 1e-6):
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    return (x - mean) / std


def _compression_balanced_threshold(attn_metric: torch.Tensor, mlp_metric: torch.Tensor, pruning_ratio: float):
    prune_metric = torch.cat([attn_metric.reshape(-1), mlp_metric.reshape(-1)])
    sorted_prune, indices = torch.sort(prune_metric, descending=True)
    compression_weight = torch.ones_like(sorted_prune)
    compression_weight[indices < attn_metric.numel()] = 512.0 / 3.0
    target = torch.sum(compression_weight) * (1 - pruning_ratio)
    threshold = sorted_prune[torch.argmin(torch.abs(torch.cumsum(compression_weight, 0) - target))]
    return threshold


def _collect_saap_group_scores(model, tokenizer, device, num_examples, seq_len, taylor, logger=None, args_like=None):
    if logger is not None:
        logger.log(f'[hybrid] loading calibration dataset={args_like.dataset}, nsamples={num_examples}, seq_len={seq_len}')
    calibration = get_examples(args_like.dataset, tokenizer, num_examples, seq_len=seq_len).to(device)
    if logger is not None:
        logger.log(f'[hybrid] calibration tensor shape = {tuple(calibration.shape)}')

    attn_acts = {}
    mlp_acts = {}
    hooks = []

    def make_input_hook(store, key):
        def _hook(module, inputs, output):
            x = inputs[0].detach().float()
            store[key] = x.abs().mean(dim=(0, 1)).cpu()
        return _hook

    def make_output_hook(store, key):
        def _hook(module, inputs, output):
            x = output.detach().float()
            store[key] = x.abs().mean(dim=(0, 1)).cpu()
        return _hook

    for layer_idx, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.q_proj.register_forward_hook(make_input_hook(attn_acts, ('attn', layer_idx))))
        hooks.append(layer.mlp.gate_proj.register_forward_hook(make_output_hook(mlp_acts, ('mlp', layer_idx))))

    for p in model.parameters():
        p.requires_grad_(True)
    model.zero_grad()

    if logger is not None:
        logger.log('[hybrid] start forward/backward for calibration batch')
    loss = model(calibration, labels=calibration).loss
    if logger is not None:
        logger.log(f'[hybrid] calibration loss = {loss.item():.6f}')
    loss.backward()
    if logger is not None:
        logger.log('[hybrid] backward finished, activation statistics captured')

    imp = SAAPImportance(
        vector_reduction='mean',
        element_reduction='sum',
        taylor=taylor,
        align_scores=False,
    )

    attn_scores = []
    mlp_scores = []
    for layer_idx, layer in enumerate(model.model.layers):
        attn_group = [
            (type('dep', (), {'target': type('tgt', (), {'module': layer.self_attn.o_proj}), 'handler': None}), None)
        ]
        mlp_group = [
            (type('dep', (), {'target': type('tgt', (), {'module': layer.mlp.down_proj}), 'handler': None}), None)
        ]
        del attn_group, mlp_group

        q = layer.self_attn.q_proj.weight
        k = layer.self_attn.k_proj.weight
        v = layer.self_attn.v_proj.weight
        o = layer.self_attn.o_proj.weight
        head_dim = layer.self_attn.head_dim
        num_heads = q.shape[0] // head_dim

        attn_mag = (
            q.detach().float().abs().view(num_heads, head_dim, -1).sum(dim=(1, 2)) +
            k.detach().float().abs().view(num_heads, head_dim, -1).sum(dim=(1, 2)) +
            v.detach().float().abs().view(num_heads, head_dim, -1).sum(dim=(1, 2)) +
            o.detach().float().abs().t().contiguous().view(num_heads, head_dim, -1).sum(dim=(1, 2))
        )
        attn_taylor = torch.zeros_like(attn_mag)
        if q.grad is not None and k.grad is not None and v.grad is not None and o.grad is not None:
            attn_taylor = (
                (q.detach().float() * q.grad.detach().float()).abs().view(num_heads, head_dim, -1).sum(dim=(1, 2)) +
                (k.detach().float() * k.grad.detach().float()).abs().view(num_heads, head_dim, -1).sum(dim=(1, 2)) +
                (v.detach().float() * v.grad.detach().float()).abs().view(num_heads, head_dim, -1).sum(dim=(1, 2)) +
                (o.detach().float() * o.grad.detach().float()).abs().t().contiguous().view(num_heads, head_dim, -1).sum(dim=(1, 2))
            )
        attn_act = attn_acts.get(('attn', layer_idx), torch.ones(q.shape[1], dtype=torch.float32))
        attn_act = attn_act.to(q.device)
        attn_act = attn_act.view(num_heads, head_dim).mean(dim=1)
        attn_score = (
            args_like.hybrid_mag_weight * attn_mag +
            args_like.hybrid_taylor_weight * attn_taylor +
            args_like.hybrid_act_weight * attn_act
        )
        attn_scores.append(attn_score.cpu())

        gate = layer.mlp.gate_proj.weight
        up = layer.mlp.up_proj.weight
        down = layer.mlp.down_proj.weight
        hidden_units = gate.shape[0]
        mlp_mag = gate.detach().float().abs().sum(dim=1) + up.detach().float().abs().sum(dim=1) + down.detach().float().abs().sum(dim=0)
        mlp_taylor = torch.zeros_like(mlp_mag)
        if gate.grad is not None and up.grad is not None and down.grad is not None:
            mlp_taylor = (
                (gate.detach().float() * gate.grad.detach().float()).abs().sum(dim=1) +
                (up.detach().float() * up.grad.detach().float()).abs().sum(dim=1) +
                (down.detach().float() * down.grad.detach().float()).abs().sum(dim=0)
            )
        mlp_act = mlp_acts.get(('mlp', layer_idx), torch.ones(gate.shape[1], dtype=torch.float32))
        mlp_act = mlp_act.to(gate.device)
        mlp_score = (
            args_like.hybrid_mag_weight * mlp_mag +
            args_like.hybrid_taylor_weight * mlp_taylor +
            args_like.hybrid_act_weight * mlp_act
        )
        mlp_scores.append(mlp_score.cpu())

    for h in hooks:
        h.remove()

    model.zero_grad()
    for _, p in model.named_parameters():
        p.grad = None

    return torch.stack(attn_scores), torch.stack(mlp_scores)


def _prune_attention_heads(layer: LlamaAttention, keep_mask: torch.Tensor):
    head_dim = layer.head_dim
    keep_heads = keep_mask.nonzero(as_tuple=False).view(-1).tolist()
    keep_idx = []
    for h in keep_heads:
        keep_idx.extend(list(range(h * head_dim, (h + 1) * head_dim)))
    keep_idx = torch.tensor(keep_idx, device=layer.q_proj.weight.device, dtype=torch.long)

    for proj in [layer.q_proj, layer.k_proj, layer.v_proj]:
        proj.out_features = keep_idx.numel()
        proj.weight = nn.Parameter(proj.weight.data.index_select(0, keep_idx))
        if proj.bias is not None:
            proj.bias = nn.Parameter(proj.bias.data.index_select(0, keep_idx))

    layer.o_proj.in_features = keep_idx.numel()
    layer.o_proj.weight = nn.Parameter(layer.o_proj.weight.data.index_select(1, keep_idx))
    layer.num_heads = len(keep_heads)
    return len(keep_heads)


def _prune_mlp_neurons(layer: LlamaMLP, keep_mask: torch.Tensor):
    keep_idx = keep_mask.nonzero(as_tuple=False).view(-1).to(layer.gate_proj.weight.device)

    for proj in [layer.gate_proj, layer.up_proj]:
        proj.out_features = keep_idx.numel()
        proj.weight = nn.Parameter(proj.weight.data.index_select(0, keep_idx))
        if proj.bias is not None:
            proj.bias = nn.Parameter(proj.bias.data.index_select(0, keep_idx))

    layer.down_proj.in_features = keep_idx.numel()
    layer.down_proj.weight = nn.Parameter(layer.down_proj.weight.data.index_select(1, keep_idx))
    return keep_idx.numel()


def run_saap_struct_hybrid(model, tokenizer, args, logger=None):
    t0 = time.time()
    attn_metric, mlp_metric = _collect_saap_group_scores(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        num_examples=args.hybrid_nsamples,
        seq_len=args.calibration_seq_len,
        taylor=args.taylor,
        logger=logger,
        args_like=args,
    )

    attn_metric = _standardize_rows(attn_metric)
    mlp_metric = _standardize_rows(mlp_metric)
    if logger is not None:
        logger.log(f'[hybrid] attn_metric shape={tuple(attn_metric.shape)}, mlp_metric shape={tuple(mlp_metric.shape)}')

    n_layers = attn_metric.shape[0]
    layer_pos = torch.linspace(-1.0, 1.0, steps=n_layers).abs()
    layer_scale = 1.0 - args.hybrid_edge_protect * layer_pos
    layer_scale = layer_scale.clamp_min(0.5)
    attn_metric = attn_metric * layer_scale.unsqueeze(1)
    mlp_metric = mlp_metric * layer_scale.unsqueeze(1)

    threshold = _compression_balanced_threshold(attn_metric, mlp_metric, args.pruning_ratio)

    if logger is not None:
        logger.log(f'[hybrid] global threshold = {threshold.item():.6f}')
        logger.log(f'[hybrid] layer protection scale = {layer_scale.tolist()}')

    kept_heads = []
    kept_neurons = []
    for layer_idx, layer in enumerate(model.model.layers):
        if logger is not None and (layer_idx % 4 == 0 or layer_idx == len(model.model.layers) - 1):
            logger.log(f'[hybrid] pruning layer {layer_idx}/{len(model.model.layers)-1}')
        attn_keep = (attn_metric[layer_idx] > threshold)
        mlp_keep = (mlp_metric[layer_idx] > threshold)

        min_heads = min(args.hybrid_min_heads, attn_metric[layer_idx].numel())
        min_neurons = min(args.hybrid_min_mlp_neurons, mlp_metric[layer_idx].numel())

        if attn_keep.sum().item() < min_heads:
            topk = torch.topk(attn_metric[layer_idx], k=min_heads).indices
            attn_keep[topk] = True
        if mlp_keep.sum().item() < min_neurons:
            topk = torch.topk(mlp_metric[layer_idx], k=min_neurons).indices
            mlp_keep[topk] = True

        kh = _prune_attention_heads(layer.self_attn, attn_keep.to(layer.self_attn.q_proj.weight.device))
        kn = _prune_mlp_neurons(layer.mlp, mlp_keep.to(layer.mlp.gate_proj.weight.device))
        kept_heads.append(kh)
        kept_neurons.append(kn)

    if logger is not None:
        logger.log(f'[hybrid] kept heads per layer: {kept_heads}')
        logger.log(f'[hybrid] kept mlp neurons per layer: {kept_neurons[:8]} ...')
        logger.log(f'[hybrid] done in {time.time() - t0:.1f}s')

    return {
        'threshold': float(threshold.item()),
        'kept_heads': kept_heads,
        'kept_neurons': kept_neurons,
        'layer_scale': layer_scale.tolist(),
    }
