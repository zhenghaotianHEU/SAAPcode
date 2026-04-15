import math
import time
import torch
from torch import nn

from core.datasets.example_samples import get_examples


def _angular_distance(x_in: torch.Tensor, x_out: torch.Tensor, eps: float = 1e-6):
    x_in = x_in.float()
    x_out = x_out.float()
    num = (x_in * x_out).sum(dim=-1)
    denom = x_in.norm(dim=-1).clamp_min(eps) * x_out.norm(dim=-1).clamp_min(eps)
    cos = (num / denom).clamp(-1.0 + eps, 1.0 - eps)
    return torch.arccos(cos) / math.pi


def _assign_layer_budgets(coarse_scores: torch.Tensor, target_pruning_ratio: float, min_keep_ratio: float = 0.35, temperature: float = 4.0):
    centered = coarse_scores - coarse_scores.mean()
    norm = torch.sigmoid(temperature * centered)
    keep = norm / norm.sum() * len(norm) * (1.0 - target_pruning_ratio)
    keep = keep.clamp(min=min_keep_ratio, max=1.0)
    return keep


def _prune_mlp_neurons(layer, keep_idx: torch.Tensor):
    keep_idx = keep_idx.to(layer.gate_proj.weight.device)
    for proj in [layer.gate_proj, layer.up_proj]:
        proj.out_features = keep_idx.numel()
        proj.weight = nn.Parameter(proj.weight.data.index_select(0, keep_idx))
        if proj.bias is not None:
            proj.bias = nn.Parameter(proj.bias.data.index_select(0, keep_idx))
    layer.down_proj.in_features = keep_idx.numel()
    layer.down_proj.weight = nn.Parameter(layer.down_proj.weight.data.index_select(1, keep_idx))
    return keep_idx.numel()


def run_cfsp_ffn_pruner(model, tokenizer, args, logger=None):
    t0 = time.time()
    calibration = get_examples(args.dataset, tokenizer, args.num_examples, seq_len=args.calibration_seq_len).to(args.device)

    if logger is not None:
        logger.log(f'[cfsp] calibration tensor shape = {tuple(calibration.shape)}')

    layer_inputs = {}
    layer_outputs = {}
    gate_outputs = {}
    hooks = []

    def make_block_pre_hook(layer_idx):
        def _hook(module, inputs):
            layer_inputs[layer_idx] = inputs[0].detach().float().cpu()
        return _hook

    def make_block_hook(layer_idx):
        def _hook(module, inputs, outputs):
            x = outputs[0] if isinstance(outputs, tuple) else outputs
            layer_outputs[layer_idx] = x.detach().float().cpu()
        return _hook

    def make_gate_hook(layer_idx):
        def _hook(module, inputs, outputs):
            gate_outputs[layer_idx] = outputs.detach().float().cpu()
        return _hook

    for layer_idx, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_pre_hook(make_block_pre_hook(layer_idx)))
        hooks.append(layer.register_forward_hook(make_block_hook(layer_idx)))
        hooks.append(layer.mlp.gate_proj.register_forward_hook(make_gate_hook(layer_idx)))

    with torch.no_grad():
        _ = model(calibration)

    for h in hooks:
        h.remove()

    coarse_scores = []
    kept_neurons = []
    layer_keep_ratios = []

    for layer_idx in range(len(model.model.layers)):
        coarse_scores.append(_angular_distance(layer_inputs[layer_idx], layer_outputs[layer_idx]).mean())
    coarse_scores = torch.stack(coarse_scores)
    keep_ratios = _assign_layer_budgets(coarse_scores, args.pruning_ratio, min_keep_ratio=args.cfsp_min_keep_ratio, temperature=args.cfsp_budget_temperature)

    if logger is not None:
        logger.log(f'[cfsp] coarse_scores={coarse_scores.tolist()}')
        logger.log(f'[cfsp] keep_ratios={keep_ratios.tolist()}')

    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx < args.block_mlp_layer_start or layer_idx >= args.block_mlp_layer_end:
            kept_neurons.append(layer.mlp.gate_proj.weight.shape[0])
            layer_keep_ratios.append(1.0)
            continue
        if layer_idx < args.protect_first_n_layers or (args.protect_last_layer and layer_idx == len(model.model.layers) - 1):
            kept_neurons.append(layer.mlp.gate_proj.weight.shape[0])
            layer_keep_ratios.append(1.0)
            continue

        gate = layer.mlp.gate_proj.weight.detach().float().cpu()
        up = layer.mlp.up_proj.weight.detach().float().cpu()
        down = layer.mlp.down_proj.weight.detach().float().cpu()
        act = gate_outputs[layer_idx].abs().mean(dim=(0, 1))
        gate_score = gate.abs().sum(dim=1)
        up_score = up.abs().sum(dim=1)
        down_score = down.abs().sum(dim=0)
        fine_score = (0.45 * gate_score + 0.45 * up_score + 0.10 * down_score) * act

        raw_keep = fine_score.numel() * keep_ratios[layer_idx].item()
        keep_n = max(128, int(round(raw_keep / 128) * 128))
        keep_n = min(keep_n, fine_score.numel())
        topk = torch.topk(fine_score, k=keep_n).indices
        kept = _prune_mlp_neurons(layer.mlp, topk)
        kept_neurons.append(kept)
        layer_keep_ratios.append(keep_n / fine_score.numel())

        if logger is not None:
            logger.log(f'[cfsp] layer={layer_idx} | coarse={coarse_scores[layer_idx].item():.6f} | keep_ratio={layer_keep_ratios[-1]:.4f} | kept_neurons={kept}')

    if logger is not None:
        logger.log(f'[cfsp] done in {time.time() - t0:.1f}s')

    return {
        'coarse_scores': [float(x) for x in coarse_scores.tolist()],
        'keep_ratios': [float(x) for x in layer_keep_ratios],
        'kept_neurons': kept_neurons,
    }
