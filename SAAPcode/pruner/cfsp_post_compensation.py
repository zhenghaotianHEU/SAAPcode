import torch
import torch.nn as nn


def flap_bias_compensation_mlp(layer, keep_idx: torch.Tensor, mlp_mean_inp: torch.Tensor, logger=None, layer_idx=None, alpha: float = 0.05):
    if mlp_mean_inp is None or keep_idx is None:
        return False
    try:
        device = layer.down_proj.weight.device
        mean_inp = mlp_mean_inp.detach().to(device='cpu', dtype=torch.float32).view(-1)
        keep_idx_cpu = keep_idx.detach().to(device='cpu', dtype=torch.long).view(-1)
        full_dim = int(layer.gate_proj.weight.shape[0])
        down_in_features = int(layer.down_proj.weight.shape[1])
        down_out_features = int(layer.down_proj.weight.shape[0])

        if logger is not None:
            keep_min = int(keep_idx_cpu.min().item()) if keep_idx_cpu.numel() > 0 else -1
            keep_max = int(keep_idx_cpu.max().item()) if keep_idx_cpu.numel() > 0 else -1
            logger.log(
                f'[flap_bias_comp_mlp][debug] layer={layer_idx} | '
                f'gate_proj_shape={tuple(layer.gate_proj.weight.shape)} | '
                f'up_proj_shape={tuple(layer.up_proj.weight.shape)} | '
                f'down_proj_shape={tuple(layer.down_proj.weight.shape)} | '
                f'mean_inp_shape={tuple(mean_inp.shape)} | '
                f'full_dim={full_dim} | down_in_features={down_in_features} | down_out_features={down_out_features} | '
                f'keep_numel={keep_idx_cpu.numel()} | keep_min={keep_min} | keep_max={keep_max}'
            )

        if keep_idx_cpu.numel() == 0:
            return False
        if int(keep_idx_cpu.min().item()) < 0 or int(keep_idx_cpu.max().item()) >= full_dim:
            raise RuntimeError(f'keep_idx_oob min={int(keep_idx_cpu.min().item())} max={int(keep_idx_cpu.max().item())} full_dim={full_dim}')

        keep_mask_cpu = torch.zeros(full_dim, dtype=torch.bool, device='cpu')
        keep_mask_cpu[keep_idx_cpu] = True
        pruned_mask_cpu = ~keep_mask_cpu
        pruned_n = int(pruned_mask_cpu.sum().item())
        if pruned_n == 0:
            return False

        if mean_inp.numel() != full_dim:
            raise RuntimeError(f'mean_inp_mismatch mean_inp_numel={mean_inp.numel()} full_dim={full_dim}')
        if down_in_features != full_dim:
            raise RuntimeError(f'downproj_infeature_mismatch down_in_features={down_in_features} full_dim={full_dim}')

        output_weight = layer.down_proj.weight.data.detach().to(device='cpu', dtype=torch.float32)
        removed_mean = mean_inp * pruned_mask_cpu.float()
        output_bias = float(alpha) * torch.matmul(removed_mean, output_weight.transpose(0, 1))
        output_bias = output_bias.to(device=device)
        if layer.down_proj.bias is None:
            layer.down_proj.bias = nn.Parameter(output_bias.to(dtype=layer.down_proj.weight.dtype))
        else:
            layer.down_proj.bias = nn.Parameter(output_bias.to(device=device, dtype=layer.down_proj.bias.dtype))
        if logger is not None:
            logger.log(f'[flap_bias_comp_mlp] layer={layer_idx} | pruned={pruned_n} | alpha={float(alpha):.4f} | status=1')
        return True
    except Exception as e:
        if logger is not None:
            logger.log(f'[flap_bias_comp_mlp] layer={layer_idx} | status=0 | err={e}')
        return False


def flap_bias_compensation_attn(layer, keep_mask_heads: torch.Tensor, attn_mean_inp: torch.Tensor, logger=None, layer_idx=None):
    if attn_mean_inp is None or keep_mask_heads is None:
        return False
    try:
        device = layer.o_proj.weight.device
        head_dim = int(layer.head_dim)
        keep_mask_heads = keep_mask_heads.to(device=device).bool()
        attn_mask = keep_mask_heads.repeat_interleave(head_dim)
        pruned_mask = ~attn_mask
        if pruned_mask.sum().item() == 0:
            return False

        mean_inp = attn_mean_inp.detach().to(device=device, dtype=torch.float32).view(-1)
        output_weight = layer.o_proj.weight.data.detach().float()
        removed_mean = mean_inp * pruned_mask.float()
        output_bias = torch.matmul(removed_mean, output_weight.transpose(0, 1))
        if layer.o_proj.bias is None:
            layer.o_proj.bias = nn.Parameter(output_bias.to(dtype=layer.o_proj.weight.dtype))
        else:
            layer.o_proj.bias = nn.Parameter(output_bias.to(device=device, dtype=layer.o_proj.bias.dtype))
        if logger is not None:
            logger.log(f'[flap_bias_comp_attn] layer={layer_idx} | pruned={int(pruned_mask.sum().item())} | status=1')
        return True
    except Exception as e:
        if logger is not None:
            logger.log(f'[flap_bias_comp_attn] layer={layer_idx} | status=0 | err={e}')
        return False


def post_taylor_swap_keep_idx(fine_score: torch.Tensor, keep_idx: torch.Tensor, taylor_score: torch.Tensor, swap_topk: int = 256, swap_margin: float = 1.05, logger=None, layer_idx=None):
    if taylor_score is None or keep_idx is None or keep_idx.numel() == 0:
        return keep_idx
    total = fine_score.numel()
    keep_mask = torch.zeros(total, dtype=torch.bool, device=keep_idx.device if keep_idx.is_cuda else fine_score.device)
    keep_idx = keep_idx.long().to(keep_mask.device)
    fine_score = fine_score.to(keep_mask.device)
    taylor_score = taylor_score.to(keep_mask.device)
    keep_mask[keep_idx] = True
    pruned_idx = (~keep_mask).nonzero(as_tuple=False).view(-1)
    if pruned_idx.numel() == 0:
        return keep_idx.cpu() if keep_idx.device.type != 'cpu' else keep_idx

    kept_fine = fine_score.index_select(0, keep_idx)
    pruned_fine = fine_score.index_select(0, pruned_idx)
    boundary_keep_n = min(int(swap_topk), keep_idx.numel())
    boundary_prune_n = min(int(swap_topk), pruned_idx.numel())
    if boundary_keep_n <= 0 or boundary_prune_n <= 0:
        return keep_idx.cpu() if keep_idx.device.type != 'cpu' else keep_idx

    keep_boundary_local = torch.topk(kept_fine, k=boundary_keep_n, largest=False).indices
    prune_boundary_local = torch.topk(pruned_fine, k=boundary_prune_n, largest=True).indices
    keep_boundary = keep_idx.index_select(0, keep_boundary_local)
    prune_boundary = pruned_idx.index_select(0, prune_boundary_local)

    keep_taylor = taylor_score.index_select(0, keep_boundary)
    prune_taylor = taylor_score.index_select(0, prune_boundary)
    keep_order = torch.argsort(keep_taylor, descending=False)
    prune_order = torch.argsort(prune_taylor, descending=True)

    swaps = []
    used_keep = set()
    used_prune = set()
    margin = max(1.0, float(swap_margin))
    pair_n = min(keep_order.numel(), prune_order.numel())
    for pos in range(pair_n):
        k_local = int(keep_order[pos].item())
        p_local = int(prune_order[pos].item())
        if k_local in used_keep or p_local in used_prune:
            continue
        k_idx = int(keep_boundary[k_local].item())
        p_idx = int(prune_boundary[p_local].item())
        k_t = float(taylor_score[k_idx].item())
        p_t = float(taylor_score[p_idx].item())
        if p_t > k_t * margin:
            swaps.append((k_idx, p_idx, k_t, p_t))
            used_keep.add(k_local)
            used_prune.add(p_local)

    if not swaps:
        if logger is not None:
            logger.log(f'[post_taylor_swap] layer={layer_idx} | swaps=0 | topk={boundary_keep_n} | margin={margin:.3f}')
        return keep_idx.cpu() if keep_idx.device.type != 'cpu' else keep_idx

    final_keep = set(int(x) for x in keep_idx.tolist())
    for k_idx, p_idx, _, _ in swaps:
        if k_idx in final_keep:
            final_keep.remove(k_idx)
            final_keep.add(p_idx)
    final_keep = torch.tensor(sorted(final_keep), dtype=torch.long)
    if logger is not None:
        best = swaps[0]
        logger.log(f'[post_taylor_swap] layer={layer_idx} | swaps={len(swaps)} | topk={boundary_keep_n} | margin={margin:.3f} | sample_swap=keep{best[0]}:{best[2]:.4f}->prune{best[1]}:{best[3]:.4f}')
    return final_keep



def post_taylor_swap_keep_mask(head_score: torch.Tensor, keep_mask: torch.Tensor, taylor_score: torch.Tensor, swap_topk: int = 8, swap_margin: float = 1.03, logger=None, layer_idx=None):
    if taylor_score is None or keep_mask is None or keep_mask.numel() == 0:
        return keep_mask
    work_device = keep_mask.device if keep_mask.is_cuda else head_score.device
    head_score = head_score.to(work_device)
    keep_mask = keep_mask.to(work_device).bool()
    taylor_score = taylor_score.to(work_device)
    keep_idx = keep_mask.nonzero(as_tuple=False).view(-1)
    pruned_idx = (~keep_mask).nonzero(as_tuple=False).view(-1)
    if keep_idx.numel() == 0 or pruned_idx.numel() == 0:
        return keep_mask.cpu() if keep_mask.device.type != 'cpu' else keep_mask

    kept_score = head_score.index_select(0, keep_idx)
    pruned_score = head_score.index_select(0, pruned_idx)
    boundary_keep_n = min(int(swap_topk), keep_idx.numel())
    boundary_prune_n = min(int(swap_topk), pruned_idx.numel())
    if boundary_keep_n <= 0 or boundary_prune_n <= 0:
        return keep_mask.cpu() if keep_mask.device.type != 'cpu' else keep_mask

    keep_boundary_local = torch.topk(kept_score, k=boundary_keep_n, largest=False).indices
    prune_boundary_local = torch.topk(pruned_score, k=boundary_prune_n, largest=True).indices
    keep_boundary = keep_idx.index_select(0, keep_boundary_local)
    prune_boundary = pruned_idx.index_select(0, prune_boundary_local)

    keep_taylor = taylor_score.index_select(0, keep_boundary)
    prune_taylor = taylor_score.index_select(0, prune_boundary)
    keep_order = torch.argsort(keep_taylor, descending=False)
    prune_order = torch.argsort(prune_taylor, descending=True)

    swaps = []
    used_keep = set()
    used_prune = set()
    margin = max(1.0, float(swap_margin))
    pair_n = min(keep_order.numel(), prune_order.numel())
    for pos in range(pair_n):
        k_local = int(keep_order[pos].item())
        p_local = int(prune_order[pos].item())
        if k_local in used_keep or p_local in used_prune:
            continue
        k_idx = int(keep_boundary[k_local].item())
        p_idx = int(prune_boundary[p_local].item())
        k_t = float(taylor_score[k_idx].item())
        p_t = float(taylor_score[p_idx].item())
        if p_t > k_t * margin:
            swaps.append((k_idx, p_idx, k_t, p_t))
            used_keep.add(k_local)
            used_prune.add(p_local)

    if not swaps:
        if logger is not None:
            logger.log(f'[attn_post_taylor_swap] layer={layer_idx} | swaps=0 | topk={boundary_keep_n} | margin={margin:.3f}')
        return keep_mask.cpu() if keep_mask.device.type != 'cpu' else keep_mask

    new_mask = keep_mask.clone().bool()
    for k_idx, p_idx, _, _ in swaps:
        new_mask[k_idx] = False
        new_mask[p_idx] = True
    if logger is not None:
        best = swaps[0]
        logger.log(f'[attn_post_taylor_swap] layer={layer_idx} | swaps={len(swaps)} | topk={boundary_keep_n} | margin={margin:.3f} | sample_swap=keep{best[0]}:{best[2]:.4f}->prune{best[1]}:{best[3]:.4f}')
    return new_mask.cpu() if new_mask.device.type != 'cpu' else new_mask



def post_reconstruct_mlp_downproj(layer, layer_input_cpu: torch.Tensor, max_tokens: int = 1024, logger=None, layer_idx=None, alpha: float = 0.05, max_delta_ratio: float = 0.03):
    if layer_input_cpu is None:
        return False
    device = layer.down_proj.weight.device
    try:
        x = layer_input_cpu.to(device=device, dtype=torch.float32)
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        if x.numel() == 0:
            return False
        if max_tokens > 0 and x.shape[0] > max_tokens:
            x = x[:max_tokens]

        with torch.no_grad():
            proj_dtype = layer.gate_proj.weight.dtype
            old_weight_fp32 = layer.down_proj.weight.data.detach().float()

            x_proj = x.to(dtype=proj_dtype)
            gate_out = layer.gate_proj(x_proj).float()
            up_out = layer.up_proj(x_proj).float()
            hidden = (layer.act_fn(gate_out) * up_out).float()
            current_out = torch.matmul(hidden, old_weight_fp32.transpose(0, 1))
            hidden_t = hidden.transpose(0, 1).contiguous()
            gram = hidden_t @ hidden
            rhs = hidden_t @ current_out
            reg = 1e-4 * torch.eye(gram.shape[0], device=device, dtype=torch.float32)
            solved = torch.linalg.solve(gram + reg, rhs)
            recon_weight_fp32 = solved.transpose(0, 1).contiguous().float()

            old_norm = old_weight_fp32.norm().clamp_min(1e-8)
            recon_norm = recon_weight_fp32.norm().clamp_min(1e-8)
            recon_weight_fp32 = recon_weight_fp32 * (old_norm / recon_norm)

            delta = recon_weight_fp32 - old_weight_fp32
            delta_norm = delta.norm().clamp_min(1e-8)
            target_delta_norm = old_norm * float(max_delta_ratio)
            if delta_norm > target_delta_norm:
                delta = delta * (target_delta_norm / delta_norm)

            new_weight_fp32 = old_weight_fp32 + float(alpha) * delta
            if not torch.isfinite(new_weight_fp32).all():
                raise RuntimeError('non-finite new_weight_fp32')

            new_weight = new_weight_fp32.to(dtype=layer.down_proj.weight.dtype)
            if not torch.isfinite(new_weight.float()).all():
                raise RuntimeError('non-finite casted new_weight')

            layer.down_proj.weight = nn.Parameter(new_weight)
        if logger is not None:
            logger.log(f'[post_reconstruct_ffn] layer={layer_idx} | tokens={x.shape[0]} | in_features={hidden.shape[1]} | out_features={current_out.shape[1]} | alpha={float(alpha):.2f} | max_delta_ratio={float(max_delta_ratio):.2f} | status=1')
        return True
    except Exception as e:
        if logger is not None:
            logger.log(f'[post_reconstruct_ffn] layer={layer_idx} | status=0 | err={e}')
        return False
