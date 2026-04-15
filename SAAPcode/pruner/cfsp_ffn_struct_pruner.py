import gc
import math
import time
import os
import json
import torch
from torch import nn

from core.datasets.example_samples import get_examples
try:
    from core.cpp_ops.loader import load_cpp_ops
except Exception:
    def load_cpp_ops(*args, **kwargs):
        return None
from core.pruner.stat_collectors import InputNormCollector, InputStatCollector
from core.pruner.cfsp_post_compensation import (
    post_taylor_swap_keep_idx as _post_taylor_swap_keep_idx,
    post_taylor_swap_keep_mask as _post_taylor_swap_keep_mask,
    post_reconstruct_mlp_downproj as _post_reconstruct_mlp_downproj,
)


def _angular_distance(x_in: torch.Tensor, x_out: torch.Tensor, eps: float = 1e-6):
    x_in = x_in.float()
    x_out = x_out.float()
    num = (x_in * x_out).sum(dim=-1)
    denom = x_in.norm(dim=-1).clamp_min(eps) * x_out.norm(dim=-1).clamp_min(eps)
    cos = (num / denom).clamp(-1.0 + eps, 1.0 - eps)
    return torch.arccos(cos) / math.pi


def _downsample_layer_inputs_for_rerank(layer_input: torch.Tensor, max_tokens: int = 4096):
    if layer_input is None:
        return None
    if max_tokens is None or max_tokens <= 0:
        return layer_input
    if layer_input.dim() != 3:
        return layer_input

    bsz, seq_len, hidden = layer_input.shape
    total_tokens = bsz * seq_len
    if total_tokens <= max_tokens:
        return layer_input

    step = max(1, int(math.ceil(total_tokens / max_tokens)))
    flat = layer_input.reshape(total_tokens, hidden)
    sampled = flat[::step]
    if sampled.shape[0] > max_tokens:
        sampled = sampled[:max_tokens]
    return sampled.unsqueeze(0).contiguous()


def _angular_distance_mean(x_in: torch.Tensor, x_out: torch.Tensor, cpp_ops=None, eps: float = 1e-6):
    if cpp_ops is not None:
        try:
            return cpp_ops.compute_angular_distance_mean(x_in, x_out, float(eps))
        except Exception:
            pass
    return _angular_distance(x_in, x_out, eps=eps).mean()


def _compute_attention_head_scores(q_in: torch.Tensor, attn_out: torch.Tensor, head_dim: int, cpp_ops=None, eps: float = 1e-6):
    if cpp_ops is not None:
        try:
            return cpp_ops.compute_attention_head_scores(q_in, attn_out, int(head_dim), float(eps))
        except Exception:
            pass

    total_heads = q_in.shape[-1] // head_dim
    head_act = q_in.abs().mean(dim=(0, 1)).view(total_heads, head_dim).mean(dim=1)
    head_fluc = attn_out.float().std(dim=1).mean(dim=0).view(total_heads, head_dim).mean(dim=1)
    head_act = head_act / head_act.mean().clamp_min(eps)
    head_fluc = head_fluc / head_fluc.mean().clamp_min(eps)
    head_score = 0.60 * head_act + 0.40 * head_fluc
    return head_act, head_fluc, head_score


def _compute_mlp_wanda_flap_scores(
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    input_rms: torch.Tensor = None,
    gate_act_mean: torch.Tensor = None,
    gate_act_std: torch.Tensor = None,
    eps: float = 1e-6,
):
    gate_norm = gate_w.abs().mean(dim=1)
    up_norm = up_w.abs().mean(dim=1)
    down_norm = down_w.abs().mean(dim=0)

    if gate_act_mean is None:
        act_score = torch.ones_like(gate_norm)
    else:
        act_score = gate_act_mean / gate_act_mean.mean().clamp_min(eps)

    if gate_act_std is None:
        fluc_score = torch.ones_like(gate_norm)
    else:
        fluc_score = gate_act_std / gate_act_std.mean().clamp_min(eps)

    down_support = down_norm / down_norm.mean().clamp_min(eps)

    if input_rms is None:
        input_support = down_support
    else:
        input_support = input_rms / input_rms.mean().clamp_min(eps)

    wanda_score = (gate_norm + up_norm).clamp_min(eps) * torch.sqrt(act_score.clamp_min(eps))
    wanda_score = wanda_score * torch.sqrt(down_support.clamp_min(eps))

    coarse_mix = (
        0.50 * (wanda_score / wanda_score.mean().clamp_min(eps))
        + 0.30 * act_score
        + 0.20 * fluc_score
    )

    return {
        'wanda_score': wanda_score,
        'act_score': act_score,
        'fluc_score': fluc_score,
        'input_support': input_support,
        'coarse_mix': coarse_mix,
    }


def _compute_attention_head_wanda_scores(
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    v_w: torch.Tensor,
    o_w: torch.Tensor,
    head_dim: int,
    head_act: torch.Tensor,
    head_fluc: torch.Tensor,
    eps: float = 1e-6,
):
    total_heads = q_w.shape[0] // head_dim

    q_norm = q_w.abs().mean(dim=1).view(total_heads, head_dim).mean(dim=1)
    k_norm = k_w.abs().mean(dim=1).view(total_heads, head_dim).mean(dim=1)
    v_norm = v_w.abs().mean(dim=1).view(total_heads, head_dim).mean(dim=1)
    o_in_norm = o_w.abs().mean(dim=0).view(total_heads, head_dim).mean(dim=1)

    struct_norm = q_norm + k_norm + v_norm + o_in_norm
    struct_norm = struct_norm / struct_norm.mean().clamp_min(eps)

    head_act = head_act / head_act.mean().clamp_min(eps)
    head_fluc = head_fluc / head_fluc.mean().clamp_min(eps)

    head_wanda = struct_norm * torch.sqrt(head_act.clamp_min(eps))
    head_mix = 0.45 * head_wanda + 0.35 * head_act + 0.20 * head_fluc

    return {
        'head_wanda': head_wanda,
        'head_mix': head_mix,
    }


def _topk_overlap_ratio(score_a: torch.Tensor, score_b: torch.Tensor, ratio: float = 0.10):
    n = score_a.numel()
    if n <= 0:
        return 0.0
    k = max(1, int(round(n * ratio)))
    idx_a = set(torch.topk(score_a, k=k, largest=True).indices.tolist())
    idx_b = set(torch.topk(score_b, k=k, largest=True).indices.tolist())
    inter = len(idx_a & idx_b)
    return inter / max(1, k)


def _coarse_weight_from_overlap(
    overlap_top20: float,
    low_thresh: float = 0.72,
    high_thresh: float = 0.84,
    min_w: float = 0.02,
    max_w: float = 0.18,
):
    x = float(overlap_top20)
    if x <= low_thresh:
        return max_w
    if x >= high_thresh:
        return min_w
    ratio = (high_thresh - x) / max(1e-6, high_thresh - low_thresh)
    return min_w + ratio * (max_w - min_w)


def _allocate_prunable_keep_ratios_continuous(
    prunable_scores: torch.Tensor,
    target_pruning_ratio: float,
    min_keep_ratio: float = 0.20,
    layer_sensitivity: torch.Tensor = None,
):
    n = len(prunable_scores)
    target_keep = 1.0 - target_pruning_ratio
    if n <= 1:
        return torch.ones_like(prunable_scores) * target_keep

    scores = prunable_scores.float().reshape(-1)
    rank = torch.argsort(scores, descending=True)

    centered = scores - scores.mean()
    saliency = centered / scores.std().clamp_min(1e-6)
    saliency = torch.tanh(1.10 * saliency)

    pos = torch.linspace(0.0, 1.0, steps=n, device=scores.device)
    late_bias = torch.exp(-((pos - 0.90) / 0.16) ** 2)
    mid_penalty = torch.exp(-((pos - 0.58) / 0.18) ** 2)
    shape_bias = 0.75 * late_bias - 0.55 * mid_penalty
    shape_bias = shape_bias - shape_bias.mean()

    if layer_sensitivity is not None:
        sens = layer_sensitivity.float().reshape(-1).to(scores.device)
        sens = sens - sens.mean()
        sens = sens / sens.std().clamp_min(1e-6)
        sens = torch.tanh(0.90 * sens)
        adaptive = 0.40 * saliency + 0.45 * sens + 0.15 * shape_bias
    else:
        adaptive = 0.60 * saliency + 0.40 * shape_bias
    adaptive = adaptive - adaptive.mean()
    adaptive = adaptive / adaptive.std().clamp_min(1e-6)
    adaptive = torch.tanh(0.95 * adaptive)
    adaptive = adaptive - adaptive.mean()

    keep_floor = float(min_keep_ratio)
    keep_ceiling = 0.98

    layer_pos = torch.linspace(0.0, 1.0, steps=n, device=scores.device)
    late_bonus = torch.exp(-((layer_pos - 0.90) / 0.16) ** 2)
    mid_suppress = torch.exp(-((layer_pos - 0.58) / 0.20) ** 2)
    flap_importance = 0.58 * saliency + 0.30 * shape_bias + 0.12 * (late_bonus - mid_suppress)
    if layer_sensitivity is not None:
        flap_importance = 0.55 * flap_importance + 0.45 * sens
    flap_importance = flap_importance - flap_importance.mean()
    flap_importance = flap_importance / flap_importance.std().clamp_min(1e-6)

    dynamic_span = max(0.012, min(0.045, 0.35 * max(0.0, target_keep - keep_floor) + 0.012))
    flap_response = torch.tanh(0.95 * flap_importance)
    keep = target_keep + dynamic_span * flap_response

    late_shape = 0.010 * late_bonus - 0.007 * mid_suppress
    late_shape = late_shape - late_shape.mean()
    keep = keep + late_shape

    keep = keep.clamp(min=keep_floor, max=keep_ceiling)

    residual = target_keep - keep.mean()
    if abs(float(residual)) > 1e-8:
        if residual < 0:
            reducible = (keep - keep_floor).clamp_min(0.0)
            if float(reducible.sum()) > 1e-8:
                keep = keep + residual * (reducible / reducible.sum().clamp_min(1e-8))
        else:
            increasable = (keep_ceiling - keep).clamp_min(0.0)
            if float(increasable.sum()) > 1e-8:
                keep = keep + residual * (increasable / increasable.sum().clamp_min(1e-8))

    keep = keep.clamp(min=keep_floor, max=keep_ceiling)
    return keep



def _estimate_effective_mlp_pruning_ratio(model, args, logger=None):
    total_params = float(sum(p.numel() for p in model.parameters()))
    protected = set(range(0, min(args.protect_first_n_layers, len(model.model.layers))))
    if args.protect_last_layer:
        protected.add(len(model.model.layers) - 1)

    prunable_mlp_total = 0.0
    prunable_mlp_min_keep = 0.0
    prunable_attn_total = 0.0
    prunable_attn_min_keep = 0.0

    for i, layer in enumerate(model.model.layers):
        gate = float(layer.mlp.gate_proj.weight.numel())
        up = float(layer.mlp.up_proj.weight.numel())
        down = float(layer.mlp.down_proj.weight.numel())
        mlp_total = gate + up + down

        q = float(layer.self_attn.q_proj.weight.numel())
        k = float(layer.self_attn.k_proj.weight.numel())
        v = float(layer.self_attn.v_proj.weight.numel())
        o = float(layer.self_attn.o_proj.weight.numel())
        attn_total = q + k + v + o

        if (i >= args.block_mlp_layer_start and i < args.block_mlp_layer_end) and i not in protected:
            full_dim = float(layer.mlp.gate_proj.weight.shape[0])
            keep_ratio_floor = float(args.cfsp_min_keep_ratio)
            prunable_mlp_total += mlp_total
            prunable_mlp_min_keep += mlp_total * keep_ratio_floor

        if (i >= args.block_attention_layer_start and i < args.block_attention_layer_end) and i not in protected:
            total_heads = float(layer.self_attn.q_proj.weight.shape[0] // layer.self_attn.head_dim)
            attn_keep_floor = max(float(args.cfsp_attn_min_heads) / max(1.0, total_heads), min(0.96, float(args.cfsp_attn_keep_ratio)))
            prunable_attn_total += attn_total
            prunable_attn_min_keep += attn_total * attn_keep_floor

    target_keep_total = total_params * (1.0 - float(args.pruning_ratio))
    fixed_keep_other = total_params - prunable_mlp_total - prunable_attn_total
    remaining_budget_for_mlp = target_keep_total - fixed_keep_other - prunable_attn_min_keep

    if prunable_mlp_total <= 0:
        return float(args.pruning_ratio)

    desired_mlp_keep_ratio = remaining_budget_for_mlp / max(1.0, prunable_mlp_total)
    desired_mlp_keep_ratio = max(float(args.cfsp_min_keep_ratio), min(0.98, desired_mlp_keep_ratio))
    effective_mlp_pruning_ratio = 1.0 - desired_mlp_keep_ratio
    calib_margin = min(0.06, max(0.0, 0.35 * float(args.pruning_ratio) + 0.01))
    effective_mlp_pruning_ratio = effective_mlp_pruning_ratio + calib_margin
    effective_mlp_pruning_ratio = max(0.0, min(0.95, effective_mlp_pruning_ratio))

    if logger is not None:
        logger.log(
            f'[cfsp_flap][budget_calib] total_params={int(total_params)} | prunable_mlp_total={int(prunable_mlp_total)} | '
            f'prunable_attn_total={int(prunable_attn_total)} | fixed_keep_other={int(fixed_keep_other)} | '
            f'attn_min_keep={int(prunable_attn_min_keep)} | target_total_keep={int(target_keep_total)} | '
            f'desired_mlp_keep_ratio={desired_mlp_keep_ratio:.6f} | calib_margin={calib_margin:.6f} | '
            f'effective_mlp_pruning_ratio={effective_mlp_pruning_ratio:.6f}'
        )

    return effective_mlp_pruning_ratio



def _estimate_joint_attn_mlp_keep_budget(model, args, prunable_mlp_layers, prunable_attn_layers, layer_attn_prune_ctx, logger=None):
    total_params = float(sum(p.numel() for p in model.parameters()))
    target_total_keep = total_params * (1.0 - float(args.pruning_ratio))

    prunable_mlp_total = 0.0
    prunable_attn_total = 0.0
    fixed_keep_other = total_params

    for i, layer in enumerate(model.model.layers):
        gate = float(layer.mlp.gate_proj.weight.numel())
        up = float(layer.mlp.up_proj.weight.numel())
        down = float(layer.mlp.down_proj.weight.numel())
        mlp_total = gate + up + down
        q = float(layer.self_attn.q_proj.weight.numel())
        k = float(layer.self_attn.k_proj.weight.numel())
        v = float(layer.self_attn.v_proj.weight.numel())
        o = float(layer.self_attn.o_proj.weight.numel())
        attn_total = q + k + v + o

        if i in prunable_mlp_layers:
            prunable_mlp_total += mlp_total
            fixed_keep_other -= mlp_total
        if i in prunable_attn_layers:
            prunable_attn_total += attn_total
            fixed_keep_other -= attn_total

    attn_floor_keep_params = 0.0
    attn_free_params = 0.0
    attn_weight_floor = 0.0
    attn_weight_free = 0.0
    for layer_idx in prunable_attn_layers:
        layer = model.model.layers[layer_idx]
        total_heads = int(layer_attn_prune_ctx[layer_idx]['total_heads'])
        min_keep_heads = max(int(args.cfsp_attn_min_heads), 1)
        layer_attn = layer.self_attn
        q = float(layer_attn.q_proj.weight.numel())
        k = float(layer_attn.k_proj.weight.numel())
        v = float(layer_attn.v_proj.weight.numel())
        o = float(layer_attn.o_proj.weight.numel())
        attn_total = q + k + v + o
        floor_ratio = min_keep_heads / max(1.0, float(total_heads))
        attn_floor_keep_params += attn_total * floor_ratio
        attn_free_params += attn_total * max(0.0, 1.0 - floor_ratio)
        attn_weight_floor += float(min_keep_heads) * (512.0 / 3.0)
        attn_weight_free += float(max(0, total_heads - min_keep_heads)) * (512.0 / 3.0)

    mlp_floor_keep_params = prunable_mlp_total * float(args.cfsp_min_keep_ratio)
    mlp_free_params = prunable_mlp_total * max(0.0, 1.0 - float(args.cfsp_min_keep_ratio))
    mlp_weight_floor = 0.0
    mlp_weight_free = 0.0
    for layer_idx in prunable_mlp_layers:
        full_dim = float(model.model.layers[layer_idx].mlp.gate_proj.weight.shape[0])
        floor_keep_n = max(128, int(round(full_dim * float(args.cfsp_min_keep_ratio) / 128) * 128))
        floor_keep_n = min(int(full_dim), floor_keep_n)
        mlp_weight_floor += float(floor_keep_n)
        mlp_weight_free += max(0.0, float(full_dim) - float(floor_keep_n))

    floor_keep_total = fixed_keep_other + attn_floor_keep_params + mlp_floor_keep_params
    free_keep_budget_params = max(0.0, target_total_keep - floor_keep_total)
    total_free_params = max(1e-6, attn_free_params + mlp_free_params)
    free_keep_ratio = max(0.0, min(1.0, free_keep_budget_params / total_free_params))

    mlp_free_keep_params = free_keep_budget_params * (mlp_free_params / total_free_params)
    attn_free_keep_params = free_keep_budget_params * (attn_free_params / total_free_params)

    mlp_target_keep_params = mlp_floor_keep_params + mlp_free_keep_params
    attn_target_keep_params = attn_floor_keep_params + attn_free_keep_params

    mlp_target_keep_weight = mlp_weight_floor + free_keep_ratio * mlp_weight_free
    attn_target_keep_weight = attn_weight_floor + free_keep_ratio * attn_weight_free

    if logger is not None:
        logger.log(
            f'[cfsp_flap][joint_budget_calib] target_total_keep={int(target_total_keep)} | fixed_keep_other={int(fixed_keep_other)} | '
            f'attn_floor_keep={int(attn_floor_keep_params)} | mlp_floor_keep={int(mlp_floor_keep_params)} | '
            f'free_keep_budget_params={int(free_keep_budget_params)} | free_keep_ratio={free_keep_ratio:.6f} | '
            f'mlp_target_keep_params={int(mlp_target_keep_params)} | attn_target_keep_params={int(attn_target_keep_params)} | '
            f'mlp_target_keep_weight={mlp_target_keep_weight:.2f} | attn_target_keep_weight={attn_target_keep_weight:.2f} | '
            f'total_floor_weight={(attn_weight_floor + mlp_weight_floor):.2f} | total_free_weight={(attn_weight_free + mlp_weight_free):.2f}'
        )

    return {
        'free_keep_budget_params': free_keep_budget_params,
        'free_keep_ratio': free_keep_ratio,
        'mlp_target_keep_params': mlp_target_keep_params,
        'attn_target_keep_params': attn_target_keep_params,
        'mlp_target_keep_weight': mlp_target_keep_weight,
        'attn_target_keep_weight': attn_target_keep_weight,
        'mlp_floor_keep_params': mlp_floor_keep_params,
        'attn_floor_keep_params': attn_floor_keep_params,
    }



def _prune_mlp_neurons(layer, keep_idx: torch.Tensor):
    cpp_ops = load_cpp_ops(verbose=False)
    keep_idx = keep_idx.to(layer.gate_proj.weight.device)
    pruned_idx = torch.tensor(
        sorted(list(set(range(layer.gate_proj.weight.shape[0])) - set(keep_idx.tolist()))),
        device=keep_idx.device,
        dtype=torch.long,
    )

    gate_full = layer.gate_proj.weight.data
    up_full = layer.up_proj.weight.data
    down_full = layer.down_proj.weight.data

    used_cpp = False
    if cpp_ops is not None:
        try:
            keep_gate, keep_up, keep_down = cpp_ops.prune_mlp_neurons_merge(
                gate_full,
                up_full,
                down_full,
                keep_idx,
                pruned_idx,
            )
            used_cpp = True
        except Exception:
            used_cpp = False

    if not used_cpp:
        keep_gate = gate_full.index_select(0, keep_idx).clone()
        keep_up = up_full.index_select(0, keep_idx).clone()
        keep_down = down_full.index_select(1, keep_idx).clone()

        if pruned_idx.numel() > 0 and keep_idx.numel() > 0:
            pruned_gate = gate_full.index_select(0, pruned_idx)
            pruned_up = up_full.index_select(0, pruned_idx)
            pruned_down = down_full.index_select(1, pruned_idx)

            gate_sim = torch.matmul(
                torch.nn.functional.normalize(pruned_gate.float(), dim=1),
                torch.nn.functional.normalize(keep_gate.float(), dim=1).transpose(0, 1)
            )
            up_sim = torch.matmul(
                torch.nn.functional.normalize(pruned_up.float(), dim=1),
                torch.nn.functional.normalize(keep_up.float(), dim=1).transpose(0, 1)
            )
            sim = 0.55 * gate_sim + 0.45 * up_sim
            merge_topk = min(4, keep_gate.size(0))
            topk_vals, topk_idx = torch.topk(sim, k=merge_topk, dim=1)
            topk_vals = topk_vals.clamp(min=0.0)
            topk_w = topk_vals / topk_vals.sum(dim=1, keepdim=True).clamp_min(1e-6)
            strength = topk_vals.mean(dim=1)

            for src_i in range(pruned_idx.numel()):
                alpha = float(strength[src_i].item())
                blend = min(0.22, 0.06 + 0.18 * alpha)
                for kk in range(merge_topk):
                    dst_i = int(topk_idx[src_i, kk].item())
                    w = float(topk_w[src_i, kk].item())
                    keep_gate[dst_i] += blend * w * pruned_gate[src_i]
                    keep_up[dst_i] += blend * w * pruned_up[src_i]
                    keep_down[:, dst_i] += blend * w * pruned_down[:, src_i]

            gate_scale = (gate_full.abs().mean().clamp_min(1e-6) / keep_gate.abs().mean().clamp_min(1e-6)).clamp(0.90, 1.10)
            up_scale = (up_full.abs().mean().clamp_min(1e-6) / keep_up.abs().mean().clamp_min(1e-6)).clamp(0.90, 1.10)
            down_scale = (down_full.abs().mean().clamp_min(1e-6) / keep_down.abs().mean().clamp_min(1e-6)).clamp(0.90, 1.10)
            keep_gate.mul_(gate_scale)
            keep_up.mul_(up_scale)
            keep_down.mul_(down_scale)

    for proj, keep_weight in [(layer.gate_proj, keep_gate), (layer.up_proj, keep_up)]:
        proj.out_features = keep_idx.numel()
        proj.weight = nn.Parameter(keep_weight)
        if proj.bias is not None:
            proj.bias = nn.Parameter(proj.bias.data.index_select(0, keep_idx))
    layer.down_proj.in_features = keep_idx.numel()
    layer.down_proj.weight = nn.Parameter(keep_down)
    return keep_idx.numel()



def _prune_attention_heads(layer, keep_mask: torch.Tensor):
    cpp_ops = load_cpp_ops(verbose=False)
    head_dim = layer.head_dim
    keep_heads = keep_mask.nonzero(as_tuple=False).view(-1).tolist()

    used_cpp = False
    if cpp_ops is not None:
        try:
            new_q, new_k, new_v, new_o, keep_idx = cpp_ops.prune_attention_heads_weights(
                layer.q_proj.weight.data,
                layer.k_proj.weight.data,
                layer.v_proj.weight.data,
                layer.o_proj.weight.data,
                keep_mask,
                int(head_dim),
            )
            used_cpp = True
        except Exception:
            used_cpp = False

    if not used_cpp:
        keep_idx = []
        for h in keep_heads:
            keep_idx.extend(list(range(h * head_dim, (h + 1) * head_dim)))
        keep_idx = torch.tensor(keep_idx, device=layer.q_proj.weight.device, dtype=torch.long)
        new_q = layer.q_proj.weight.data.index_select(0, keep_idx)
        new_k = layer.k_proj.weight.data.index_select(0, keep_idx)
        new_v = layer.v_proj.weight.data.index_select(0, keep_idx)
        new_o = layer.o_proj.weight.data.index_select(1, keep_idx)

    for proj, new_w in [(layer.q_proj, new_q), (layer.k_proj, new_k), (layer.v_proj, new_v)]:
        proj.out_features = keep_idx.numel()
        proj.weight = nn.Parameter(new_w)
        if proj.bias is not None:
            proj.bias = nn.Parameter(proj.bias.data.index_select(0, keep_idx))

    layer.o_proj.in_features = keep_idx.numel()
    layer.o_proj.weight = nn.Parameter(new_o)
    layer.num_heads = len(keep_heads)
    return len(keep_heads)


def _distill_rerank_candidates(layer, fine_score, keep_n, layer_input_cpu, rerank_ratio=0.50, eval_device='cuda', logger=None, layer_idx=None, rerank_batch_size=16, layer_importance=0.5, grad_signal=None, use_true_local_grad=False, true_grad_topk=128, taylor_signal=None, taylor_weight=0.0, boundary_taylor_primary=False, boundary_taylor_window_ratio=0.35, flap_function_first=False, flap_importance_weight=0.25, coarse_score=None, coarse_overlap_top20=None):
    cpp_ops = load_cpp_ops(verbose=False)
    total = fine_score.numel()
    prune_n = max(0, total - keep_n)
    if prune_n <= 0:
        return torch.topk(fine_score, k=keep_n).indices

    candidate_pool = max(prune_n, int(total * rerank_ratio))
    candidate_pool = min(candidate_pool, total)

    base_order = torch.argsort(fine_score, descending=False)
    boundary_keep = max(256, candidate_pool - prune_n)
    boundary_span = min(candidate_pool, max(1024, min(candidate_pool, prune_n * 2)))
    boundary_start = max(0, candidate_pool - boundary_keep - boundary_span // 2)
    boundary_end = min(candidate_pool, boundary_start + boundary_span)
    boundary_start = max(0, boundary_end - boundary_span)

    always_prune_idx = base_order[:boundary_start]
    rerank_seed_idx = base_order[boundary_start:boundary_end]
    always_keep_idx = base_order[boundary_end:]

    protected_needed = keep_n - always_keep_idx.numel()
    protected_needed = max(0, min(protected_needed, rerank_seed_idx.numel()))

    if eval_device == 'cpu' or not torch.cuda.is_available() or protected_needed == 0:
        keep_idx = torch.cat([always_keep_idx, rerank_seed_idx[prune_n - always_prune_idx.numel():]])
        return keep_idx[:keep_n]

    device = torch.device(eval_device)
    gate_w_full = layer.mlp.gate_proj.weight.detach().to(device=device, dtype=torch.float32)
    up_w_full = layer.mlp.up_proj.weight.detach().to(device=device, dtype=torch.float32)
    down_w_full = layer.mlp.down_proj.weight.detach().to(device=device, dtype=torch.float32)
    layer_input = layer_input_cpu.to(device=device, dtype=torch.float32)
    fine_score_dev = fine_score.to(device=device, dtype=torch.float32)
    coarse_score_dev = coarse_score.to(device=device, dtype=torch.float32) if coarse_score is not None else None
    grad_signal_dev = grad_signal.to(device=device, dtype=torch.float32) if grad_signal is not None else None
    taylor_signal_dev = taylor_signal.to(device=device, dtype=torch.float32) if taylor_signal is not None else None
    act_fn = layer.mlp.act_fn

    with torch.no_grad():
        dense_gate = torch.nn.functional.linear(layer_input, gate_w_full)
        dense_up = torch.nn.functional.linear(layer_input, up_w_full)
        dense_hidden = act_fn(dense_gate) * dense_up
        dense_out = torch.nn.functional.linear(dense_hidden, down_w_full)

    if not torch.isfinite(dense_hidden).all() or not torch.isfinite(dense_out).all():
        if logger is not None:
            logger.log(f'[distill_rerank_v3] layer={layer_idx} | non-finite dense tensors detected | fallback=topk_fine_score')
        return torch.topk(fine_score, k=keep_n).indices

    rerank_t0 = time.time()
    candidate_list = rerank_seed_idx.tolist()
    total_batches = math.ceil(len(candidate_list) / rerank_batch_size) if rerank_batch_size > 0 else 0
    overlap20 = 1.0 if coarse_overlap_top20 is None else float(coarse_overlap_top20)
    adaptive_coarse_w = _coarse_weight_from_overlap(overlap20)
    if logger is not None:
        logger.log(f'[distill_rerank_v3] layer={layer_idx} | overlap20={overlap20:.4f} | coarse_w={adaptive_coarse_w:.4f} | candidate_pool={candidate_pool} | boundary=({boundary_start},{boundary_end}) | rerank_candidates={len(candidate_list)} | always_prune={always_prune_idx.numel()} | always_keep={always_keep_idx.numel()} | protected_needed={protected_needed} | batch_size={rerank_batch_size} | total_batches={total_batches} | compute_dtype=float32')

    flat_hidden = dense_hidden.reshape(-1, total)
    out_dim = down_w_full.shape[0]

    use_cpp_select = (cpp_ops is not None) and not (taylor_signal_dev is not None and float(taylor_weight) > 0.0)
    if use_cpp_select:
        try:
            keep_idx = cpp_ops.rerank_select_keep_idx(
                flat_hidden,
                down_w_full,
                fine_score_dev,
                grad_signal_dev if grad_signal_dev is not None else torch.empty(0, device=device, dtype=torch.float32),
                int(keep_n),
                float(rerank_ratio),
                float(layer_importance),
                int(rerank_batch_size),
            )[0]
            if logger is not None:
                logger.log(f'[distill_rerank_v4] layer={layer_idx} | cpp_select=1 | rerank_candidates={len(candidate_list)} | protected={protected_needed} | final_keep={keep_idx.numel()} | elapsed={time.time()-rerank_t0:.1f}s')
            return keep_idx[:keep_n].to(dtype=torch.long).cpu()
        except Exception as e:
            if logger is not None:
                logger.log(f'[distill_rerank_v4] layer={layer_idx} | cpp_select failed, fallback python | err={e}')
    elif logger is not None and taylor_signal_dev is not None and float(taylor_weight) > 0.0:
        logger.log(f'[distill_rerank_v4] layer={layer_idx} | cpp_select=0 | reason=taylor_python_fusion')

    scores = []
    for batch_start in range(0, len(candidate_list), rerank_batch_size):
        batch = candidate_list[batch_start: batch_start + rerank_batch_size]
        idx_batch = torch.tensor(batch, device=device, dtype=torch.long)

        local_recon = []
        local_tok = []
        for neuron_idx in idx_batch.tolist():
            hidden_j = flat_hidden[:, neuron_idx]
            down_j = down_w_full[:, neuron_idx]
            contrib2d = hidden_j.unsqueeze(1) * down_j.unsqueeze(0)
            contrib = contrib2d.view_as(dense_out)
            sq_err = contrib.pow(2)
            abs_diff = sq_err.mean()
            token_err = sq_err.mean(dim=2)
            topk_n = max(1, min(8, token_err.shape[-1]))
            top_token_diff = torch.topk(token_err, k=topk_n, dim=1).values.mean()
            local_recon.append(abs_diff)
            local_tok.append(top_token_diff)

        recon_err = torch.stack(local_recon)
        tok_err = torch.stack(local_tok)

        if not torch.isfinite(recon_err).all() or not torch.isfinite(tok_err).all():
            if logger is not None:
                logger.log(f'[distill_rerank_v3] layer={layer_idx} | batch has non-finite recon/tok | fallback_local=fine_score | batch_start={batch_start}')
            recon_err = torch.ones_like(recon_err)
            tok_err = torch.ones_like(tok_err)

        recon_err = recon_err / recon_err.mean().clamp_min(1e-6)
        top_token_diff = tok_err / tok_err.mean().clamp_min(1e-6)
        contrib_hidden = torch.stack([flat_hidden[:, neuron_idx] for neuron_idx in idx_batch.tolist()], dim=1)
        contrib_energy = contrib_hidden.pow(2).mean(dim=0).clamp_min(1e-12)
        hidden_drift = torch.sqrt(contrib_energy)
        hidden_drift = hidden_drift / hidden_drift.mean().clamp_min(1e-6)

        local_fine = fine_score_dev.index_select(0, idx_batch)
        local_fine = local_fine / local_fine.mean().clamp_min(1e-6)
        if coarse_score_dev is not None:
            local_coarse = coarse_score_dev.index_select(0, idx_batch)
            local_coarse = local_coarse / local_coarse.mean().clamp_min(1e-6)
        else:
            local_coarse = local_fine
        if grad_signal_dev is not None:
            local_grad = grad_signal_dev.index_select(0, idx_batch)
            local_grad = local_grad / local_grad.mean().clamp_min(1e-6)
        else:
            local_grad = local_fine
        if taylor_signal_dev is not None:
            local_taylor = taylor_signal_dev.index_select(0, idx_batch)
            local_taylor = local_taylor / local_taylor.mean().clamp_min(1e-6)
        else:
            local_taylor = local_grad

        importance = float(layer_importance)
        taylor_mix = max(0.0, float(taylor_weight))
        if flap_function_first:
            functional_loss = (
                (0.46 + 0.08 * importance) * recon_err
                + (0.22 + 0.04 * importance) * top_token_diff
                + 0.14 * hidden_drift
                + adaptive_coarse_w * local_coarse
            )
            importance_bonus = 0.40 * local_taylor + 0.30 * local_grad - 0.10 * local_fine
            fused = functional_loss + max(0.0, float(flap_importance_weight)) * importance_bonus
        else:
            recon_w = 0.34 + 0.06 * importance - 0.05 * taylor_mix
            tok_w = 0.16 + 0.04 * importance - 0.02 * taylor_mix
            coarse_w = adaptive_coarse_w
            grad_w = 0.14 * (1.0 - 0.4 * taylor_mix)
            taylor_w = 0.18 * max(1.0, taylor_mix)
            fine_w = 0.08
            fused = (
                recon_w * recon_err
                + tok_w * top_token_diff
                + coarse_w * local_coarse
                + grad_w * local_grad
                + taylor_w * local_taylor
                - fine_w * local_fine
            )
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1e6, neginf=-1e6)
        if boundary_taylor_primary and taylor_signal_dev is not None and idx_batch.numel() > 1:
            local_window = max(1, int(round(idx_batch.numel() * float(boundary_taylor_window_ratio))))
            if local_window < idx_batch.numel():
                boundary_local = torch.topk(local_fine, k=local_window, largest=False).indices
                fused_boundary = 0.65 * local_taylor.index_select(0, boundary_local) + 0.35 * fused.index_select(0, boundary_local)
                fused.index_copy_(0, boundary_local, fused_boundary)

        scores.extend(list(zip(batch, fused.detach().cpu().tolist())))

    scores.sort(key=lambda x: x[1], reverse=True)
    protected = [idx for idx, _ in scores[:protected_needed]]
    keep_idx = torch.tensor(sorted(list(always_keep_idx.tolist()) + protected), dtype=torch.long)

    if logger is not None:
        logger.log(f'[distill_rerank_v2] layer={layer_idx} | done | rerank_candidates={len(candidate_list)} | protected={len(protected)} | final_keep={keep_idx.numel()} | elapsed={time.time()-rerank_t0:.1f}s | cpp_select=0')

    return keep_idx[:keep_n]


def run_cfsp_ffn_struct_pruner(model, tokenizer, args, logger=None):
    if logger is not None:
        logger.log('prune_probe: entered_SAAP')
    class _CfspFlapLogFilter:
        def __init__(self, base_logger):
            self._base_logger = base_logger

        def log(self, info):
            if self._base_logger is None:
                return
            text = '' if info is None else str(info)
            if '[cfsp_flap]' in text:
                return
            self._base_logger.log(info)

        def __getattr__(self, name):
            return getattr(self._base_logger, name)

    logger = _CfspFlapLogFilter(logger) if logger is not None else None
    t0 = time.time()
    data_t0 = time.time()
    calibration = get_examples(args.dataset, tokenizer, args.num_examples, seq_len=args.calibration_seq_len, logger=logger).to(args.device)

    if logger is not None:
        logger.log('prune_probe: calibration_ready')
        logger.log(f'[cfsp_flap] calibration tensor shape = {tuple(calibration.shape)} | load_elapsed={time.time()-data_t0:.1f}s')

    layer_inputs = {}
    layer_input_chunks = {}
    rerank_layer_inputs = {}
    layer_outputs = {}
    gate_outputs = {}
    attn_inputs = {}
    attn_outputs = {}
    layer_coarse_stats = {}
    gate_output_stats = {}
    attn_input_stats = {}
    attn_output_stats = {}
    mlp_taylor_scores = {}
    attn_taylor_scores = {}
    mlp_input_norm_collectors = {}
    mlp_input_stat_collectors = {}
    attn_o_input_norm_collectors = {}
    attn_o_input_stat_collectors = {}
    hooks = []

    stats_dtype = torch.float16
    cpp_ops = load_cpp_ops(verbose=False)
    use_streaming_stats = cpp_ops is not None
    rerank_max_tokens = int(getattr(args, 'cfsp_rerank_max_tokens', 4096))

    def _accumulate_cpu(cache, layer_idx, tensor):
        tensor = tensor.detach().to(dtype=stats_dtype).cpu()
        if layer_idx not in cache:
            cache[layer_idx] = [tensor]
        else:
            cache[layer_idx].append(tensor)

    def _accumulate_streaming(stats_cache, layer_idx, tensor):
        tensor = tensor.detach().to(dtype=stats_dtype).cpu()
        if cpp_ops is None:
            _accumulate_cpu(stats_cache, layer_idx, tensor)
            return
        if layer_idx not in stats_cache:
            abs_mean, mean, sq_mean, count = cpp_ops.accumulate_feature_stats(tensor)
        else:
            prev = stats_cache[layer_idx]
            abs_mean, mean, sq_mean, count = cpp_ops.combine_feature_stats(
                prev['abs_mean'], prev['mean'], prev['sq_mean'], prev['count'], tensor
            )
        stats_cache[layer_idx] = {
            'abs_mean': abs_mean,
            'mean': mean,
            'sq_mean': sq_mean,
            'count': count,
        }

    def make_block_pre_hook(layer_idx):
        def _hook(module, inputs):
            x = inputs[0]
            rerank_sample = _downsample_layer_inputs_for_rerank(x.detach().to(dtype=stats_dtype).cpu(), max_tokens=rerank_max_tokens)
            _accumulate_cpu(rerank_layer_inputs, layer_idx, rerank_sample)
            if use_streaming_stats:
                layer_input_chunks[layer_idx] = x.detach().to(dtype=stats_dtype).cpu()
            else:
                _accumulate_cpu(layer_inputs, layer_idx, x)
        return _hook

    def make_block_hook(layer_idx):
        def _hook(module, inputs, outputs):
            x = outputs[0] if isinstance(outputs, tuple) else outputs
            if use_streaming_stats:
                x_cpu = x.detach().to(dtype=stats_dtype).cpu()
                in_cpu = layer_input_chunks[layer_idx]
                cur_sum, cur_count = cpp_ops.accumulate_angular_distance_stats(in_cpu, x_cpu, 1e-6)
                if layer_idx not in layer_coarse_stats:
                    layer_coarse_stats[layer_idx] = {'sum': cur_sum, 'count': cur_count}
                else:
                    prev = layer_coarse_stats[layer_idx]
                    total_sum, total_count = cpp_ops.combine_scalar_stats(prev['sum'], prev['count'], cur_sum, cur_count)
                    layer_coarse_stats[layer_idx] = {'sum': total_sum, 'count': total_count}
                del layer_input_chunks[layer_idx]
            else:
                _accumulate_cpu(layer_outputs, layer_idx, x)
        return _hook

    def make_gate_hook(layer_idx):
        def _hook(module, inputs, outputs):
            if use_streaming_stats:
                _accumulate_streaming(gate_output_stats, layer_idx, outputs)
            else:
                _accumulate_cpu(gate_outputs, layer_idx, outputs)
        return _hook

    def make_attn_input_hook(layer_idx):
        def _hook(module, inputs, outputs):
            if use_streaming_stats:
                _accumulate_streaming(attn_input_stats, layer_idx, inputs[0])
            else:
                _accumulate_cpu(attn_inputs, layer_idx, inputs[0])
        return _hook

    def make_attn_output_hook(layer_idx):
        def _hook(module, inputs, outputs):
            x = outputs[0] if isinstance(outputs, tuple) else outputs
            if use_streaming_stats:
                _accumulate_streaming(attn_output_stats, layer_idx, x)
            else:
                _accumulate_cpu(attn_outputs, layer_idx, x)
        return _hook

    def make_mlp_down_input_hook(layer_idx):
        def _hook(module, inputs, outputs):
            x = inputs[0]
            if layer_idx not in mlp_input_norm_collectors:
                mlp_input_norm_collectors[layer_idx] = InputNormCollector(device='cpu', dtype=torch.float32)
                mlp_input_stat_collectors[layer_idx] = InputStatCollector(device='cpu', dtype=torch.float32)
            mlp_input_norm_collectors[layer_idx].add_batch(x)
            mlp_input_stat_collectors[layer_idx].add_batch(x)
        return _hook

    def make_attn_o_input_hook(layer_idx):
        def _hook(module, inputs, outputs):
            x = inputs[0]
            if layer_idx not in attn_o_input_norm_collectors:
                attn_o_input_norm_collectors[layer_idx] = InputNormCollector(device='cpu', dtype=torch.float32)
                attn_o_input_stat_collectors[layer_idx] = InputStatCollector(device='cpu', dtype=torch.float32)
            attn_o_input_norm_collectors[layer_idx].add_batch(x)
            attn_o_input_stat_collectors[layer_idx].add_batch(x)
        return _hook

    use_taylor_rerank = bool(getattr(args, 'cfsp_use_taylor_rerank', False))
    taylor_weight = float(getattr(args, 'cfsp_taylor_rerank_weight', 0.35))
    use_taylor_finescore = bool(getattr(args, 'cfsp_use_taylor_finescore', False))
    taylor_finescore_weight = float(getattr(args, 'cfsp_taylor_finescore_weight', 0.30))
    use_post_taylor_swap = bool(getattr(args, 'cfsp_post_taylor_swap', False))
    post_taylor_swap_topk = int(getattr(args, 'cfsp_post_taylor_swap_topk', 256))
    post_taylor_swap_margin = float(getattr(args, 'cfsp_post_taylor_swap_margin', 1.05))
    post_taylor_mode = str(getattr(args, 'cfsp_post_taylor_mode', 'param_first'))
    use_attn_post_taylor_swap = bool(getattr(args, 'cfsp_attention_post_taylor_swap', False))
    attn_post_taylor_swap_topk = int(getattr(args, 'cfsp_attention_post_taylor_swap_topk', 8))
    attn_post_taylor_swap_margin = float(getattr(args, 'cfsp_attention_post_taylor_swap_margin', 1.03))
    use_boundary_taylor_primary = bool(getattr(args, 'cfsp_boundary_taylor_primary', False))
    boundary_taylor_window_ratio = float(getattr(args, 'cfsp_boundary_taylor_window_ratio', 0.35))
    use_late_layer_parammix = bool(getattr(args, 'cfsp_late_layer_parammix', False))
    late_layer_parammix_start_ratio = float(getattr(args, 'cfsp_late_layer_parammix_start_ratio', 0.67))
    use_flap_function_first = bool(getattr(args, 'cfsp_struct_function_first', False))
    flap_importance_weight = float(getattr(args, 'cfsp_struct_importance_weight', 0.25))
    use_post_reconstruct_ffn = bool(getattr(args, 'cfsp_post_reconstruct_ffn', False))
    post_reconstruct_tokens = int(getattr(args, 'cfsp_post_reconstruct_tokens', 1024))
    post_reconstruct_layers = int(getattr(args, 'cfsp_post_reconstruct_layers', 8))
    log_wanda_flap_scores = bool(getattr(args, 'cfsp_log_wanda_struct_scores', False))
    mlp_coarse_mode = str(getattr(args, 'cfsp_mlp_coarse_mode', 'disabled'))
    attn_score_mode = str(getattr(args, 'cfsp_attn_score_mode', 'disabled'))
    use_mlp_wf_collectors = (mlp_coarse_mode == 'wanda_flap')
    use_attn_wf_collectors = (attn_score_mode == 'act_fluc_wanda')
    for layer_idx, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_pre_hook(make_block_pre_hook(layer_idx)))
        hooks.append(layer.register_forward_hook(make_block_hook(layer_idx)))
        hooks.append(layer.mlp.gate_proj.register_forward_hook(make_gate_hook(layer_idx)))
        if use_mlp_wf_collectors:
            hooks.append(layer.mlp.down_proj.register_forward_hook(make_mlp_down_input_hook(layer_idx)))
        hooks.append(layer.self_attn.q_proj.register_forward_hook(make_attn_input_hook(layer_idx)))
        hooks.append(layer.self_attn.register_forward_hook(make_attn_output_hook(layer_idx)))
        if use_attn_wf_collectors:
            hooks.append(layer.self_attn.o_proj.register_forward_hook(make_attn_o_input_hook(layer_idx)))


    if logger is not None:
        logger.log(f'[cfsp_flap] start calibration forward for coarse/fine statistics | taylor_rerank={int(use_taylor_rerank)} | taylor_weight={taylor_weight:.3f} | taylor_finescore={int(use_taylor_finescore)} | taylor_finescore_weight={taylor_finescore_weight:.3f}')
    fw_t0 = time.time()
    calib_chunk_size = max(1, int(getattr(args, 'cfsp_calib_chunk_size', 16)))
    total_chunks = math.ceil(calibration.shape[0] / calib_chunk_size)
    if use_taylor_rerank:
        model.zero_grad(set_to_none=True)
        for chunk_idx, start in enumerate(range(0, calibration.shape[0], calib_chunk_size), start=1):
            end = min(start + calib_chunk_size, calibration.shape[0])
            calib_chunk = calibration[start:end]
            chunk_t0 = time.time()
            loss = model(calib_chunk, labels=calib_chunk).loss
            loss.backward()
            del loss, calib_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if logger is not None:
                progress = 100.0 * end / max(1, calibration.shape[0])
                logger.log(f'[cfsp_flap] calibration backward chunk {chunk_idx}/{total_chunks} | rows={start}:{end} | progress={progress:.1f}% | chunk_elapsed={time.time()-chunk_t0:.1f}s | total_elapsed={time.time()-fw_t0:.1f}s | cache_cleared=1')
    with torch.no_grad():
        for chunk_idx, start in enumerate(range(0, calibration.shape[0], calib_chunk_size), start=1):
            end = min(start + calib_chunk_size, calibration.shape[0])
            calib_chunk = calibration[start:end]
            chunk_t0 = time.time()
            _ = model(calib_chunk)
            del calib_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if logger is not None:
                progress = 100.0 * end / max(1, calibration.shape[0])
                logger.log(f'[cfsp_flap] calibration forward chunk {chunk_idx}/{total_chunks} | rows={start}:{end} | progress={progress:.1f}% | chunk_elapsed={time.time()-chunk_t0:.1f}s | total_elapsed={time.time()-fw_t0:.1f}s | cache_cleared=1')
    if logger is not None:
        logger.log(f'[cfsp_flap] calibration forward done | elapsed={time.time()-fw_t0:.1f}s | chunks={total_chunks} | chunk_size={calib_chunk_size}')

    for h in hooks:
        h.remove()

    for layer_idx, parts in list(layer_inputs.items()):
        if isinstance(parts, list):
            layer_inputs[layer_idx] = torch.cat(parts, dim=0)

    for layer_idx, parts in list(rerank_layer_inputs.items()):
        if isinstance(parts, list):
            rerank_layer_inputs[layer_idx] = torch.cat(parts, dim=0)
            rerank_layer_inputs[layer_idx] = _downsample_layer_inputs_for_rerank(rerank_layer_inputs[layer_idx], max_tokens=rerank_max_tokens)

    if not use_streaming_stats:
        for cache in [layer_outputs]:
            for layer_idx, parts in list(cache.items()):
                if isinstance(parts, list):
                    cache[layer_idx] = torch.cat(parts, dim=0)

    if not use_streaming_stats:
        for cache in [gate_outputs, attn_inputs, attn_outputs]:
            for layer_idx, parts in list(cache.items()):
                if isinstance(parts, list):
                    cache[layer_idx] = torch.cat(parts, dim=0)

    coarse_scores = []
    kept_neurons = []
    kept_heads = []
    layer_keep_ratios = []
    layer_stats_cpp = {}

    for layer_idx, layer in enumerate(model.model.layers):
        if cpp_ops is not None:
            try:
                layer_attn = layer.self_attn
                if use_streaming_stats:
                    coarse = cpp_ops.finalize_mean_from_stats(
                        layer_coarse_stats[layer_idx]['sum'],
                        layer_coarse_stats[layer_idx]['count'],
                        1e-12,
                    )
                    mlp_stats = cpp_ops.finalize_mlp_act_grad_proxy_from_stats(
                        layer.mlp.gate_proj.weight.detach().float().cpu(),
                        layer.mlp.up_proj.weight.detach().float().cpu(),
                        layer.mlp.down_proj.weight.detach().float().cpu(),
                        gate_output_stats[layer_idx]['abs_mean'],
                        1e-6,
                    )
                    attn_stats = cpp_ops.finalize_attention_head_scores_from_stats(
                        attn_input_stats[layer_idx]['abs_mean'],
                        attn_output_stats[layer_idx]['mean'],
                        attn_output_stats[layer_idx]['sq_mean'],
                        int(layer_attn.head_dim),
                        1e-6,
                    )
                    layer_stats_cpp[layer_idx] = {
                        'coarse': coarse.reshape(()),
                        'act': mlp_stats[0].reshape(-1),
                        'grad_proxy': mlp_stats[1].reshape(-1),
                        'head_act': attn_stats[0].reshape(-1),
                        'head_fluc': attn_stats[1].reshape(-1),
                        'head_score': attn_stats[2].reshape(-1),
                    }
                    coarse_scores.append(coarse)
                    continue
                else:
                    stats = cpp_ops.compute_layer_pruning_stats(
                        layer_inputs[layer_idx],
                        layer_outputs[layer_idx],
                        layer.mlp.gate_proj.weight.detach().float().cpu(),
                        layer.mlp.up_proj.weight.detach().float().cpu(),
                        layer.mlp.down_proj.weight.detach().float().cpu(),
                        gate_outputs[layer_idx],
                        attn_inputs[layer_idx],
                        attn_outputs[layer_idx],
                        int(layer_attn.head_dim),
                    )
                    layer_stats_cpp[layer_idx] = {
                        'coarse': stats[0],
                        'act': stats[1],
                        'grad_proxy': stats[2],
                        'head_act': stats[3],
                        'head_fluc': stats[4],
                        'head_score': stats[5],
                    }
                    coarse_scores.append(stats[0])
                    continue
            except Exception:
                if use_streaming_stats:
                    raise
                pass
        coarse_scores.append(_angular_distance_mean(layer_inputs[layer_idx], layer_outputs[layer_idx], cpp_ops=cpp_ops))
    coarse_scores = torch.stack([x.reshape(()) for x in coarse_scores]).reshape(-1)
    if logger is not None:
        logger.log('prune_probe: coarse_fine_stats_ready')
    keep_ratios = torch.ones_like(coarse_scores)
    mlp_layer_func_scores = torch.zeros_like(coarse_scores)
    prunable_mlp_layers = [
        i for i in range(len(model.model.layers))
        if (i >= args.block_mlp_layer_start and i < args.block_mlp_layer_end)
        and not (i < args.protect_first_n_layers or (args.protect_last_layer and i == len(model.model.layers) - 1))
    ]
    layer_to_keep_ratio = {}
    layer_mlp_prune_ctx = {}
    layer_attn_prune_ctx = {}

    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx < args.block_mlp_layer_start or layer_idx >= args.block_mlp_layer_end:
            continue
        if layer_idx < args.protect_first_n_layers or (args.protect_last_layer and layer_idx == len(model.model.layers) - 1):
            continue

        layer_attn = layer.self_attn
        gate = layer.mlp.gate_proj.weight.detach().float().cpu()
        up = layer.mlp.up_proj.weight.detach().float().cpu()
        down = layer.mlp.down_proj.weight.detach().float().cpu()

        act = None
        grad_proxy = None
        if layer_idx in layer_stats_cpp:
            act = layer_stats_cpp[layer_idx]['act']
            grad_proxy = layer_stats_cpp[layer_idx]['grad_proxy']

        used_cpp_fine = False
        if cpp_ops is not None:
            try:
                if act is None or grad_proxy is None:
                    act, grad_proxy = cpp_ops.compute_mlp_act_grad_proxy(gate, up, down, gate_outputs[layer_idx])
                fine_score_glu, wanda_like, fine_score = cpp_ops.compute_mlp_fine_stats(gate, up, down, act)
                used_cpp_fine = True
            except Exception:
                used_cpp_fine = False

        if not used_cpp_fine:
            if act is None:
                act = gate_outputs[layer_idx].float().abs().mean(dim=(0, 1))
            if grad_proxy is None:
                grad_proxy = (gate.abs().mean(dim=1) * up.abs().mean(dim=1) * down.abs().mean(dim=0) * act).sqrt()

        mlp_input_rms = None
        mlp_input_std = None
        if layer_idx in mlp_input_norm_collectors:
            mlp_input_rms = mlp_input_norm_collectors[layer_idx].rms()
        if layer_idx in mlp_input_stat_collectors:
            mlp_input_std = mlp_input_stat_collectors[layer_idx].std()

        mlp_wf_scores = None
        rerank_coarse_score = None
        coarse_overlap_top20 = None
        if mlp_coarse_mode == 'wanda_flap':
            mlp_wf_scores = _compute_mlp_wanda_flap_scores(
                gate,
                up,
                down,
                input_rms=mlp_input_rms,
                gate_act_mean=act if act is not None else None,
                gate_act_std=mlp_input_std if mlp_input_std is not None else None,
            )
            rerank_coarse_score = mlp_wf_scores["coarse_mix"]
            if logger is not None and log_wanda_flap_scores:
                wanda_mean = mlp_wf_scores["wanda_score"].mean().item()
                wanda_min = mlp_wf_scores["wanda_score"].min().item()
                wanda_max = mlp_wf_scores["wanda_score"].max().item()
                coarse_mean = mlp_wf_scores["coarse_mix"].mean().item()
                coarse_min = mlp_wf_scores["coarse_mix"].min().item()
                coarse_max = mlp_wf_scores["coarse_mix"].max().item()
                logger.log(
                    f'[wf_mlp] layer={layer_idx} | '
                    f'wanda(mean/min/max)=({wanda_mean:.4f}/{wanda_min:.4f}/{wanda_max:.4f}) | '
                    f'coarse_mix(mean/min/max)=({coarse_mean:.4f}/{coarse_min:.4f}/{coarse_max:.4f})'
                )
                if 'fine_score' in locals() and fine_score is not None:
                    overlap_10 = _topk_overlap_ratio(mlp_wf_scores["coarse_mix"], fine_score, ratio=0.10)
                    overlap_20 = _topk_overlap_ratio(mlp_wf_scores["coarse_mix"], fine_score, ratio=0.20)
                    coarse_overlap_top20 = overlap_20
                    logger.log(
                        f'[wf_mlp_overlap] layer={layer_idx} | top10={overlap_10:.4f} | top20={overlap_20:.4f}'
                    )

        if use_taylor_rerank or use_taylor_finescore or use_post_taylor_swap:
            gate_grad = layer.mlp.gate_proj.weight.grad.detach().float().cpu() if layer.mlp.gate_proj.weight.grad is not None else None
            up_grad = layer.mlp.up_proj.weight.grad.detach().float().cpu() if layer.mlp.up_proj.weight.grad is not None else None
            down_grad = layer.mlp.down_proj.weight.grad.detach().float().cpu() if layer.mlp.down_proj.weight.grad is not None else None
            if gate_grad is not None and up_grad is not None and down_grad is not None:
                gate_first = (gate * gate_grad).abs().sum(dim=1)
                up_first = (up * up_grad).abs().sum(dim=1)
                down_first = (down * down_grad).abs().sum(dim=0)
                mlp_taylor_first = (gate_first * up_first).clamp_min(1e-6).sqrt() * (0.7 + 0.3 * (down_first / down_first.mean().clamp_min(1e-6)))

                layer_depth_ratio = float(layer_idx) / max(1, len(model.model.layers) - 1)
                layer_post_mode = post_taylor_mode
                if use_late_layer_parammix and layer_depth_ratio >= float(late_layer_parammix_start_ratio):
                    layer_post_mode = 'param_mix'

                if layer_post_mode == 'param_mix':
                    gate_mix = gate_first + 0.5 * (gate_grad.pow(2)).sum(dim=1) * gate.abs().mean(dim=1)
                    up_mix = up_first + 0.5 * (up_grad.pow(2)).sum(dim=1) * up.abs().mean(dim=1)
                    down_mix = down_first + 0.5 * (down_grad.pow(2)).sum(dim=0) * down.abs().mean(dim=0)
                    mlp_taylor_post = (gate_mix * up_mix).clamp_min(1e-6).sqrt() * (0.7 + 0.3 * (down_mix / down_mix.mean().clamp_min(1e-6)))
                else:
                    mlp_taylor_post = mlp_taylor_first

                mlp_taylor_first = mlp_taylor_first / mlp_taylor_first.mean().clamp_min(1e-6)
                mlp_taylor_post = mlp_taylor_post / mlp_taylor_post.mean().clamp_min(1e-6)
                mlp_taylor_scores[layer_idx] = mlp_taylor_first
                if use_post_taylor_swap:
                    mlp_taylor_scores[(layer_idx, 'postswap')] = mlp_taylor_post

        if not used_cpp_fine:
            gate_score = gate.abs().sum(dim=1)
            up_score = up.abs().sum(dim=1)
            down_score = down.abs().sum(dim=0)
            rel_act = act / act.mean().clamp_min(1e-6)
            down_support = down_score / down_score.mean().clamp_min(1e-6)

            wanda_base = (gate.abs().mean(dim=1) + up.abs().mean(dim=1)).clamp_min(1e-6) * rel_act
            wanda_base = wanda_base * (0.88 + 0.12 * torch.sqrt(down_support.clamp_min(1e-6)))

            glu_core = torch.sqrt((gate_score * up_score).clamp_min(1e-6))
            gate_up_balance = torch.minimum(gate_score, up_score)
            fine_score_glu = glu_core * rel_act * (0.82 + 0.18 * down_support) + 0.20 * gate_up_balance

            wanda_like = torch.sqrt((gate_score + up_score).clamp_min(1e-6)) * rel_act * torch.sqrt(down_support.clamp_min(1e-6))

            wanda_base = wanda_base / wanda_base.mean().clamp_min(1e-6)
            wanda_like = wanda_like / wanda_like.mean().clamp_min(1e-6)
            fine_score_glu = fine_score_glu / fine_score_glu.mean().clamp_min(1e-6)
            fine_score = 0.45 * wanda_base + 0.30 * fine_score_glu + 0.25 * wanda_like

        if use_taylor_finescore and layer_idx in mlp_taylor_scores:
            mlp_taylor = mlp_taylor_scores[layer_idx]
            taylor_mix = max(0.0, min(0.9, float(taylor_finescore_weight)))
            base_mix = max(1e-6, 1.0 - taylor_mix)
            fine_score = base_mix * fine_score + taylor_mix * mlp_taylor
            fine_score = fine_score / fine_score.mean().clamp_min(1e-6)
            if logger is not None and (layer_idx % 4 == 0 or layer_idx == len(model.model.layers) - 1):
                logger.log(f'[cfsp_flap] layer={layer_idx} | applied taylor_finescore mix | weight={taylor_mix:.3f} | taylor(mean/min/max)=({mlp_taylor.mean().item():.6f}/{mlp_taylor.min().item():.6f}/{mlp_taylor.max().item():.6f})')

        rel_act_for_func = act / act.mean().clamp_min(1e-6)
        act_mean_for_func = rel_act_for_func.mean().reshape(())
        head_fluc_mean = torch.ones((), dtype=torch.float32)
        if layer_idx in layer_stats_cpp:
            head_fluc_mean = layer_stats_cpp[layer_idx]['head_fluc'].float().mean().reshape(())
        elif layer_idx in attn_outputs and layer_idx in attn_inputs:
            try:
                _, head_fluc_tmp, _ = _compute_attention_head_scores(attn_inputs[layer_idx], attn_outputs[layer_idx], int(layer_attn.head_dim), cpp_ops=cpp_ops)
                head_fluc_mean = head_fluc_tmp.float().mean().reshape(())
            except Exception:
                head_fluc_mean = torch.ones((), dtype=torch.float32)
        mlp_layer_func_scores[layer_idx] = (
            0.45 * coarse_scores[layer_idx].reshape(())
            + 0.30 * act_mean_for_func
            + 0.25 * head_fluc_mean
        )

        layer_mlp_prune_ctx[layer_idx] = {
            'layer': layer,
            'layer_attn': layer_attn,
            'full_dim': layer.mlp.gate_proj.weight.shape[0],
            'total_heads': layer_attn.q_proj.weight.shape[0] // layer_attn.head_dim,
            'gate': gate,
            'up': up,
            'down': down,
            'act': act,
            'grad_proxy': grad_proxy,
            'fine_score': fine_score,
            'rerank_coarse_score': rerank_coarse_score,
            'coarse_overlap_top20': coarse_overlap_top20,
        }

        fine_score_std = (fine_score - fine_score.mean()) / fine_score.std().clamp_min(1e-6)
        coarse_global = fine_score_std
        if rerank_coarse_score is not None:
            coarse_global = (rerank_coarse_score.reshape(-1) - rerank_coarse_score.mean()) / rerank_coarse_score.std().clamp_min(1e-6)
        layer_func_scalar = mlp_layer_func_scores[layer_idx].reshape(()).float()
        global_score = 0.70 * coarse_global + 0.30 * fine_score_std
        global_score = global_score * layer_func_scalar.clamp_min(1e-6)
        global_score = (global_score - global_score.mean()) / global_score.std().clamp_min(1e-6)
        layer_mlp_prune_ctx[layer_idx]['fine_score_std'] = fine_score_std
        layer_mlp_prune_ctx[layer_idx]['fine_score_flap_global'] = global_score

        head_dim = layer_attn.head_dim
        total_heads = layer_attn.q_proj.weight.shape[0] // head_dim
        if layer_idx in layer_stats_cpp:
            head_act = layer_stats_cpp[layer_idx]['head_act'].float().cpu()
            head_fluc = layer_stats_cpp[layer_idx]['head_fluc'].float().cpu()
            head_score = layer_stats_cpp[layer_idx]['head_score'].float().cpu()
        else:
            q_in = attn_inputs[layer_idx].float()
            attn_out = attn_outputs[layer_idx].float()
            head_act, head_fluc, head_score = _compute_attention_head_scores(
                q_in,
                attn_out,
                head_dim=head_dim,
                cpp_ops=cpp_ops,
            )
            head_act = head_act.float().cpu()
            head_fluc = head_fluc.float().cpu()
            head_score = head_score.float().cpu()

        attn_global = head_score
        if attn_score_mode == 'act_fluc_wanda':
            attn_wf_scores = _compute_attention_head_wanda_scores(
                layer.self_attn.q_proj.weight.detach().float().cpu(),
                layer.self_attn.k_proj.weight.detach().float().cpu(),
                layer.self_attn.v_proj.weight.detach().float().cpu(),
                layer.self_attn.o_proj.weight.detach().float().cpu(),
                head_dim,
                head_act,
                head_fluc,
            )
            attn_global = 0.70 * attn_wf_scores['head_mix'].float().cpu() + 0.30 * head_score

        attn_global_std = (attn_global - attn_global.mean()) / attn_global.std().clamp_min(1e-6)
        layer_attn_prune_ctx[layer_idx] = {
            'head_dim': head_dim,
            'total_heads': total_heads,
            'head_act': head_act,
            'head_fluc': head_fluc,
            'head_score': head_score,
            'attn_global_std': attn_global_std,
        }


    if len(prunable_mlp_layers) > 0:
        prunable_idx = torch.tensor(prunable_mlp_layers, dtype=torch.long)
        prunable_scores = coarse_scores[prunable_idx]
        prunable_func_scores = mlp_layer_func_scores[prunable_idx]
        effective_mlp_pruning_ratio = _estimate_effective_mlp_pruning_ratio(model, args, logger=logger)
        prunable_keep = _allocate_prunable_keep_ratios_continuous(
            prunable_scores,
            effective_mlp_pruning_ratio,
            min_keep_ratio=args.cfsp_min_keep_ratio,
            layer_sensitivity=prunable_func_scores,
        ).reshape(-1)

        prunable_attn_layers = [
            i for i in range(len(model.model.layers))
            if (i >= args.block_attention_layer_start and i < args.block_attention_layer_end)
            and not (i < args.protect_first_n_layers or (args.protect_last_layer and i == len(model.model.layers) - 1))
        ]

        budget_info = _estimate_joint_attn_mlp_keep_budget(
            model,
            args,
            prunable_mlp_layers,
            prunable_attn_layers,
            layer_attn_prune_ctx,
            logger=logger,
        )

        mlp_metric = []
        for layer_idx in prunable_mlp_layers:
            mlp_metric.append(layer_mlp_prune_ctx[layer_idx]['fine_score_flap_global'].reshape(-1))
        mlp_metric = torch.cat(mlp_metric, dim=0) if len(mlp_metric) > 0 else torch.empty(0)
        mlp_weight = torch.ones(mlp_metric.numel(), dtype=torch.float32)
        mlp_sorted_metric, mlp_sorted_idx = torch.sort(mlp_metric, descending=True)
        mlp_sorted_weight = mlp_weight[mlp_sorted_idx]
        mlp_keep_ptr = int(torch.argmin((torch.cumsum(mlp_sorted_weight, dim=0) - budget_info['mlp_target_keep_weight']).abs()).item()) if mlp_metric.numel() > 0 else 0
        mlp_global_threshold = float(mlp_sorted_metric[mlp_keep_ptr].item()) if mlp_metric.numel() > 0 else float('-inf')

        attn_metric = []
        for layer_idx in prunable_attn_layers:
            attn_metric.append(layer_attn_prune_ctx[layer_idx]['attn_global_std'].reshape(-1))
        attn_metric = torch.cat(attn_metric, dim=0) if len(attn_metric) > 0 else torch.empty(0)
        attn_weight = torch.full((attn_metric.numel(),), 512.0 / 3.0, dtype=torch.float32)
        attn_sorted_metric, attn_sorted_idx = torch.sort(attn_metric, descending=True)
        attn_sorted_weight = attn_weight[attn_sorted_idx]
        attn_keep_ptr = int(torch.argmin((torch.cumsum(attn_sorted_weight, dim=0) - budget_info['attn_target_keep_weight']).abs()).item()) if attn_metric.numel() > 0 else 0
        attn_global_threshold = float(attn_sorted_metric[attn_keep_ptr].item()) if attn_metric.numel() > 0 else float('-inf')

        prunable_keep_debug = []
        attn_keep_debug = []
        for idx, layer_idx in enumerate(prunable_mlp_layers):
            std_score = layer_mlp_prune_ctx[layer_idx]['fine_score_flap_global'].reshape(-1)
            global_keep_mask = std_score >= mlp_global_threshold
            keep_from_global = int(global_keep_mask.sum().item())
            min_keep_n = max(128, int(round(std_score.numel() * float(args.cfsp_min_keep_ratio) / 128) * 128))
            keep_n_global = max(min_keep_n, keep_from_global)
            keep_n_global = min(std_score.numel(), max(128, int(round(keep_n_global / 128) * 128)))
            keep_value = keep_n_global / max(1, std_score.numel())
            keep_ratios[layer_idx] = keep_value
            layer_to_keep_ratio[layer_idx] = keep_value
            layer_mlp_prune_ctx[layer_idx]['global_keep_n'] = keep_n_global
            layer_mlp_prune_ctx[layer_idx]['global_threshold'] = mlp_global_threshold
            prunable_keep_debug.append(float(keep_value))

        for layer_idx in prunable_attn_layers:
            std_score = layer_attn_prune_ctx[layer_idx]['attn_global_std'].reshape(-1)
            global_keep_mask = std_score >= attn_global_threshold
            keep_from_global = int(global_keep_mask.sum().item())
            total_heads = int(layer_attn_prune_ctx[layer_idx]['total_heads'])
            min_keep_heads = max(int(args.cfsp_attn_min_heads), 1)
            keep_heads_n = max(min_keep_heads, keep_from_global)
            keep_heads_n = min(total_heads, keep_heads_n)
            layer_attn_prune_ctx[layer_idx]['global_keep_heads_n'] = keep_heads_n
            layer_attn_prune_ctx[layer_idx]['global_threshold'] = attn_global_threshold
            attn_keep_debug.append(float(keep_heads_n / max(1, total_heads)))

        if logger is not None:
            logger.log(
                f'[cfsp_flap][continuous_keep_debug] prunable_layers={prunable_mlp_layers} | '
                f'prunable_func_scores={[float(x) for x in prunable_func_scores.tolist()]} | '
                f'prunable_keep={prunable_keep_debug} | attn_keep={attn_keep_debug} | '
                f'mlp_global_threshold={mlp_global_threshold:.6f} | attn_global_threshold={attn_global_threshold:.6f} | '
                f'mlp_target_keep_weight={budget_info["mlp_target_keep_weight"]:.2f} | attn_target_keep_weight={budget_info["attn_target_keep_weight"]:.2f}'
            )

    if logger is not None:
        logger.log(f'[cfsp_flap] coarse score stats | mean={coarse_scores.mean().item():.6f} | min={coarse_scores.min().item():.6f} | max={coarse_scores.max().item():.6f}')
        logger.log(f'[cfsp_flap] keep ratio stats | mean={keep_ratios.mean().item():.6f} | min={keep_ratios.min().item():.6f} | max={keep_ratios.max().item():.6f}')

    if logger is not None:
        logger.log('prune_probe: layer_prune_loop_start')

    for layer_idx, layer in enumerate(model.model.layers):
        full_dim = layer.mlp.gate_proj.weight.shape[0]
        layer_attn = layer.self_attn
        total_heads = layer_attn.q_proj.weight.shape[0] // layer_attn.head_dim
        attn_keep = total_heads
        if layer_idx < args.block_mlp_layer_start or layer_idx >= args.block_mlp_layer_end:
            kept_neurons.append(full_dim)
            kept_heads.append(attn_keep)
            layer_keep_ratios.append(1.0)
            continue
        if layer_idx < args.protect_first_n_layers or (args.protect_last_layer and layer_idx == len(model.model.layers) - 1):
            kept_neurons.append(full_dim)
            kept_heads.append(attn_keep)
            layer_keep_ratios.append(1.0)
            continue

        if logger is not None:
            logger.log(f'prune_layer: mlp layer={layer_idx}')

        ctx = layer_mlp_prune_ctx[layer_idx]
        gate = ctx['gate']
        up = ctx['up']
        down = ctx['down']
        act = ctx['act']
        grad_proxy = ctx['grad_proxy']
        fine_score = ctx['fine_score']
        rerank_coarse_score = ctx['rerank_coarse_score']
        coarse_overlap_top20 = ctx['coarse_overlap_top20']

        late_layer_rescue_start = max(args.block_mlp_layer_start, len(model.model.layers) - 8)
        late_layer_more_keep_start = max(args.block_mlp_layer_start, len(model.model.layers) - 4)
        if layer_idx >= late_layer_rescue_start:
            rel_act = act / act.mean().clamp_min(1e-6)
            down_support = down.abs().sum(dim=0)
            down_support = down_support / down_support.mean().clamp_min(1e-6)
            late_rescue = torch.sqrt((rel_act * down_support).clamp_min(1e-6))
            if layer_idx in mlp_taylor_scores:
                late_rescue = 0.80 * late_rescue + 0.20 * mlp_taylor_scores[layer_idx]
            late_rescue = late_rescue / late_rescue.mean().clamp_min(1e-6)
            fine_score = 0.88 * fine_score + 0.12 * late_rescue
            fine_score = fine_score / fine_score.mean().clamp_min(1e-6)
            if logger is not None:
                logger.log(f'[cfsp_flap][late_rescue] layer={layer_idx} | enabled=1 | rescue_start={late_layer_rescue_start} | more_keep_start={late_layer_more_keep_start} | late_rescue(mean/min/max)=({late_rescue.mean().item():.6f}/{late_rescue.min().item():.6f}/{late_rescue.max().item():.6f})')

        if logger is not None and (layer_idx % 4 == 0 or layer_idx == len(model.model.layers) - 1):
            logger.log(f'[cfsp_flap] scoring layer={layer_idx} | coarse={coarse_scores[layer_idx].item():.6f} | fine_score(mean/min/max)=({fine_score.mean().item():.6f}/{fine_score.min().item():.6f}/{fine_score.max().item():.6f})')

        target_keep_ratio = layer_to_keep_ratio.get(layer_idx, float(keep_ratios[layer_idx].item()))
        keep_n = int(layer_mlp_prune_ctx[layer_idx].get('global_keep_n', max(128, int(round(fine_score.numel() * target_keep_ratio / 128) * 128))))
        keep_n = min(keep_n, fine_score.numel())
        target_keep_ratio = keep_n / max(1, fine_score.numel())
        raw_keep = fine_score.numel() * target_keep_ratio
        actual_keep_ratio_pre = keep_n / max(1, fine_score.numel())
        if logger is not None:
            logger.log(
                f'[cfsp_flap][keep_debug] layer={layer_idx} | target_keep={target_keep_ratio:.6f} | total={fine_score.numel()} '
                f'| raw_keep={raw_keep:.2f} | rounded_keep={keep_n} | actual_keep_pre={actual_keep_ratio_pre:.6f} '
                f'| rounding_delta={keep_n - raw_keep:.2f}'
            )

        if getattr(args, 'cfsp_distill_rerank', False):
            if logger is not None:
                logger.log(f'[cfsp_flap] layer={layer_idx} | enter distill rerank | keep_n={keep_n} | total={fine_score.numel()}')
            keep_idx = _distill_rerank_candidates(
                layer=layer,
                fine_score=fine_score,
                keep_n=keep_n,
                layer_input_cpu=rerank_layer_inputs[layer_idx],
                rerank_ratio=args.cfsp_candidate_pool_ratio,
                eval_device=args.eval_device,
                logger=logger,
                layer_idx=layer_idx,
                rerank_batch_size=args.cfsp_rerank_batch_size,
                layer_importance=(coarse_scores[layer_idx] / coarse_scores.max().clamp_min(1e-6)).item(),
                grad_signal=grad_proxy,
                taylor_signal=mlp_taylor_scores.get(layer_idx),
                taylor_weight=taylor_weight if use_taylor_rerank else 0.0,
                boundary_taylor_primary=use_boundary_taylor_primary,
                boundary_taylor_window_ratio=boundary_taylor_window_ratio,
                flap_function_first=use_flap_function_first,
                flap_importance_weight=flap_importance_weight,
                coarse_score=rerank_coarse_score,
                coarse_overlap_top20=coarse_overlap_top20,
            )
        else:
            keep_idx = torch.topk(fine_score, k=keep_n).indices

        if use_post_taylor_swap and ((layer_idx, 'postswap') in mlp_taylor_scores or layer_idx in mlp_taylor_scores):
            late_layer_rescue_start = max(args.block_mlp_layer_start, len(model.model.layers) - 8)
            late_layer_more_keep_start = max(args.block_mlp_layer_start, len(model.model.layers) - 4)
            layer_swap_topk = int(post_taylor_swap_topk)
            if layer_idx >= late_layer_more_keep_start:
                layer_swap_topk = max(layer_swap_topk, int(post_taylor_swap_topk * 2))
            elif layer_idx >= late_layer_rescue_start:
                layer_swap_topk = max(layer_swap_topk, int(post_taylor_swap_topk * 3 / 2))
            keep_idx = _post_taylor_swap_keep_idx(
                fine_score=fine_score,
                keep_idx=keep_idx,
                taylor_score=mlp_taylor_scores.get((layer_idx, 'postswap'), mlp_taylor_scores.get(layer_idx)),
                swap_topk=layer_swap_topk,
                swap_margin=post_taylor_swap_margin,
                logger=logger,
                layer_idx=layer_idx,
            )

        kept = _prune_mlp_neurons(layer.mlp, keep_idx)
        if use_post_reconstruct_ffn and layer_idx >= max(args.block_mlp_layer_start, len(model.model.layers) - post_reconstruct_layers):
            _post_reconstruct_mlp_downproj(
                layer.mlp,
                rerank_layer_inputs.get(layer_idx),
                max_tokens=post_reconstruct_tokens,
                logger=logger,
                layer_idx=layer_idx,
            )

        attn_keep_ratio = 1.0
        if layer_idx >= args.block_attention_layer_start and layer_idx < args.block_attention_layer_end:
            if logger is not None:
                logger.log(f'prune_layer: attn layer={layer_idx}')
            layer_importance = (coarse_scores[layer_idx] / coarse_scores.max().clamp_min(1e-6)).item()
            low_saliency = coarse_scores[layer_idx] < coarse_scores.mean() * (1.0 + 0.10 * (layer_importance - 0.5))

            head_dim = layer_attn.head_dim
            total_heads = layer_attn.q_proj.weight.shape[0] // head_dim
            if layer_idx in layer_attn_prune_ctx:
                head_act = layer_attn_prune_ctx[layer_idx]['head_act']
                head_fluc = layer_attn_prune_ctx[layer_idx]['head_fluc']
                head_score = layer_attn_prune_ctx[layer_idx]['head_score']
            elif layer_idx in layer_stats_cpp:
                head_act = layer_stats_cpp[layer_idx]['head_act']
                head_fluc = layer_stats_cpp[layer_idx]['head_fluc']
                head_score = layer_stats_cpp[layer_idx]['head_score']
            else:
                q_in = attn_inputs[layer_idx].float()
                attn_out = attn_outputs[layer_idx].float()
                head_act, head_fluc, head_score = _compute_attention_head_scores(
                    q_in,
                    attn_out,
                    head_dim=head_dim,
                    cpp_ops=cpp_ops,
                )
            attn_wf_scores = None
            if attn_score_mode == 'act_fluc_wanda':
                attn_wf_scores = _compute_attention_head_wanda_scores(
                    layer.self_attn.q_proj.weight.detach().float().cpu(),
                    layer.self_attn.k_proj.weight.detach().float().cpu(),
                    layer.self_attn.v_proj.weight.detach().float().cpu(),
                    layer.self_attn.o_proj.weight.detach().float().cpu(),
                    head_dim,
                    head_act.float().cpu(),
                    head_fluc.float().cpu(),
                )
                if logger is not None and log_wanda_flap_scores:
                    wanda_mean = attn_wf_scores["head_wanda"].mean().item()
                    wanda_min = attn_wf_scores["head_wanda"].min().item()
                    wanda_max = attn_wf_scores["head_wanda"].max().item()
                    mix_mean = attn_wf_scores["head_mix"].mean().item()
                    mix_min = attn_wf_scores["head_mix"].min().item()
                    mix_max = attn_wf_scores["head_mix"].max().item()
                    logger.log(
                        f'[wf_attn] layer={layer_idx} | '
                        f'wanda(mean/min/max)=({wanda_mean:.4f}/{wanda_min:.4f}/{wanda_max:.4f}) | '
                        f'head_mix(mean/min/max)=({mix_mean:.4f}/{mix_min:.4f}/{mix_max:.4f})'
                    )

            if use_taylor_rerank or use_attn_post_taylor_swap:
                qg = layer_attn.q_proj.weight.grad.detach().float().cpu() if layer_attn.q_proj.weight.grad is not None else None
                kg = layer_attn.k_proj.weight.grad.detach().float().cpu() if layer_attn.k_proj.weight.grad is not None else None
                vg = layer_attn.v_proj.weight.grad.detach().float().cpu() if layer_attn.v_proj.weight.grad is not None else None
                og = layer_attn.o_proj.weight.grad.detach().float().cpu() if layer_attn.o_proj.weight.grad is not None else None
                if qg is not None and kg is not None and vg is not None and og is not None:
                    q_t = (layer_attn.q_proj.weight.detach().float().cpu() * qg).abs().sum(dim=1).view(total_heads, head_dim).sum(dim=1)
                    k_t = (layer_attn.k_proj.weight.detach().float().cpu() * kg).abs().sum(dim=1).view(total_heads, head_dim).sum(dim=1)
                    v_t = (layer_attn.v_proj.weight.detach().float().cpu() * vg).abs().sum(dim=1).view(total_heads, head_dim).sum(dim=1)
                    o_t = (layer_attn.o_proj.weight.detach().float().cpu() * og).abs().sum(dim=0).view(total_heads, head_dim).sum(dim=1)
                    attn_taylor = (q_t + k_t + v_t + o_t) / 4.0
                    attn_taylor = attn_taylor / attn_taylor.mean().clamp_min(1e-6)
                    attn_taylor_scores[layer_idx] = attn_taylor
                    if use_taylor_rerank:
                        head_score = (1.0 - taylor_weight) * head_score + taylor_weight * attn_taylor
            keep_heads_n = int(layer_attn_prune_ctx.get(layer_idx, {}).get('global_keep_heads_n', max(max(args.cfsp_attn_min_heads, 20), int(round(total_heads * args.cfsp_attn_keep_ratio)))))
            keep_heads_n = min(keep_heads_n, total_heads)
            attn_keep_ratio_layer = keep_heads_n / max(1, total_heads)
            top_heads = torch.topk(head_score, k=keep_heads_n).indices
            keep_mask = torch.zeros(total_heads, dtype=torch.bool)
            keep_mask[top_heads] = True
            if use_attn_post_taylor_swap and layer_idx in attn_taylor_scores:
                keep_mask = _post_taylor_swap_keep_mask(
                    head_score=head_score,
                    keep_mask=keep_mask,
                    taylor_score=attn_taylor_scores[layer_idx],
                    swap_topk=attn_post_taylor_swap_topk,
                    swap_margin=attn_post_taylor_swap_margin,
                    logger=logger,
                    layer_idx=layer_idx,
                )
            attn_keep = _prune_attention_heads(layer_attn, keep_mask)
            attn_keep_ratio = attn_keep / total_heads
            if logger is not None:
                logger.log(
                    f'[cfsp_flap] layer={layer_idx} | low_saliency={low_saliency} | layer_imp={layer_importance:.4f} | attn_keep_ratio_layer={attn_keep_ratio_layer:.4f} '
                    f'| attn_score(mean/min/max/std)=({head_score.mean().item():.6f}/{head_score.min().item():.6f}/{head_score.max().item():.6f}/{head_score.std().item():.6f}) '
                    f'| attn_act(mean/min/max/std)=({head_act.mean().item():.6f}/{head_act.min().item():.6f}/{head_act.max().item():.6f}/{head_act.std().item():.6f}) '
                    f'| attn_fluc(mean/min/max/std)=({head_fluc.mean().item():.6f}/{head_fluc.min().item():.6f}/{head_fluc.max().item():.6f}/{head_fluc.std().item():.6f})'
                )

        kept_neurons.append(kept)
        kept_heads.append(attn_keep)
        layer_keep_ratios.append(keep_n / fine_score.numel())

        if logger is not None:
            logger.log(f'[cfsp_flap] layer={layer_idx} | coarse={coarse_scores[layer_idx].item():.6f} | target_keep={keep_ratios[layer_idx].item():.4f} | actual_keep={layer_keep_ratios[-1]:.4f} | kept_neurons={kept} | kept_heads={attn_keep} | attn_keep_ratio={attn_keep_ratio:.4f}')

    if bool(getattr(args, 'export_calibration_bundle', True)) and logger is not None:
        try:
            bundle_dir = os.path.join(logger.sub_dir, 'calibration_bundle')
            os.makedirs(bundle_dir, exist_ok=True)
            calib_cpu = calibration.detach().cpu()
            torch.save(calib_cpu, os.path.join(bundle_dir, 'calibration_tokens.pt'))
            manifest = {
                'dataset': getattr(args, 'dataset', ''),
                'num_examples': int(getattr(args, 'num_examples', calib_cpu.shape[0] if calib_cpu.ndim > 0 else 0)),
                'calibration_seq_len': int(getattr(args, 'calibration_seq_len', calib_cpu.shape[-1] if calib_cpu.ndim > 1 else 0)),
                'saved_files': {
                    'calibration_tokens': 'calibration_tokens.pt',
                },
            }
            with open(os.path.join(bundle_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            logger.log(f'[cfsp_flap] calibration bundle exported | dir={bundle_dir} | tokens_shape={tuple(calib_cpu.shape)}')
        except Exception as e:
            logger.log(f'[cfsp_flap] calibration bundle export failed | err={e}')

    del layer_stats_cpp
    del layer_inputs
    del rerank_layer_inputs
    del layer_outputs
    del gate_outputs
    del attn_inputs
    del attn_outputs
    del layer_coarse_stats
    del gate_output_stats
    del attn_input_stats
    del attn_output_stats
    model.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if logger is not None:
        logger.log(f'[cfsp_flap] done in {time.time() - t0:.1f}s')

    return {
        'coarse_scores': [float(x) for x in coarse_scores.tolist()],
        'keep_ratios': [float(x) for x in layer_keep_ratios],
        'kept_neurons': kept_neurons,
        'kept_heads': kept_heads,
    }
