import math
import torch
import core.torch_pruning as tp
from core.pruner import hf_llama_pruner as llama_pruner


def _safe_zscore(x: torch.Tensor, eps: float = 1e-6):
    if x.numel() <= 1:
        return torch.zeros_like(x)
    mean = x.mean()
    std = x.std(unbiased=False)
    if torch.isnan(std) or std < eps:
        return torch.zeros_like(x)
    return (x - mean) / (std + eps)


def _normal_cdf(x: torch.Tensor):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _normal_icdf(p: torch.Tensor):
    eps = torch.finfo(p.dtype).eps if p.is_floating_point() else 1e-6
    p = p.clamp(min=eps, max=1.0 - eps)
    return math.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)


def _quantile_map_to_normal(x: torch.Tensor):
    if x.numel() <= 1:
        return torch.zeros_like(x)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    ranks = torch.argsort(torch.argsort(x))
    probs = (ranks.to(x.dtype) + 0.5) / x.numel()
    mapped = _normal_icdf(probs)
    return torch.nan_to_num(mapped, nan=0.0, posinf=0.0, neginf=0.0)


class GradientImportance(tp.importance.Importance):
    def __init__(self, group_reduction="mean", normalizer=None, use_abs_grad=True):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.use_abs_grad = use_abs_grad

    def _reduce(self, group_imp):
        if self.group_reduction == "sum":
            group_imp = group_imp.sum(dim=0)
        elif self.group_reduction == "mean":
            group_imp = group_imp.mean(dim=0)
        elif self.group_reduction == "max":
            group_imp = group_imp.max(dim=0)[0]
        elif self.group_reduction == "prod":
            group_imp = torch.prod(group_imp, dim=0)
        elif self.group_reduction == 'first':
            group_imp = group_imp[0]
        elif self.group_reduction is None:
            group_imp = group_imp
        else:
            raise NotImplementedError
        return group_imp

    @torch.no_grad()
    def __call__(self, group, ch_groups=1, consecutive_groups=1):
        group_imp = []
        for dep, idxs in group:
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler

            def grad_score(tensor):
                grad = getattr(tensor, 'grad', None)
                if grad is None:
                    return None
                score = grad.abs() if self.use_abs_grad else grad
                return score

            if prune_fn in [tp.prune_linear_out_channels, llama_pruner.hf_linear_pruner.prune_out_channels]:
                score = grad_score(layer.weight)
                if score is None:
                    continue
                local = score[idxs].flatten(1).sum(1)
                group_imp.append(local)
            elif prune_fn in [tp.prune_linear_in_channels, llama_pruner.hf_linear_pruner.prune_in_channels]:
                score = grad_score(layer.weight)
                if score is None:
                    continue
                local = score.sum(0)[idxs]
                group_imp.append(local)
            elif prune_fn == llama_pruner.hf_rmsnorm_pruner.prune_out_channels:
                score = grad_score(layer.weight)
                if score is None:
                    continue
                group_imp.append(score[idxs])
            elif prune_fn == tp.prune_embedding_out_channels:
                score = grad_score(layer.weight)
                if score is None:
                    continue
                group_imp.append(score[:, idxs].sum(0))
            elif prune_fn == llama_pruner.hf_attention_pruner.prune_out_channels:
                local = 0
                valid = False
                for sub_layer in [layer.o_proj]:
                    score = grad_score(sub_layer.weight)
                    if score is not None:
                        local = local + score[idxs].flatten(1).sum(1)
                        valid = True
                for sub_layer in [layer.q_proj, layer.k_proj, layer.v_proj]:
                    score = grad_score(sub_layer.weight)
                    if score is not None:
                        local = local + score.sum(0)[idxs]
                        valid = True
                if valid:
                    group_imp.append(local)

        if len(group_imp) == 0:
            return None
        min_imp_size = min([len(imp) for imp in group_imp])
        aligned_group_imp = []
        for imp in group_imp:
            if len(imp) > min_imp_size and len(imp) % min_imp_size == 0:
                imp = imp.view(len(imp) // min_imp_size, min_imp_size).sum(0)
                aligned_group_imp.append(imp)
            elif len(imp) == min_imp_size:
                aligned_group_imp.append(imp)
        group_imp = torch.stack(aligned_group_imp, dim=0)
        group_imp = self._reduce(group_imp)
        if self.normalizer is not None:
            group_imp = self.normalizer(group, group_imp)
        return group_imp


class SAAPImportance(tp.importance.Importance):
    """
    Practical SAAP-style importance for LLM-Pruner.

    Engineering approximation of the paper's pipeline:
    1) compute vector-wise importance (coarse structural salience)
    2) compute element-wise/taylor importance (fine-grained salience)
    3) calibrate each branch with a quadratic transform
    4) fuse branches with inverse-variance weighting
    5) align per-group scores to a comparable cross-layer scale

    Extra engineering stabilizers for better post-pruning accuracy:
    - score temperature / compression to avoid overly aggressive tail pruning
    - optional module-aware biasing (e.g. protect attention more than MLP)

    Notes:
    - This is a faithful, runnable approximation for the existing codebase.
    - We intentionally optimize for robust results instead of exact paper replication.
    """

    def __init__(
        self,
        vector_reduction: str = "mean",
        element_reduction: str = "sum",
        taylor: str = "param_first",
        beta_v=(0.0, 1.0, 0.0),
        beta_e=(0.0, 1.0, 0.0),
        var_eps: float = 1e-6,
        align_scores: bool = True,
        alignment_mode: str = "quantile",
        score_temperature: float = 1.0,
        score_floor_quantile: float = 0.0,
        module_score_bias: float = 0.0,
        use_grad_branch: bool = False,
        grad_branch_reduction: str = "mean",
        grad_branch_weight: float = 0.25,
        use_abs_grad_branch: bool = True,
        normalizer=None,
    ):
        self.vector_imp = llama_pruner.MagnitudeImportance(
            p=2, group_reduction=vector_reduction, normalizer=None
        )
        self.element_imp = llama_pruner.TaylorImportance(
            group_reduction=element_reduction, normalizer=None, taylor=taylor
        )
        self.beta_v = beta_v
        self.beta_e = beta_e
        self.var_eps = var_eps
        self.align_scores = align_scores
        self.alignment_mode = alignment_mode
        self.score_temperature = score_temperature
        self.score_floor_quantile = score_floor_quantile
        self.module_score_bias = module_score_bias
        self.use_grad_branch = use_grad_branch
        self.grad_branch_weight = grad_branch_weight
        self.grad_imp = GradientImportance(
            group_reduction=grad_branch_reduction,
            normalizer=None,
            use_abs_grad=use_abs_grad_branch,
        ) if use_grad_branch else None
        self.normalizer = normalizer

    def _quadratic_calibration(self, x: torch.Tensor, beta):
        b0, b1, b2 = beta
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        pred = b0 + b1 * x + b2 * (x ** 2)
        resid = pred - pred.mean()
        if resid.numel() <= 1:
            var = torch.full_like(pred, self.var_eps)
        else:
            scalar_var = resid.pow(2).mean().clamp_min(self.var_eps)
            var = torch.full_like(pred, scalar_var)
        return pred, var


    def _apply_module_bias(self, group, fused: torch.Tensor):
        if self.module_score_bias == 0.0 or group is None:
            return fused
        try:
            module = group[0][0].target.module
            module_name = module.__class__.__name__.lower()
        except Exception:
            return fused

        bias = 0.0
        if 'attention' in module_name:
            bias += abs(self.module_score_bias)
        elif 'mlp' in module_name:
            bias -= abs(self.module_score_bias)
        if bias == 0.0:
            return fused
        return fused + bias

    def _apply_score_regularization(self, fused: torch.Tensor):
        fused = torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)
        if self.score_temperature != 1.0:
            fused = torch.sign(fused) * torch.pow(fused.abs().clamp_min(1e-12), 1.0 / self.score_temperature)
        if self.score_floor_quantile > 0.0 and fused.numel() > 1:
            q = torch.quantile(fused, self.score_floor_quantile)
            fused = torch.maximum(fused, q)
        return fused

    @torch.no_grad()
    def __call__(self, group, ch_groups=1, consecutive_groups=1):
        v = self.vector_imp(group, ch_groups=ch_groups, consecutive_groups=consecutive_groups)
        e = self.element_imp(group, ch_groups=ch_groups, consecutive_groups=consecutive_groups)
        g = self.grad_imp(group, ch_groups=ch_groups, consecutive_groups=consecutive_groups) if self.grad_imp is not None else None

        if v is None and e is None and g is None:
            return None
        if v is None and e is None:
            fused = g
        elif v is None and g is None:
            fused = e
        elif e is None and g is None:
            fused = v
        else:
            v_pred = None if v is None else self._quadratic_calibration(v, self.beta_v)[0]
            e_pred = None if e is None else self._quadratic_calibration(e, self.beta_e)[0]
            g_pred = None if g is None else _safe_zscore(torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0))

            base = None
            if v_pred is not None and e_pred is not None:
                candidate_k = max(1, int(math.ceil(0.35 * v_pred.numel())))
                candidate_idx = torch.argsort(v_pred)[:candidate_k]
                base = v_pred.clone()
                if candidate_idx.numel() > 0:
                    e_centered = e_pred - e_pred.mean()
                    refine = e_centered[candidate_idx]
                    refine_scale = v_pred.abs().mean().clamp_min(self.var_eps)
                    base[candidate_idx] = v_pred[candidate_idx] + 0.35 * refine / refine.abs().mean().clamp_min(self.var_eps) * refine_scale
            elif v_pred is not None:
                base = v_pred
            elif e_pred is not None:
                base = e_pred

            if base is None:
                fused = g_pred
            elif g_pred is None:
                fused = base
            else:
                candidate_k = max(1, int(math.ceil(0.40 * base.numel())))
                candidate_idx = torch.argsort(base)[:candidate_k]
                fused = base.clone()
                grad_centered = g_pred - g_pred.mean()
                grad_scale = base.abs().mean().clamp_min(self.var_eps)
                fused[candidate_idx] = fused[candidate_idx] + self.grad_branch_weight * grad_centered[candidate_idx] / grad_centered[candidate_idx].abs().mean().clamp_min(self.var_eps) * grad_scale

        fused = self._apply_module_bias(group, fused)
        fused = self._apply_score_regularization(fused)
        if self.align_scores:
            if self.alignment_mode == 'quantile':
                fused = _quantile_map_to_normal(fused)
            elif self.alignment_mode == 'zscore':
                fused = _safe_zscore(fused)
            else:
                raise ValueError(f'Unsupported alignment_mode: {self.alignment_mode}')
        if self.normalizer is not None:
            fused = self.normalizer(group, fused)
        return fused
