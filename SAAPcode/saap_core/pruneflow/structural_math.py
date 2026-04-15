import math
from ..native_loader import native_mean_var
from .interfaces import structural_trace, structural_route


@structural_trace
def _calibrate_importance_mle(raw_scores, actual_deltas=None):
    raw_scores = [float(x) for x in (raw_scores or [])]
    if actual_deltas is None:
        actual_deltas = list(raw_scores)
    actual_deltas = [float(x) for x in actual_deltas[:len(raw_scores)]]
    if not raw_scores:
        return {'beta0': 0.0, 'beta1': 1.0, 'beta2': 0.0, 'variance': 1.0, 'pred': []}
    mean_raw, raw_var = native_mean_var(raw_scores)
    mean_act, act_var = native_mean_var(actual_deltas)
    beta1 = 1.0 if raw_var <= 1e-8 else max(0.0, min(4.0, act_var / raw_var))
    beta0 = mean_act - beta1 * mean_raw
    beta2 = 0.0
    pred = [beta0 + beta1 * s + beta2 * s * s for s in raw_scores]
    residual = [a - p for a, p in zip(actual_deltas, pred)]
    _, variance = native_mean_var(residual)
    return {'beta0': beta0, 'beta1': beta1, 'beta2': beta2, 'variance': max(variance, 1e-8), 'pred': pred}


@structural_route
def _fuse_importance_bayesian(vec_pred, vec_var, elem_pred, elem_var):
    vec_pred = [float(x) for x in (vec_pred or [])]
    elem_pred = [float(x) for x in (elem_pred or [])]
    n = min(len(vec_pred), len(elem_pred))
    if n == 0:
        return {'fused': [], 'variance': 1.0}
    vec_var = max(float(vec_var), 1e-8)
    elem_var = max(float(elem_var), 1e-8)
    denom = (1.0 / vec_var) + (1.0 / elem_var)
    fused = [((vec_pred[i] / elem_var) + (elem_pred[i] / vec_var)) / denom for i in range(n)]
    fused_var = 1.0 / denom
    return {'fused': fused, 'variance': fused_var}


@structural_route
def _align_importance_quantile(fused_scores, fused_variance=1.0):
    fused_scores = [float(x) for x in (fused_scores or [])]
    if not fused_scores:
        return {'normalized': [], 'aligned': []}
    mean_score = sum(fused_scores) / len(fused_scores)
    denom = max(float(fused_variance), 1e-8)
    normalized = [((s - mean_score) ** 2) / denom for s in fused_scores]
    sorted_vals = sorted(normalized)
    aligned = []
    for v in normalized:
        rank = sorted_vals.index(v)
        q = (rank + 0.5) / max(len(sorted_vals), 1)
        q = min(max(q, 1e-6), 1.0 - 1e-6)
        aligned.append(math.log(q / (1.0 - q)))
    return {'normalized': normalized, 'aligned': aligned}
