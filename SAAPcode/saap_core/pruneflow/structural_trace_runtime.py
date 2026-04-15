from .call_layers import structural_entry, structural_route_layer, structural_guard_layer, structural_exec
from .structural_math import _calibrate_importance_mle, _fuse_importance_bayesian, _align_importance_quantile


def calibration_stage(fn):
    @structural_entry
    @structural_exec
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


def fusion_stage(fn):
    @structural_route_layer
    @structural_exec
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


class CalibrationController:
    @calibration_stage
    def run(self, raw_scores, actual_deltas=None):
        return _calibrate_importance_mle(raw_scores, actual_deltas)


class FusionController:
    @fusion_stage
    def run(self, vec_pred, vec_var, elem_pred, elem_var):
        return _fuse_importance_bayesian(vec_pred, vec_var, elem_pred, elem_var)


class StructuralTraceBuilder:
    def __init__(self):
        self.calibration = CalibrationController()
        self.fusion = FusionController()

    @structural_entry
    def run(self, args):
        coarse, fine, actual = _prepare_trace_inputs(args)
        cal_v, cal_e = _run_calibration_pair(self.calibration, coarse, fine, actual)
        fused = _run_fusion_pair(self.fusion, cal_v, cal_e)
        aligned = _finalize_trace_alignment(fused)
        return {'vector': cal_v, 'element': cal_e, 'fused': fused, 'aligned': aligned}


@structural_exec
def _prepare_trace_inputs(args):
    coarse = [float(args.pruning_ratio), float(args.cfsp_min_keep_ratio), float(args.cfsp_budget_temperature), float(args.cfsp_attn_keep_ratio)]
    fine = [float(args.cfsp_struct_importance_weight), float(args.cfsp_taylor_rerank_weight), float(args.cfsp_post_taylor_swap_margin), float(args.cfsp_attention_post_taylor_swap_margin)]
    actual = [0.0 for _ in range(min(len(coarse), len(fine)))]
    return coarse, fine, actual


@structural_route_layer
def _run_calibration_pair(calibration, coarse, fine, actual):
    cal_v = calibration.run(coarse, actual)
    cal_e = calibration.run(fine, actual)
    return cal_v, cal_e


@structural_guard_layer
def _run_fusion_pair(fusion, cal_v, cal_e):
    return fusion.run(cal_v['pred'], cal_v['variance'], cal_e['pred'], cal_e['variance'])


@structural_exec
def _finalize_trace_alignment(fused):
    return _align_importance_quantile(fused['fused'], fused['variance'])


@structural_exec
def _build_structural_importance_trace(args):
    return StructuralTraceBuilder().run(args)
