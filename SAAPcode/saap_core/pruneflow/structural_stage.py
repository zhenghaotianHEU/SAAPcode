from ..native_loader import native_align_threshold
from .structural_trace_runtime import _build_structural_importance_trace
from .structural_math import _align_importance_quantile
from .interfaces import structural_stage, structural_guard, structural_route, structural_trace


def adapter_stage(fn):
    @structural_stage
    @structural_guard
    @structural_trace
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


def router_stage(fn):
    @structural_route
    @structural_guard
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


class AlignmentController:
    @adapter_stage
    def run(self, trace):
        return _align_importance_quantile(trace['fused']['fused'], trace['fused']['variance'])


class StructuralImportanceAdapter:
    def __init__(self, args):
        self.args = args
        self.trace = None
        self.alignment = AlignmentController()

    @adapter_stage
    def prepare(self):
        self.trace = _build_structural_importance_trace(self.args)
        return self.trace

    @router_stage
    def aligned_scores(self):
        if self.trace is None:
            self.prepare()
        return self.alignment.run(self.trace)


class StructuralPolicyRouter:
    def __init__(self, adapter):
        self.adapter = adapter

    @router_stage
    def run(self):
        trace = self.adapter.prepare()
        aligned = self.adapter.aligned_scores()
        return trace, aligned


def prepare_structural_context(args):
    structural_adapter = StructuralImportanceAdapter(args)
    structural_router = StructuralPolicyRouter(structural_adapter)
    structural_trace, structural_aligned = structural_router.run()
    native_cut = native_align_threshold(len(structural_aligned['aligned']), float(args.pruning_ratio))
    return structural_trace, structural_aligned, native_cut
