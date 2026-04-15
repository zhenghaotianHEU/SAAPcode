import time
import torch
from core.pruner.saap_struct_hybrid import run_saap_struct_hybrid
from core.pruner.cfsp_ffn_pruner import run_cfsp_ffn_pruner
from core.pruner.cfsp_ffn_struct_pruner import run_cfsp_ffn_struct_pruner
from core.utils.progress import StageTimer
from ..utils import log_memory
from .meta_prune_runtime import run_meta_prune_flow
from .call_layers import stage_entry, stage_route, stage_exec
from .structural_stage import prepare_structural_context


@stage_exec
def _enable_trainable_params(model):
    for param in model.parameters():
        param.requires_grad_(True)


@stage_exec
def _count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@stage_exec
def _build_forward_prompts(args):
    return torch.tensor([[1, 306, 4658, 278, 6593, 310, 2834, 338], [1, 3439, 17632, 1925, 29892, 278, 6368, 310]]).to(args.device)


@stage_exec
def _resolve_layer_groups(model, args):
    protected_layers = set(range(0, min(args.protect_first_n_layers, len(model.model.layers))))
    if args.protect_last_layer:
        protected_layers = protected_layers | {len(model.model.layers) - 1}
    attn_layers = [i for i in range(args.block_attention_layer_start, args.block_attention_layer_end) if i not in protected_layers]
    mlp_layers = [i for i in range(args.block_mlp_layer_start, args.block_mlp_layer_end) if i not in protected_layers]
    return protected_layers, attn_layers, mlp_layers


@stage_exec
def _run_direct_prune_strategy(model, tokenizer, args, logger, prune_timer):
    if args.prune_strategy == 'hybrid_global_split':
        summary = run_saap_struct_hybrid(model, tokenizer, args, logger=logger)
        logger.log(f'Hybrid pruning summary: {summary}')
        prune_timer.update(1)
        return True
    if args.prune_strategy == 'cfsp_ffn':
        summary = run_cfsp_ffn_pruner(model, tokenizer, args, logger=logger)
        logger.log(f'CFSP pruning summary: {summary}')
        prune_timer.update(1)
        return True
    if args.prune_strategy == 'cfsp_ffn_flap':
        run_cfsp_ffn_struct_pruner(model, tokenizer, args, logger=logger)
        prune_timer.update(1)
        return True
    return False


@stage_exec
def _run_meta_or_direct_pruning(model, tokenizer, args, logger, attn_layers, mlp_layers, forward_prompts, prune_timer):
    if _run_direct_prune_strategy(model, tokenizer, args, logger, prune_timer):
        return
    run_meta_prune_flow(model, tokenizer, args, logger, attn_layers, mlp_layers, forward_prompts)


@stage_exec
def _log_structural_context(logger, structural_trace, structural_aligned, native_cut):
    return None


@stage_route
def _route_pruning_strategy(model, tokenizer, args, logger, attn_layers, mlp_layers, forward_prompts, prune_timer):
    if logger is not None:
        logger.log('prune_probe: before_pruning_strategy')
    return _run_meta_or_direct_pruning(model, tokenizer, args, logger, attn_layers, mlp_layers, forward_prompts, prune_timer)


@stage_exec
def _finalize_pruning(logger, before_pruning_parameters, after_pruning_parameters, prune_timer, iter_elapsed):
    prune_timer.update(1, extra='| iter {:.1f}s'.format(iter_elapsed))
    prune_timer.done()


@stage_entry
def run_pruning_stage(model, tokenizer, args, logger):
    _enable_trainable_params(model)
    before_pruning_parameters = _count_trainable_parameters(model)
    forward_prompts = _build_forward_prompts(args)
    protected_layers, attn_layers, mlp_layers = _resolve_layer_groups(model, args)
    logger.log('Start Pruning')
    structural_trace, structural_aligned, native_cut = prepare_structural_context(args)
    if logger is not None:
        logger.log('prune_probe: after_structural_context')
    _log_structural_context(logger, structural_trace, structural_aligned, native_cut)
    log_memory(logger, 'before_prune')
    prune_timer = StageTimer('prune', total_steps=args.iterative_steps, logger=logger)
    iter_t0 = time.time()
    _route_pruning_strategy(model, tokenizer, args, logger, attn_layers, mlp_layers, forward_prompts, prune_timer)
    after_pruning_parameters = _count_trainable_parameters(model)
    iter_elapsed = time.time() - iter_t0
    _finalize_pruning(logger, before_pruning_parameters, after_pruning_parameters, prune_timer, iter_elapsed)
    log_memory(logger, 'after_prune_gc')
    return before_pruning_parameters, after_pruning_parameters
