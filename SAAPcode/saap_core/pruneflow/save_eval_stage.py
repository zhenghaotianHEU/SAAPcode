import os
import gc
import torch
from ..eval import run_auto_saved_model_eval
from .call_layers import stage_entry, stage_route, stage_exec


@stage_exec
def _resolve_save_paths(logger):
    eval_model_dir = None
    prune_only_ckpt_path = os.path.join(logger.sub_dir, 'pytorch_model_pruned.bin')
    return prune_only_ckpt_path, eval_model_dir


@stage_exec
def _save_pruned_checkpoint(model, tokenizer, prune_only_ckpt_path):
    model.half()
    torch.save({'model': model, 'tokenizer': tokenizer}, prune_only_ckpt_path)


@stage_exec
def _run_post_save_auto_eval(logger, args, prune_only_ckpt_path):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    run_auto_saved_model_eval(
        logger,
        model_dir=logger.sub_dir,
        object_ckpt=prune_only_ckpt_path,
        output_subdir='eval_after_prune',
        eval_device=getattr(args, 'auto_eval_device', args.eval_device),
        lm_eval_batch_size=int(getattr(args, 'auto_eval_lm_eval_batch_size', 32)),
        ppl_batch_size=int(getattr(args, 'auto_eval_ppl_batch_size', 16)),
        ppl_max_seq_len=int(getattr(args, 'auto_eval_ppl_max_seq_len', 128)),
    )


@stage_route
def _route_save_and_eval(model, tokenizer, args, logger, prune_only_ckpt_path):
    if bool(getattr(args, 'save_model', False)):
        _save_pruned_checkpoint(model, tokenizer, prune_only_ckpt_path)
        if bool(getattr(args, 'auto_eval_after_prune', True)):
            _run_post_save_auto_eval(logger, args, prune_only_ckpt_path)


@stage_entry
def save_and_eval_stage(model, tokenizer, args, logger, before_pruning_parameters):
    prune_only_ckpt_path, eval_model_dir = _resolve_save_paths(logger)
    _route_save_and_eval(model, tokenizer, args, logger, prune_only_ckpt_path)
    return prune_only_ckpt_path, eval_model_dir
