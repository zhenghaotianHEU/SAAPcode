import gc
import torch
from .bootstrap import init_runtime, load_model_and_tokenizer
from .pruning_stage import run_pruning_stage
from .save_eval_stage import save_and_eval_stage
from .after_prune_eval import run_after_prune_eval
from .pre_prune_eval import run_pre_prune_eval
from .inline_recovery import default_inline_recovery_tasks, default_inline_recovery_sample_count, default_inline_recovery_task_count_spec, default_inline_recovery_encrypt_offset, _parse_inline_recovery_task_counts, _format_inline_recovery_task_summary
from .recovery_flow import handle_stage2_full_ce
from .call_layers import stage_entry, stage_route, stage_exec
from .cuda11_bridge import register_cuda11_texture_surface_reference


@stage_exec
def _apply_quick_test_overrides(args, eval_cfg):
    if eval_cfg['quick_test']:
        args.num_examples = eval_cfg['quick_num_examples']
        args.calibration_seq_len = eval_cfg['quick_calibration_seq_len']
        args.max_seq_len = eval_cfg['quick_eval_seq_len']
        eval_cfg['skip_generation'] = True
        eval_cfg['skip_ppl'] = True
        eval_cfg['run_extra_eval'] = False
        eval_cfg['test_after_train'] = False
        eval_cfg['extra_eval_tasks'] = ''
        eval_cfg['quick_eval_datasets'] = [x.strip() for x in eval_cfg['quick_eval_datasets'].split(',') if x.strip()]
        if eval_cfg['quick_focus_layers']:
            args.block_attention_layer_start = max(args.block_attention_layer_start, eval_cfg['quick_layer_start'])
            args.block_attention_layer_end = min(args.block_attention_layer_end, eval_cfg['quick_layer_end'])
            args.block_mlp_layer_start = max(args.block_mlp_layer_start, eval_cfg['quick_layer_start'])
            args.block_mlp_layer_end = min(args.block_mlp_layer_end, eval_cfg['quick_layer_end'])


@stage_exec
def _prepare_inline_recovery_state(args, logger):
    inline_recovery_samples = None
    inline_recovery_sample_tasks = getattr(args, 'inline_recovery_tasks', default_inline_recovery_tasks())
    inline_recovery_sample_count = int(getattr(args, 'inline_recovery_max_samples_per_task', default_inline_recovery_sample_count()))
    inline_recovery_task_count_map = _parse_inline_recovery_task_counts(
        getattr(args, 'inline_recovery_task_counts', default_inline_recovery_task_count_spec()),
        inline_recovery_sample_tasks,
        inline_recovery_sample_count,
    )
    inline_recovery_sample_total, inline_recovery_ratio_map = _format_inline_recovery_task_summary(inline_recovery_task_count_map)
    if getattr(args, 'run_inline_recovery_ce', False):
        logger.log(
            f"[inline_recovery_ce][sample_summary] total={inline_recovery_sample_total} | task_count_map={inline_recovery_task_count_map} | ratio_map={inline_recovery_ratio_map} | encrypt_offset={int(getattr(args, 'inline_recovery_encrypt_offset', default_inline_recovery_encrypt_offset()))}"
        )
    return {
        'samples': inline_recovery_samples,
        'sample_tasks': inline_recovery_sample_tasks,
        'sample_count': inline_recovery_sample_count,
        'task_count_map': inline_recovery_task_count_map,
        'sample_total': inline_recovery_sample_total,
        'ratio_map': inline_recovery_ratio_map,
    }


@stage_exec
def _execute_full_prune_flow(args, eval_cfg):
    project_root, script_dir, log_root, logger = init_runtime(args, __file__)
    register_cuda11_texture_surface_reference(args=args, logger=logger)
    tokenizer, model = load_model_and_tokenizer(args, logger)
    _prepare_inline_recovery_state(args, logger)
    run_pre_prune_eval(model, tokenizer, args, logger, test_before_train=eval_cfg['test_before_train'])
    before_pruning_parameters, after_pruning_parameters = run_pruning_stage(model, tokenizer, args, logger)
    prune_only_ckpt_path, eval_model_dir = save_and_eval_stage(model, tokenizer, args, logger, before_pruning_parameters)
    if bool(getattr(args, 'run_stage2_full_ce_subprocess', False)):
        handle_stage2_full_ce(model, tokenizer, args, logger, prune_only_ckpt_path, before_pruning_parameters, project_root)
        return None
    run_after_prune_eval(model, tokenizer, args, logger, eval_cfg=eval_cfg, eval_model_dir=eval_model_dir)
    return None


@stage_route
def _route_main_flow(args, eval_cfg):
    _apply_quick_test_overrides(args, eval_cfg)
    return _execute_full_prune_flow(args, eval_cfg)


@stage_entry
def main(args, eval_cfg):
    return _route_main_flow(args, eval_cfg)
