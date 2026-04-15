import os
import gc
import subprocess
import torch
from ..eval import run_auto_saved_model_eval
from .call_layers import stage_entry, stage_route, stage_exec
from .inline_recovery import default_inline_recovery_tasks, default_inline_recovery_sample_count, default_inline_recovery_sample_limit, default_inline_recovery_max_length


@stage_exec
def export_pruned_hf_model(model, tokenizer, export_dir, logger=None):
    os.makedirs(export_dir, exist_ok=True)
    model.config.save_pretrained(export_dir)
    model.save_pretrained(export_dir, safe_serialization=False)
    tokenizer.save_pretrained(export_dir)
    if logger is not None:
        logger.log(f"[export_hf] done | export_dir={export_dir}")
    return export_dir


@stage_exec
def _build_stage2_command(logger, args, prune_only_ckpt_path, project_root):
    output_dir = logger.sub_dir
    cmd = [
        '/opt/anaconda3/envs/torch201-py39-cuda118/bin/python',
        os.path.join(project_root, 'post_training_recovery_mcq_ce.py'),
        '--student_model', prune_only_ckpt_path,
        '--teacher_model', args.base_model,
        '--output_dir', output_dir,
        '--tasks', getattr(args, 'inline_recovery_tasks', default_inline_recovery_tasks()),
        '--max_samples_per_task', str(int(getattr(args, 'inline_recovery_max_samples_per_task', default_inline_recovery_sample_count()))),
        '--max_length', str(int(getattr(args, 'inline_recovery_max_length', default_inline_recovery_max_length()))),
        '--per_device_train_batch_size', str(int(getattr(args, 'inline_recovery_per_device_train_batch_size', 1))),
        '--gradient_accumulation_steps', str(int(getattr(args, 'inline_recovery_gradient_accumulation_steps', 1))),
        '--learning_rate', str(float(getattr(args, 'inline_recovery_learning_rate', 1e-5))),
        '--num_train_epochs', str(float(getattr(args, 'inline_recovery_num_train_epochs', 1.0))),
        '--full_train',
    ]
    if bool(getattr(args, 'inline_recovery_bf16', False)):
        cmd.append('--bf16')
    if bool(getattr(args, 'inline_recovery_fp16', False)):
        cmd.append('--fp16')
    return cmd


@stage_exec
def _build_stage2_env():
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['HF_DATASETS_OFFLINE'] = env.get('HF_DATASETS_OFFLINE', '1')
    env['HF_HUB_OFFLINE'] = env.get('HF_HUB_OFFLINE', '1')
    env['TRANSFORMERS_OFFLINE'] = env.get('TRANSFORMERS_OFFLINE', '1')
    return env


@stage_exec
def _run_stage2_subprocess(logger, cmd, project_root):
    output_dir = logger.sub_dir
    env = _build_stage2_env()
    log_path = os.path.join(output_dir, 'stage2_full_ce.log')
    logger.log(f"[stage2_full_ce] start | cmd={' '.join(cmd)} | log={log_path}")
    with open(log_path, 'w', encoding='utf-8') as lf:
        proc = subprocess.run(cmd, cwd=project_root, env=env, stdout=lf, stderr=subprocess.STDOUT, text=True)
    logger.log(f'[stage2_full_ce] done | returncode={proc.returncode} | log={log_path}')
    return proc.returncode, log_path


@stage_entry
def spawn_stage2_full_ce(logger, args, prune_only_ckpt_path, project_root):
    cmd = _build_stage2_command(logger, args, prune_only_ckpt_path, project_root)
    return _run_stage2_subprocess(logger, cmd, project_root)


@stage_exec
def _release_model_memory(model):
    try:
        model.to('cpu')
    except Exception:
        pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@stage_exec
def _load_stage2_final_payload(logger, tokenizer):
    final_object_ckpt = os.path.join(logger.sub_dir, 'recovered_model_object.bin')
    if not os.path.exists(final_object_ckpt):
        raise RuntimeError(f'stage2 full ce finished but missing final object checkpoint: {final_object_ckpt}')
    final_payload = torch.load(final_object_ckpt, map_location='cpu')
    model = final_payload['model']
    tokenizer = final_payload.get('tokenizer', tokenizer)
    del final_payload
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, tokenizer


@stage_exec
def _save_and_eval_recovered_model(model, tokenizer, args, logger):
    if args.save_model:
        model.half()
        torch.save({'model': model, 'tokenizer': tokenizer}, logger.best_checkpoint_path)
        logger.log(f'[post_ce] saved final CE checkpoint to {logger.best_checkpoint_path}')
        if bool(getattr(args, 'export_hf_after_save', True)):
            export_pruned_hf_model(model, tokenizer, os.path.join(logger.sub_dir, 'exported_hf_model'), logger=logger)
        if bool(getattr(args, 'auto_eval_after_save', True)):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            run_auto_saved_model_eval(logger, model_dir=logger.sub_dir, object_ckpt=logger.best_checkpoint_path, output_subdir='eval_after_ce', eval_device=getattr(args, 'auto_eval_device', args.eval_device), lm_eval_batch_size=int(getattr(args, 'auto_eval_lm_eval_batch_size', 32)), ppl_batch_size=int(getattr(args, 'auto_eval_ppl_batch_size', 16)), ppl_max_seq_len=int(getattr(args, 'auto_eval_ppl_max_seq_len', 128)))


@stage_route
def _route_stage2_postprocess(model, tokenizer, args, logger, before_pruning_parameters):
    model, tokenizer = _load_stage2_final_payload(logger, tokenizer)
    total_after_ce_parameters = sum(p.numel() for p in model.parameters())
    logger.log(f"[post_ce] total_parameters={total_after_ce_parameters} | ratio_vs_before={100.0 * total_after_ce_parameters / before_pruning_parameters:.4f}% | prune_vs_before={100.0 * (1.0 - total_after_ce_parameters / before_pruning_parameters):.4f}%")
    _save_and_eval_recovered_model(model, tokenizer, args, logger)


@stage_entry
def handle_stage2_full_ce(model, tokenizer, args, logger, prune_only_ckpt_path, before_pruning_parameters, project_root):
    logger.log('[stage2_full_ce] using separate process orchestration instead of inline recovery')
    _release_model_memory(model)
    retcode, stage2_log = spawn_stage2_full_ce(logger, args, prune_only_ckpt_path, project_root)
    if retcode != 0:
        raise RuntimeError(f'stage2 full ce failed, see {stage2_log}')
    _route_stage2_postprocess(model, tokenizer, args, logger, before_pruning_parameters)
