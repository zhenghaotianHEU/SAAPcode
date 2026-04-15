import os
import gc
import json
import time
try:
    from eval_recovered_object_model import run_lm_eval as run_saved_model_lm_eval, run_ppl as run_saved_model_ppl
except Exception:
    run_saved_model_lm_eval = None
    run_saved_model_ppl = None
from .utils import project_root_from_file


def run_lm_eval(logger, args, project_root, eval_model_dir=None):
    return None


def run_auto_saved_model_eval(logger, model_dir, object_ckpt, output_subdir='eval_7task_ppl_object', eval_device='cuda', lm_eval_batch_size=64, ppl_batch_size=16, ppl_max_seq_len=128, file_path=__file__):
    if run_saved_model_lm_eval is None or run_saved_model_ppl is None:
        if logger is not None:
            logger.log('[auto_eval_after_save] skipped | reason=eval_recovered_object_model_missing')
        return None
    project_root = project_root_from_file(file_path)
    output_dir = os.path.join(model_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    summary = {
        'model_dir': model_dir,
        'object_ckpt': object_ckpt,
        'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    run_saved_model_ppl(object_ckpt, model_dir, output_dir, max_seq_len=ppl_max_seq_len, batch_size=ppl_batch_size, eval_device=eval_device)
    summary['ppl_json'] = os.path.join(output_dir, 'ppl.json')
    code, lm_json, lm_log = run_saved_model_lm_eval(model_dir, object_ckpt, eval_device, lm_eval_batch_size, output_dir, project_root)
    summary['lm_eval_returncode'] = code
    summary['lm_eval_json'] = lm_json
    summary['lm_eval_log'] = lm_log
    summary['finished_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    summary_path = os.path.join(output_dir, 'eval_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.log(f'[auto_eval_after_save] done | summary={summary_path}')
