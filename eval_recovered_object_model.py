import os
import gc
import json
import time
import argparse
import subprocess

import torch
from transformers import AutoTokenizer

from core.evaluator.ppl import PPLMetric


def _rewrite_lm_eval_json_minimal(output_path):
    if not os.path.exists(output_path):
        return
    with open(output_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    results = payload.get('results', {}) if isinstance(payload, dict) else {}
    minimal = {}
    for task, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        best = None
        for key in ('acc', 'acc_norm', 'acc,none', 'acc_norm,none'):
            val = metrics.get(key)
            if isinstance(val, (int, float)):
                best = float(val) if best is None else max(best, float(val))
        if best is not None:
            minimal[task] = best
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(minimal, f, ensure_ascii=False, indent=2)


def _trim_lm_eval_log_after_progress(log_path):
    if not os.path.exists(log_path):
        return
    text = open(log_path, 'r', encoding='utf-8', errors='ignore').read()
    marker = '100%|██████████|'
    idx = text.rfind(marker)
    if idx == -1:
        return
    end = text.find('\n', idx)
    if end == -1:
        end = len(text)
    trimmed = text[:end + 1]
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(trimmed)


def run_lm_eval(model_dir, object_ckpt, eval_device, batch_size, output_dir, project_root):
    tasks = ['arc_easy', 'arc_challenge', 'boolq', 'hellaswag', 'piqa', 'winogrande', 'openbookqa']
    harness_root = os.path.join(project_root, 'lm-evaluation-harness')
    if not os.path.exists(harness_root):
        fallback_harness_root = '/home/easyai/下载/SAAP_code/LLM-Pruner-cfsp-distill-exp/lm-evaluation-harness'
        if os.path.exists(fallback_harness_root):
            harness_root = fallback_harness_root

    os.makedirs(output_dir, exist_ok=True)
    env = os.environ.copy()
    env['PYTHONPATH'] = project_root + os.pathsep + harness_root + os.pathsep + env.get('PYTHONPATH', '')
    env['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    env['HF_DATASETS_CACHE'] = '/home/easyai/hf_datasets_cache'
    env['HF_DATASETS_OFFLINE'] = env.get('HF_DATASETS_OFFLINE', '1')
    env['HF_HUB_OFFLINE'] = env.get('HF_HUB_OFFLINE', '1')
    env['TRANSFORMERS_OFFLINE'] = env.get('TRANSFORMERS_OFFLINE', '1')

    output_path = os.path.join(output_dir, 'lm_eval_7tasks.json')
    log_path = os.path.join(output_dir, 'lm_eval_7tasks.log')
    cmd = [
        '/opt/anaconda3/envs/torch201-py39-cuda118/bin/python', 'main.py',
        '--model', 'hf-pruned-object',
        '--model_args', f'pretrained={object_ckpt},tokenizer={model_dir}',
        '--tasks', ','.join(tasks),
        '--device', eval_device,
        '--batch_size', str(batch_size),
        '--no_cache',
        '--output_path', output_path,
    ]
    with open(log_path, 'w', encoding='utf-8') as lf:
        proc = subprocess.run(cmd, cwd=harness_root, env=env, stdout=lf, stderr=subprocess.STDOUT, text=True)
    if proc.returncode == 0:
        _rewrite_lm_eval_json_minimal(output_path)
        _trim_lm_eval_log_after_progress(log_path)
    return proc.returncode, output_path, log_path


def run_ppl(object_ckpt, model_dir, output_dir, max_seq_len=128, batch_size=16, eval_device='cuda'):
    payload = torch.load(object_ckpt, map_location='cpu')
    model = payload['model']
    tokenizer = payload.get('tokenizer', None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id < 0:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.half().to(eval_device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], max_seq_len, batch_size=batch_size, device=eval_device)
    out_path = os.path.join(output_dir, 'ppl.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'ppl': str(ppl)}, f, ensure_ascii=False, indent=2)

    try:
        model.to('cpu')
    except Exception:
        pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--object_ckpt', type=str, default='')
    parser.add_argument('--eval_device', type=str, default='cuda')
    parser.add_argument('--lm_eval_batch_size', type=int, default=8)
    parser.add_argument('--ppl_batch_size', type=int, default=16)
    parser.add_argument('--ppl_max_seq_len', type=int, default=128)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.abspath(args.model_dir)
    object_ckpt = args.object_ckpt.strip() or os.path.join(model_dir, 'recovered_model_object.bin')
    output_dir = os.path.join(model_dir, 'eval_7task_ppl_object')
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        'model_dir': model_dir,
        'object_ckpt': object_ckpt,
        'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    code, lm_json, lm_log = run_lm_eval(model_dir, object_ckpt, args.eval_device, args.lm_eval_batch_size, output_dir, project_root)
    summary['lm_eval_returncode'] = code
    summary['lm_eval_json'] = lm_json
    summary['lm_eval_log'] = lm_log

    run_ppl(object_ckpt, model_dir, output_dir, max_seq_len=args.ppl_max_seq_len, batch_size=args.ppl_batch_size, eval_device=args.eval_device)
    summary['ppl_json'] = os.path.join(output_dir, 'ppl.json')
    summary['finished_at'] = time.strftime('%Y-%m-%d %H:%M:%S')

    summary_path = os.path.join(output_dir, 'eval_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
