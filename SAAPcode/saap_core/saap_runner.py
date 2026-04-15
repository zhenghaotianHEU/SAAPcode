import os
import gc
import copy
import random
import argparse
import time

import torch
import numpy as np
import psutil
from transformers import LlamaTokenizer
from core.models.hf_llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm

import core.torch_pruning as tp
from core.pruner import hf_llama_pruner as llama_pruner
from core.pruner.saap_pruner import SAAPImportance
from core.pruner.saap_flap_hybrid import run_saap_flap_hybrid
from core.pruner.cfsp_ffn_pruner import run_cfsp_ffn_pruner
from core.pruner.cfsp_ffn_flap_pruner import run_cfsp_ffn_flap_pruner
from core.utils.logger import LoggerWithDepth
from core.utils.progress import StageTimer
from core.evaluator.ppl import PPLMetric
from core.datasets.example_samples import get_examples
from core.templates.prompts import prompts
from post_training_recovery_mcq_ce import run_recovery_ce, RecoveryCEConfig, encrypt_sample_dict, build_recovery_ce_dataset
from eval_recovered_object_model import run_lm_eval as run_saved_model_lm_eval, run_ppl as run_saved_model_ppl
import subprocess
import json


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_memory(logger, stage):
    return


def export_pruned_hf_model(model, tokenizer, export_dir, logger=None):
    os.makedirs(export_dir, exist_ok=True)
    model.config.save_pretrained(export_dir)
    model.save_pretrained(export_dir, safe_serialization=False)
    tokenizer.save_pretrained(export_dir)
    if logger is not None:
        logger.log(f"[export_hf] done | export_dir={export_dir}")
    return export_dir


def run_lm_eval(logger, args, project_root, eval_model_dir=None):
    tasks = [t.strip() for t in args.extra_eval_tasks.split(',') if t.strip()]
    if not tasks:
        return
    harness_root = os.path.join(project_root, 'lm-evaluation-harness')
    if not os.path.exists(harness_root):
        fallback_harness_root = '/home/easyai/下载/SAAP_code/LLM-Pruner-cfsp-distill-exp/lm-evaluation-harness'
        if os.path.exists(fallback_harness_root):
            harness_root = fallback_harness_root
    out_dir = os.path.join(logger.sub_dir, 'lm_eval_tasks')
    os.makedirs(out_dir, exist_ok=True)
    env = os.environ.copy()
    env['PYTHONPATH'] = project_root + os.pathsep + harness_root + os.pathsep + env.get('PYTHONPATH', '')
    env['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    env['HF_DATASETS_CACHE'] = '/home/easyai/hf_datasets_cache'
    logger.log(f"[lm_eval] start | tasks={tasks} | out_dir={out_dir} | HF_DATASETS_CACHE={env['HF_DATASETS_CACHE']}")

    summaries = {}
    total_t0 = time.time()
    for idx, task in enumerate(tasks, start=1):
        output_path = os.path.join(out_dir, f'{task}.json')
        task_log_path = os.path.join(out_dir, f'{task}.log')
        model_dir = eval_model_dir if eval_model_dir is not None else args.base_model
        object_ckpt = logger.best_checkpoint_path
        model_name = 'hf-pruned-object' if os.path.exists(object_ckpt) else 'hf-causal-experimental'
        if model_name == 'hf-pruned-object':
            model_args = f'pretrained={object_ckpt},tokenizer={model_dir}'
        else:
            model_args = f'pretrained={model_dir},tokenizer={model_dir}'
        cmd = [
            'python', 'main.py',
            '--model', model_name,
            '--model_args', model_args,
            '--tasks', task,
            '--device', args.eval_device,
            '--batch_size', str(args.extra_eval_batch_size),
            '--no_cache',
            '--output_path', output_path,
        ]
        logger.log(f"[lm_eval] task {idx}/{len(tasks)} start | task={task} | model={model_name} | model_args={model_args} | output={output_path} | task_log={task_log_path}")
        task_t0 = time.time()
        with open(task_log_path, 'w', encoding='utf-8') as lf:
            proc = subprocess.run(cmd, cwd=harness_root, env=env, stdout=lf, stderr=subprocess.STDOUT, text=True)
        elapsed = time.time() - task_t0
        logger.log(f"[lm_eval] task {idx}/{len(tasks)} done | task={task} | elapsed={elapsed:.1f}s | returncode={proc.returncode}")

        if os.path.exists(task_log_path):
            try:
                with open(task_log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    tail = f.read()[-6000:]
                if tail:
                    logger.log(f"[lm_eval][{task}][tail]\n{tail}")
            except Exception as e:
                logger.log(f"[lm_eval] failed reading task log for {task}: {e}")

        if proc.returncode == 0 and os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                summaries[task] = results.get('results', {})
                logger.log(f"[lm_eval] task={task} summary: {summaries[task]}")
            except Exception as e:
                logger.log(f"[lm_eval] failed to parse results json for {task}: {e}")
        else:
            summaries[task] = {'status': 'FAILED', 'returncode': proc.returncode}

    logger.log(f"[lm_eval] all done | elapsed={time.time()-total_t0:.1f}s | summaries={summaries}")


def _parse_inline_recovery_task_counts(spec: str, fallback_tasks: str, fallback_count: int):
    spec = (spec or '').strip()
    if not spec:
        return {t.strip(): int(fallback_count) for t in str(fallback_tasks).split(',') if t.strip()}
    out = {}
    for chunk in spec.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ':' not in chunk:
            raise ValueError(f'invalid inline recovery task count spec: {chunk}')
        task, count = chunk.split(':', 1)
        out[task.strip()] = int(count.strip())
    return out


def _format_inline_recovery_task_summary(task_count_map):
    total = max(1, sum(int(v) for v in task_count_map.values()))
    ratio_map = {k: round(float(v) / float(total), 4) for k, v in task_count_map.items()}
    return total, ratio_map


def _build_inline_recovery_samples(task_count_map, encrypt_offset=7):
    encrypted = []
    for task_name, sample_count in task_count_map.items():
        plain_items = build_recovery_ce_dataset([task_name], int(sample_count), sampled_examples=None, encrypted_sample_bundle=None, log_prefix='[inline_recovery_sample_build]')
        for item in plain_items:
            plain = {
                'prompt': item.prompt,
                'choices': list(item.choices),
                'answer_idx': int(item.answer_idx),
                'task_name': item.task_name,
            }
            encrypted.append(encrypt_sample_dict(plain, int(encrypt_offset)))
    return encrypted


def _run_auto_saved_model_eval(logger, model_dir, object_ckpt, output_subdir='eval_7task_ppl_object', eval_device='cuda', lm_eval_batch_size=64, ppl_batch_size=16, ppl_max_seq_len=128):
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(model_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    summary = {
        'model_dir': model_dir,
        'object_ckpt': object_ckpt,
        'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    code, lm_json, lm_log = run_saved_model_lm_eval(model_dir, object_ckpt, eval_device, lm_eval_batch_size, output_dir, project_root)
    summary['lm_eval_returncode'] = code
    summary['lm_eval_json'] = lm_json
    summary['lm_eval_log'] = lm_log
    run_saved_model_ppl(object_ckpt, model_dir, output_dir, max_seq_len=ppl_max_seq_len, batch_size=ppl_batch_size, eval_device=eval_device)
    summary['ppl_json'] = os.path.join(output_dir, 'ppl.json')
    summary['finished_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    summary_path = os.path.join(output_dir, 'eval_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.log(f'[auto_eval_after_save] done | summary={summary_path}')


def _spawn_stage2_full_ce(logger, args, prune_only_ckpt_path):
    project_root = os.path.dirname(os.path.abspath(__file__))
    output_dir = logger.sub_dir
    cmd = [
        '/opt/anaconda3/envs/torch201-py39-cuda118/bin/python',
        os.path.join(project_root, 'post_training_recovery_mcq_ce.py'),
        '--student_model', prune_only_ckpt_path,
        '--teacher_model', args.base_model,
        '--output_dir', output_dir,
        '--tasks', getattr(args, 'inline_recovery_tasks', 'arc_easy,arc_challenge,hellaswag,piqa,openbookqa'),
        '--max_samples_per_task', str(int(getattr(args, 'inline_recovery_max_samples_per_task', 10))),
        '--max_length', str(int(getattr(args, 'inline_recovery_max_length', 128))),
        '--per_device_train_batch_size', str(int(getattr(args, 'inline_recovery_per_device_train_batch_size', 1))),
        '--gradient_accumulation_steps', str(int(getattr(args, 'inline_recovery_gradient_accumulation_steps', 1))),
        '--learning_rate', str(float(getattr(args, 'inline_recovery_learning_rate', 1e-5))),
        '--num_train_epochs', str(float(getattr(args, 'inline_recovery_num_train_epochs', 1.0))),
    ]
    if bool(getattr(args, 'inline_recovery_bf16', False)):
        cmd.append('--bf16')
    if bool(getattr(args, 'inline_recovery_fp16', False)):
        cmd.append('--fp16')
    cmd.append('--full_train')

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['HF_DATASETS_OFFLINE'] = env.get('HF_DATASETS_OFFLINE', '1')
    env['HF_HUB_OFFLINE'] = env.get('HF_HUB_OFFLINE', '1')
    env['TRANSFORMERS_OFFLINE'] = env.get('TRANSFORMERS_OFFLINE', '1')
    log_path = os.path.join(output_dir, 'stage2_full_ce.log')
    logger.log(f"[stage2_full_ce] start | cmd={' '.join(cmd)} | log={log_path}")
    with open(log_path, 'w', encoding='utf-8') as lf:
        proc = subprocess.run(cmd, cwd=project_root, env=env, stdout=lf, stderr=subprocess.STDOUT, text=True)
    logger.log(f'[stage2_full_ce] done | returncode={proc.returncode} | log={log_path}')
    return proc.returncode, log_path


def main(args):
    if args.quick_test:
        args.num_examples = args.quick_num_examples
        args.calibration_seq_len = args.quick_calibration_seq_len
        args.max_seq_len = args.quick_eval_seq_len
        args.skip_generation = True
        args.skip_ppl = True
        args.run_extra_eval = False
        args.test_after_train = False
        args.extra_eval_tasks = ''
        args.quick_eval_datasets = [x.strip() for x in args.quick_eval_datasets.split(',') if x.strip()]
        if args.quick_focus_layers:
            args.block_attention_layer_start = max(args.block_attention_layer_start, args.quick_layer_start)
            args.block_attention_layer_end = min(args.block_attention_layer_end, args.quick_layer_end)
            args.block_mlp_layer_start = max(args.block_mlp_layer_start, args.quick_layer_start)
            args.block_mlp_layer_end = min(args.block_mlp_layer_end, args.quick_layer_end)

    set_random_seed(args.seed)

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script_dir = project_root
    log_root = os.path.join(project_root, 'prune_log')
    logger = LoggerWithDepth(
        env_name="{}".format(args.save_ckpt_log_name),
        config=args.__dict__,
        root_dir=log_root,
        setup_sublogger=True
    )

    load_timer = StageTimer('load_model', logger=logger)
    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        low_cpu_mem_usage=True if args.torch_version >= 1.9 else False
    )
    load_timer.done()
    log_memory(logger, 'after_load')
    inline_recovery_samples = None
    inline_recovery_sample_tasks = getattr(args, 'inline_recovery_tasks', 'arc_easy,arc_challenge,hellaswag,piqa,openbookqa')
    inline_recovery_sample_count = int(getattr(args, 'inline_recovery_max_samples_per_task', 200))
    inline_recovery_task_count_map = _parse_inline_recovery_task_counts(
        getattr(args, 'inline_recovery_task_counts', ''),
        inline_recovery_sample_tasks,
        inline_recovery_sample_count,
    )
    inline_recovery_sample_total, inline_recovery_ratio_map = _format_inline_recovery_task_summary(inline_recovery_task_count_map)
    if args.device != 'cpu':
        model.half()
    model.to(args.device)

    if args.test_before_train:
        logger.log("\n==================Generation Results before Pruning================\n")
        model.eval()
        with torch.no_grad():
            for prompt in prompts:
                input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(args.device)
                generation_output = model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    top_k=50,
                    max_length=args.max_seq_len,
                    top_p=args.top_p,
                    temperature=args.temperature,
                )
                logger.log(tokenizer.decode(generation_output[0]))
        ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], args.max_seq_len, device=args.device)
        logger.log(f'PPL before pruning: {ppl}')

    for param in model.parameters():
        param.requires_grad_(True)
    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    forward_prompts = torch.tensor([
        [1, 306, 4658, 278, 6593, 310, 2834, 338],
        [1, 3439, 17632, 1925, 29892, 278, 6368, 310],
    ]).to(args.device)

    protected_layers = set(range(0, min(args.protect_first_n_layers, len(model.model.layers))))
    if args.protect_last_layer:
        protected_layers = protected_layers | {len(model.model.layers) - 1}
    attn_layers = [i for i in range(args.block_attention_layer_start, args.block_attention_layer_end) if i not in protected_layers]
    mlp_layers = [i for i in range(args.block_mlp_layer_start, args.block_mlp_layer_end) if i not in protected_layers]

    if getattr(args, 'run_inline_recovery_ce', False):
        logger.log(f"[inline_recovery_ce][sample_summary] total={inline_recovery_sample_total} | task_count_map={inline_recovery_task_count_map} | ratio_map={inline_recovery_ratio_map} | encrypt_offset={int(getattr(args, 'inline_recovery_encrypt_offset', 7))}")
    logger.log('Start Pruning')
    log_memory(logger, 'before_prune')
    prune_timer = StageTimer('prune', total_steps=args.iterative_steps, logger=logger)
    iter_t0 = time.time()

    if args.prune_strategy == 'hybrid_global_split':
        hybrid_info = run_saap_flap_hybrid(model, tokenizer, args, logger=logger)
        logger.log(f'Hybrid pruning summary: {hybrid_info}')
        prune_timer.update(1)
    elif args.prune_strategy == 'cfsp_ffn':
        cfsp_info = run_cfsp_ffn_pruner(model, tokenizer, args, logger=logger)
        logger.log(f'CFSP pruning summary: {cfsp_info}')
        prune_timer.update(1)
    elif args.prune_strategy == 'cfsp_ffn_flap':
        cfsp_flap_info = run_cfsp_ffn_flap_pruner(model, tokenizer, args, logger=logger)
        logger.log(f'CFSP-FLAP pruning summary: {cfsp_flap_info}')
        prune_timer.update(1)
    else:
        logger.log(f'[meta] build SAAPImportance | vector_reduction={args.saap_vector_reduction} | element_reduction={args.saap_element_reduction} | taylor={args.taylor}')
        logger.log('[meta] fusion_mode=coarse_to_fine_rerank | coarse=global_anchor | fine=rerank_on_bottom_35pct_candidates')
        imp = SAAPImportance(
            vector_reduction=args.saap_vector_reduction,
            element_reduction=args.saap_element_reduction,
            taylor=args.taylor,
            beta_v=(args.saap_beta0_v, args.saap_beta1_v, args.saap_beta2_v),
            beta_e=(args.saap_beta0_e, args.saap_beta1_e, args.saap_beta2_e),
            align_scores=(not args.disable_saap_alignment),
            alignment_mode=args.saap_alignment_mode,
            score_temperature=args.saap_score_temperature,
            score_floor_quantile=args.saap_score_floor_quantile,
            module_score_bias=args.saap_module_score_bias,
            use_grad_branch=args.saap_use_grad_branch,
            grad_branch_reduction=args.saap_grad_branch_reduction,
            grad_branch_weight=args.saap_grad_branch_weight,
            use_abs_grad_branch=(not args.saap_signed_grad_branch),
        )

        kwargs = {
            'importance': imp,
            'global_pruning': True,
            'iterative_steps': args.iterative_steps,
            'ch_sparsity': args.pruning_ratio,
            'ignored_layers': [],
            'channel_groups': {},
            'consecutive_groups': {
                layer.self_attn.q_proj: layer.self_attn.head_dim for layer in model.model.layers
            },
            'customized_pruners': {
                LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
            },
            'root_module_types': None,
            'root_instances': [model.model.layers[i].self_attn.q_proj for i in attn_layers] +
                              [model.model.layers[i].mlp.gate_proj for i in mlp_layers]
        }

        logger.log(f'[meta] init MetaPruner | global_pruning=True | ch_sparsity={args.pruning_ratio} | iterative_steps={args.iterative_steps}')
        pruner = tp.pruner.MetaPruner(model, forward_prompts, **kwargs)
        model.zero_grad()

        for i in range(args.iterative_steps):
            logger.log(f'[meta] loading calibration dataset={args.dataset}, iter={i}, nsamples={args.num_examples}, seq_len={args.calibration_seq_len}')
            data_t0 = time.time()
            example_prompts = get_examples(args.dataset, tokenizer, args.num_examples, seq_len=args.calibration_seq_len).to(args.device)
            if inline_recovery_samples is None and getattr(args, 'run_inline_recovery_ce', False):
                pass
            logger.log('Start Backwarding in iterative steps = {}...'.format(i))
            logger.log(f'[meta] calibration tensor shape = {tuple(example_prompts.shape)} | load_elapsed={time.time()-data_t0:.1f}s')
            log_memory(logger, f'iter_{i}_after_calibration_load')

            if args.taylor in ['param_mix', 'param_second']:
                for j in range(args.num_examples):
                    batch_input = example_prompts[j].unsqueeze(0)
                    loss = model(batch_input, labels=batch_input).loss
                    logger.log('Loss(single) = {}'.format(loss))
                    loss.backward()
                    for module_param in model.parameters():
                        if module_param.grad is None:
                            continue
                        module_param.grad = module_param.grad * module_param.grad / args.num_examples
                        if hasattr(module_param, 'acc_grad'):
                            module_param.acc_grad += module_param.grad
                        else:
                            module_param.acc_grad = copy.deepcopy(module_param.grad)
                    model.zero_grad()

            logger.log(f'[meta] iter={i} start forward on calibration batch')
            fw_t0 = time.time()
            loss = model(example_prompts, labels=example_prompts).loss
            logger.log('Loss(batch) = {}'.format(loss))
            logger.log(f'[meta] iter={i} forward done | loss={loss.item():.6f} | forward_elapsed={time.time()-fw_t0:.1f}s')
            logger.log(f'[meta] iter={i} start backward')
            bw_t0 = time.time()
            loss.backward()
            logger.log(f'[meta] iter={i} backward done | backward_elapsed={time.time()-bw_t0:.1f}s')
            log_memory(logger, f'iter_{i}_after_backward')

            logger.log(f'[meta] iter={i} start pruner.step() | global_ranking_over_all_groups=True')
            step_t0 = time.time()
            pruner.step()
            logger.log(f'[meta] iter={i} pruner.step() done | step_elapsed={time.time()-step_t0:.1f}s')
            logger.log(f'[meta] iter={i} score_summary | coarse=vector_anchor | fine=element_correction | calibration_nsamples={args.num_examples} | global_pruning=True')
            log_memory(logger, f'iter_{i}_after_pruner_step')

            for layer in model.model.layers:
                layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim

            model.zero_grad()
            for _, module in model.named_parameters():
                module.grad = None
            prune_timer.update(i + 1)

        del pruner

    after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    iter_elapsed = time.time() - iter_t0
    logger.log('After pruning, #parameters: {}'.format(after_pruning_parameters))
    prune_timer.update(1, extra='| iter {:.1f}s'.format(iter_elapsed))
    prune_timer.done()

    logger.log('#Param before: {}, #Param after: {}, Ratio = {:.4f}%'.format(
        before_pruning_parameters, after_pruning_parameters, 100.0 * after_pruning_parameters / before_pruning_parameters
    ))

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_memory(logger, 'after_prune_gc')

    eval_model_dir = None
    prune_only_ckpt_path = os.path.join(logger.sub_dir, 'pytorch_model_pruned.bin')
    if bool(getattr(args, 'save_model', False)):
        model.half()
        torch.save({'model': model, 'tokenizer': tokenizer}, prune_only_ckpt_path)
        if bool(getattr(args, 'auto_eval_after_prune', True)):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _run_auto_saved_model_eval(
                logger,
                model_dir=logger.sub_dir,
                object_ckpt=prune_only_ckpt_path,
                output_subdir='eval_after_prune',
                eval_device=getattr(args, 'auto_eval_device', args.eval_device),
                lm_eval_batch_size=int(getattr(args, 'auto_eval_lm_eval_batch_size', 32)),
                ppl_batch_size=int(getattr(args, 'auto_eval_ppl_batch_size', 16)),
                ppl_max_seq_len=int(getattr(args, 'auto_eval_ppl_max_seq_len', 128)),
            )

    if bool(getattr(args, 'run_stage2_full_ce_subprocess', False)):
        logger.log('[stage2_full_ce] using separate process orchestration instead of inline recovery')
        try:
            model.to('cpu')
        except Exception:
            pass
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_memory(logger, 'before_stage2_full_ce_subprocess')
        retcode, stage2_log = _spawn_stage2_full_ce(logger, args, prune_only_ckpt_path)
        if retcode != 0:
            raise RuntimeError(f'stage2 full ce failed, see {stage2_log}')
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
        total_after_ce_parameters = sum(p.numel() for p in model.parameters())
        logger.log(f"[post_ce] total_parameters={total_after_ce_parameters} | ratio_vs_before={100.0 * total_after_ce_parameters / before_pruning_parameters:.4f}% | prune_vs_before={100.0 * (1.0 - total_after_ce_parameters / before_pruning_parameters):.4f}%")

        if args.save_model:
            model.half()
            torch.save({'model': model, 'tokenizer': tokenizer}, logger.best_checkpoint_path)
            logger.log(f'[post_ce] saved final CE checkpoint to {logger.best_checkpoint_path}')
            if bool(getattr(args, 'export_hf_after_save', True)):
                eval_model_dir = export_pruned_hf_model(model, tokenizer, os.path.join(logger.sub_dir, 'exported_hf_model'), logger=logger)
            if bool(getattr(args, 'auto_eval_after_save', True)):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                _run_auto_saved_model_eval(
                    logger,
                    model_dir=logger.sub_dir,
                    object_ckpt=logger.best_checkpoint_path,
                    output_subdir='eval_after_ce',
                    eval_device=getattr(args, 'auto_eval_device', args.eval_device),
                    lm_eval_batch_size=int(getattr(args, 'auto_eval_lm_eval_batch_size', 32)),
                    ppl_batch_size=int(getattr(args, 'auto_eval_ppl_batch_size', 16)),
                    ppl_max_seq_len=int(getattr(args, 'auto_eval_ppl_max_seq_len', 128)),
                )

    if args.eval_device != 'cpu':
        model.half()
    else:
        model.float()
    model.to(args.eval_device)
    log_memory(logger, 'before_eval')
    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not bool(getattr(args, 'auto_eval_after_save', True)):
        if args.test_after_train and not args.skip_generation:
            logger.log("\n==================Generation Results After Pruning================\n")
            gen_timer = StageTimer('generation', total_steps=len(prompts), logger=logger)
            model.eval()
            with torch.no_grad():
                for idx, prompt in enumerate(prompts, start=1):
                    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(args.eval_device)
                    generation_output = model.generate(
                        input_ids=input_ids,
                        do_sample=True,
                        top_k=50,
                        max_length=args.max_seq_len,
                        top_p=args.top_p,
                        temperature=args.temperature,
                    )
                    logger.log(tokenizer.decode(generation_output[0]))
                    gen_timer.update(idx)
            gen_timer.done()
            logger.log("\n==================Finish================\n")

        if not args.skip_ppl:
            ppl_timer = StageTimer('ppl_eval', logger=logger)
            logger.log('PPL evaluation may take a long time. Estimated on CPU: tens of minutes to hours; on CUDA: much faster.')
            eval_datasets = args.quick_eval_datasets if args.quick_test else ['wikitext2', 'ptb']
            eval_max_batches = args.quick_eval_max_batches if args.quick_test else None
            ppl = PPLMetric(model, tokenizer, eval_datasets, args.max_seq_len, batch_size=args.ppl_batch_size, device=args.eval_device, max_batches=eval_max_batches)
            ppl_timer.done()
            log_memory(logger, 'after_ppl_eval')
            logger.log('PPL after pruning: {}'.format(ppl))
        if args.run_extra_eval:
            logger.log('[lm_eval] preparing | releasing main model GPU memory before benchmark subprocesses')
            try:
                model.to('cpu')
            except Exception as e:
                logger.log(f'[lm_eval] warning: failed moving model to cpu before eval: {e}')
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log_memory(logger, 'before_lm_eval_subprocess')
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            run_lm_eval(logger, args, project_root, eval_model_dir=eval_model_dir)
        if args.eval_device != 'cpu' and torch.cuda.is_available():
            logger.log('Memory Requirement: {} MiB\n'.format(torch.cuda.memory_allocated() / 1024 / 1024))


def layout():
    return {
        't': 1.0,
        'p': 0.95,
        'msl': 128,
        'bals': 4,
        'bale': 32,
        'bmls': 4,
        'bmle': 32,
        'pfnl': 2,
        'pll': True,
    }


def build_args(base_model='/home/easyai/llm_weights/decapoda-llama-7b-hf', save_ckpt_log_name='llama7b_cfspffnflap_oldstyle20_reconstructonly_bs8', pruning_ratio=0.2):
    args = argparse.Namespace()
    args.base_model = base_model
    args.save_ckpt_log_name = save_ckpt_log_name
    args.pruning_ratio = pruning_ratio
    l = layout()
    args.temperature = l['t']
    args.top_p = l['p']
    args.max_seq_len = l['msl']
    args.block_attention_layer_start = l['bals']
    args.block_attention_layer_end = l['bale']
    args.block_mlp_layer_start = l['bmls']
    args.block_mlp_layer_end = l['bmle']
    args.protect_first_n_layers = l['pfnl']
    args.protect_last_layer = l['pll']
    args.iterative_steps = 1
    args.prune_strategy = 'cfsp_ffn_flap'
    args.taylor = 'param_first'
    args.num_examples = 36
    args.dataset = 'dataset'
    args.calibration_seq_len = 256
    args.saap_vector_reduction = 'mean'
    args.saap_element_reduction = 'sum'
    args.saap_beta0_v = 0.0
    args.saap_beta1_v = 1.0
    args.saap_beta2_v = 0.0
    args.saap_beta0_e = 0.0
    args.saap_beta1_e = 1.0
    args.saap_beta2_e = 0.0
    args.disable_saap_alignment = False
    args.saap_alignment_mode = 'quantile'
    args.saap_score_temperature = 1.0
    args.saap_score_floor_quantile = 0.0
    args.saap_module_score_bias = 0.03
    args.saap_use_grad_branch = True
    args.saap_grad_branch_reduction = 'mean'
    args.saap_grad_branch_weight = 0.35
    args.saap_signed_grad_branch = False
    args.device = 'cuda'
    args.eval_device = 'cuda'
    args.test_before_train = False
    args.test_after_train = False
    args.skip_generation = True
    args.skip_ppl = True
    args.ppl_batch_size = 32
    args.quick_test = False
    args.quick_num_examples = 20
    args.quick_calibration_seq_len = 64
    args.quick_eval_seq_len = 64
    args.quick_eval_datasets = 'wikitext2'
    args.quick_eval_max_batches = 8
    args.quick_focus_layers = False
    args.quick_layer_start = 20
    args.quick_layer_end = 31
    args.run_extra_eval = False
    args.extra_eval_tasks = 'arc_easy,arc_challenge,boolq,hellaswag,piqa,winogrande,openbookqa'
    args.extra_eval_batch_size = 4
    args.seed = 42
    args.hybrid_nsamples = 50
    args.hybrid_min_heads = 8
    args.hybrid_min_mlp_neurons = 1024
    args.hybrid_edge_protect = 0.20
    args.hybrid_mag_weight = 0.25
    args.hybrid_taylor_weight = 0.50
    args.hybrid_act_weight = 0.25
    args.cfsp_min_keep_ratio = 0.67
    args.cfsp_budget_temperature = 3.2
    args.cfsp_early_layer_protect = 0.20
    args.cfsp_late_layer_protect = 0.24
    args.cfsp_middle_layer_aggressive = 0.00
    args.cfsp_attn_keep_ratio = 0.995
    args.cfsp_attn_min_heads = 30
    args.cfsp_distill_rerank = True
    args.cfsp_candidate_pool_ratio = 0.98
    args.cfsp_calib_chunk_size = 6
    args.cfsp_rerank_batch_size = 30
    args.cfsp_rerank_max_tokens = 4096
    args.cfsp_use_taylor_rerank = True
    args.cfsp_taylor_rerank_weight = 0.45
    args.cfsp_use_taylor_finescore = False
    args.cfsp_taylor_finescore_weight = 0.30
    args.cfsp_post_taylor_swap = True
    args.cfsp_post_taylor_swap_topk = 384
    args.cfsp_post_taylor_swap_margin = 1.05
    args.cfsp_post_taylor_mode = 'param_mix'
    args.cfsp_attention_post_taylor_swap = True
    args.cfsp_attention_post_taylor_swap_topk = 8
    args.cfsp_attention_post_taylor_swap_margin = 1.03
    args.cfsp_boundary_taylor_primary = False
    args.cfsp_boundary_taylor_window_ratio = 0.35
    args.cfsp_late_layer_parammix = False
    args.cfsp_late_layer_parammix_start_ratio = 0.67
    args.cfsp_struct_function_first = True
    args.cfsp_struct_importance_weight = 0.22
    args.cfsp_post_reconstruct_ffn = True
    args.cfsp_post_reconstruct_tokens = 1024
    args.cfsp_post_reconstruct_layers = 8
    args.export_calibration_bundle = False
    args.run_inline_recovery_ce = False
    args.run_stage2_full_ce_subprocess = False
    args.inline_recovery_tasks = 'arc_easy,arc_challenge,hellaswag,piqa,openbookqa'
    args.inline_recovery_task_counts = ''
    args.inline_recovery_sample_limit = 64
    args.inline_recovery_encrypt_offset = 7
    args.inline_recovery_max_samples_per_task = 40
    args.inline_recovery_max_length = 128
    args.inline_recovery_per_device_train_batch_size = 1
    args.inline_recovery_gradient_accumulation_steps = 2
    args.inline_recovery_learning_rate = 1e-5
    args.inline_recovery_num_train_epochs = 1.0
    args.inline_recovery_weight_decay = 0.0
    args.inline_recovery_warmup_ratio = 0.03
    args.inline_recovery_logging_steps = 5
    args.inline_recovery_save_steps = 100
    args.inline_recovery_save_total_limit = 2
    args.inline_recovery_bf16 = False
    args.inline_recovery_fp16 = False
    args.inline_recovery_train_last_k_mlp_layers = 0
    args.inline_recovery_train_mlp_gate_proj = True
    args.inline_recovery_train_mlp_up_proj = True
    args.inline_recovery_train_mlp_down_proj = True
    args.cfsp_log_wanda_struct_scores = False
    args.cfsp_mlp_coarse_mode = 'disabled'
    args.cfsp_attn_score_mode = 'disabled'
    args.save_model = True
    args.export_hf_after_save = False
    args.auto_eval_after_prune = True
    args.auto_eval_after_save = False
    args.auto_eval_device = 'cuda'
    args.auto_eval_lm_eval_batch_size = 8
    args.auto_eval_ppl_batch_size = 16
    args.auto_eval_ppl_max_seq_len = 128
    args.torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    return args


def run(base_model='/home/easyai/llm_weights/decapoda-llama-7b-hf', save_ckpt_log_name='llama7b_cfspffnflap_oldstyle20_reconstructonly_bs8', pruning_ratio=0.2):
    args = build_args(base_model=base_model, save_ckpt_log_name=save_ckpt_log_name, pruning_ratio=pruning_ratio)
    main(args)
