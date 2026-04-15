import copy
import time
import torch
import core.torch_pruning as tp
from core.pruner import hf_llama_pruner as llama_pruner
from core.pruner.saap_pruner import SAAPImportance
from core.models.hf_llama.modeling_llama import LlamaRMSNorm
from core.datasets.example_samples import get_examples
from ..utils import log_memory
from .call_layers import stage_entry, stage_route, stage_exec
from .structural_trace_runtime import _build_structural_importance_trace


@stage_exec
def _build_importance(args, logger):
    logger.log(f'[meta] build SAAPImportance | vector_reduction={args.saap_vector_reduction} | element_reduction={args.saap_element_reduction} | taylor={args.taylor}')
    logger.log('[meta] fusion_mode=coarse_to_fine_rerank | coarse=global_anchor | fine=rerank_on_bottom_35pct_candidates')
    structural_trace = _build_structural_importance_trace(args)
    logger.log(f"[meta][structural_importance] vec_var={structural_trace['vector']['variance']:.6f} | elem_var={structural_trace['element']['variance']:.6f} | fused_var={structural_trace['fused']['variance']:.6f} | aligned_n={len(structural_trace['aligned']['aligned'])}")
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
    return imp, structural_trace


@stage_exec
def _build_pruner(model, args, logger, attn_layers, mlp_layers, forward_prompts, imp):
    kwargs = {
        'importance': imp,
        'global_pruning': True,
        'iterative_steps': args.iterative_steps,
        'ch_sparsity': args.pruning_ratio,
        'ignored_layers': [],
        'channel_groups': {},
        'consecutive_groups': {layer.self_attn.q_proj: layer.self_attn.head_dim for layer in model.model.layers},
        'customized_pruners': {LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner},
        'root_module_types': None,
        'root_instances': [model.model.layers[i].self_attn.q_proj for i in attn_layers] + [model.model.layers[i].mlp.gate_proj for i in mlp_layers],
    }
    logger.log(f'[meta] init MetaPruner | global_pruning=True | ch_sparsity={args.pruning_ratio} | iterative_steps={args.iterative_steps}')
    return tp.pruner.MetaPruner(model, forward_prompts, **kwargs)


@stage_exec
def _load_calibration_batch(tokenizer, args, logger, i):
    logger.log(f'[meta] loading calibration dataset={args.dataset}, iter={i}, nsamples={args.num_examples}, seq_len={args.calibration_seq_len}')
    data_t0 = time.time()
    example_prompts = get_examples(args.dataset, tokenizer, args.num_examples, seq_len=args.calibration_seq_len).to(args.device)
    logger.log('Start Backwarding in iterative steps = {}...'.format(i))
    logger.log(f'[meta] calibration tensor shape = {tuple(example_prompts.shape)} | load_elapsed={time.time()-data_t0:.1f}s')
    log_memory(logger, f'iter_{i}_after_calibration_load')
    return example_prompts


@stage_exec
def _accumulate_param_mix_gradients(model, example_prompts, args, logger):
    if args.taylor not in ['param_mix', 'param_second']:
        return
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


@stage_exec
def _run_forward_backward_step(model, example_prompts, args, logger, i):
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


@stage_exec
def _run_pruner_step(pruner, model, args, logger, i):
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


@stage_route
def _run_single_meta_iteration(pruner, model, tokenizer, args, logger, i, inline_recovery_samples=None):
    example_prompts = _load_calibration_batch(tokenizer, args, logger, i)
    if inline_recovery_samples is None and getattr(args, 'run_inline_recovery_ce', False):
        pass
    _accumulate_param_mix_gradients(model, example_prompts, args, logger)
    _run_forward_backward_step(model, example_prompts, args, logger, i)
    _run_pruner_step(pruner, model, args, logger, i)


@stage_entry
def run_meta_prune_flow(model, tokenizer, args, logger, attn_layers, mlp_layers, forward_prompts, inline_recovery_samples=None):
    imp, structural_trace = _build_importance(args, logger)
    pruner = _build_pruner(model, args, logger, attn_layers, mlp_layers, forward_prompts, imp)
    model.zero_grad()
    for i in range(args.iterative_steps):
        _run_single_meta_iteration(pruner, model, tokenizer, args, logger, i, inline_recovery_samples=inline_recovery_samples)
    del pruner
