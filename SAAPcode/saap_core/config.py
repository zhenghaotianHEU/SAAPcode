import argparse
import torch
from .config_layout import layout, apply_layout
from .config_structural import apply_structural
from .config_pruning import apply_pruning
from .config_runtime import apply_runtime


def eval_switches():
    return {
        'test_before_train': False,
        'test_after_train': False,
        'skip_generation': True,
        'skip_ppl': True,
        'ppl_batch_size': 32,
        'quick_test': False,
        'quick_num_examples': 20,
        'quick_calibration_seq_len': 64,
        'quick_eval_seq_len': 64,
        'quick_eval_datasets': 'wikitext2',
        'quick_eval_max_batches': 8,
        'quick_focus_layers': False,
        'quick_layer_start': 20,
        'quick_layer_end': 31,
        'run_extra_eval': False,
        'extra_eval_tasks': 'arc_easy,arc_challenge,boolq,hellaswag,piqa,winogrande,openbookqa',
    }


def build_args(base_model='/home/easyai/llm_weights/decapoda-llama-7b-hf', save_ckpt_log_name='prunellm', pruning_ratio=0.2):
    args = argparse.Namespace()
    args.base_model = base_model
    args.save_ckpt_log_name = save_ckpt_log_name
    args.pruning_ratio = pruning_ratio
    args = apply_layout(args)
    args = apply_structural(args)
    args = apply_pruning(args)
    args = apply_runtime(args)
    args.torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    return args
