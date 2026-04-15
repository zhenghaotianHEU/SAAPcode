from .config import build_args, eval_switches
from .pruneflow.main_flow import main


def run(base_model='/home/easyai/llm_weights/decapoda-llama-7b-hf', save_ckpt_log_name='prunellm', pruning_ratio=0.2):
    args = build_args(base_model=base_model, save_ckpt_log_name=save_ckpt_log_name, pruning_ratio=pruning_ratio)
    main(args, eval_cfg=eval_switches())
