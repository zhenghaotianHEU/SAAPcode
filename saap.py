import argparse
from core.saap_core.runner import run

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', type=str, default='/home/easyai/llm_weights/decapoda-llama-7b-hf')
parser.add_argument('--save_ckpt_log_name', type=str, default='prunellm')
parser.add_argument('--pruning_ratio', type=float, default=0.2)
args = parser.parse_args()
run(base_model=args.base_model, save_ckpt_log_name=args.save_ckpt_log_name, pruning_ratio=args.pruning_ratio)
