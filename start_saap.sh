#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="/opt/anaconda3/envs/torch201-py39-cuda118/bin/python"
BASE_MODEL="${BASE_MODEL:-/home/easyai/llm_weights/decapoda-llama-7b-hf}"
SAVE_CKPT_LOG_NAME="${SAVE_CKPT_LOG_NAME:-prunellm}"
PRUNING_RATIO="${PRUNING_RATIO:-0.2}"

cd "$ROOT_DIR"
exec "$PYTHON_BIN" saap.py \
  --base_model "$BASE_MODEL" \
  --save_ckpt_log_name "$SAVE_CKPT_LOG_NAME" \
  --pruning_ratio "$PRUNING_RATIO"
