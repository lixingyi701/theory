#!/usr/bin/env bash
# Activate the dedicated theory Conda environment before running this script.
set -euo pipefail
: "${THEORY_MODEL_DIR:?Set THEORY_MODEL_DIR to the pinned Qwen3-8B directory}"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python - <<'PY'
import importlib.metadata as m
assert m.version('vllm') == '0.10.2'
assert m.version('torch').split('+')[0] == '2.8.0'
PY
exec vllm serve "$THEORY_MODEL_DIR" --served-model-name qwen3-8b \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 --max-model-len 32768 \
  --gpu-memory-utilization 0.85 --tensor-parallel-size 1 --max-num-seqs 4 \
  --generation-config vllm --enable-prefix-caching
