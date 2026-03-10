#!/bin/bash
# ==================================================================
# Evaluate original autoregressive decoding (no speculative decoding)
# Used as the 1.0x speedup baseline.
#
# Usage:
#   cd Efficient-Draft-Adaptation
#   bash scripts/eval_original.sh
# ==================================================================

set -e

# ─── MODIFY THESE ────────────────────────────────────────────
BASE_MODEL="/path/to/Qwen2.5-Math-7B"
TASK="gsm8k"
TEMPERATURE=0.0
GPU="0"
MODEL_ID="original_${TASK}"
OUTPUT_DIR="results/${MODEL_ID}"
# ─────────────────────────────────────────────────────────────

export PYTHONPATH=$(pwd):$PYTHONPATH
mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "Original Autoregressive Decoding"
echo "Base model : ${BASE_MODEL}"
echo "Task       : ${TASK}"
echo "============================================================"

CUDA_VISIBLE_DEVICES=${GPU} python -m eda.evaluation.eval_original \
    --base-model-path "${BASE_MODEL}" \
    --bench-name "${TASK}" \
    --model-id "${MODEL_ID}" \
    --temperature "${TEMPERATURE}" \
    --answer-file "${OUTPUT_DIR}/${MODEL_ID}.jsonl" \
    2>&1 | tee "${OUTPUT_DIR}/${MODEL_ID}.log"
