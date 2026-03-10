#!/bin/bash
# ==================================================================
# Evaluate EAGLE Base Transfer (base draft model applied to domain target)
# ==================================================================
# Uses the Stage 1 (base) draft model checkpoint directly on the
# domain target model, without Stage 2 fine-tuning.
# This serves as the "base transfer" baseline in the paper.
#
# Usage:
#   cd Efficient-Draft-Adaptation
#   bash scripts/eval_base_transfer.sh
# ==================================================================

set -e

# ─── MODIFY THESE ────────────────────────────────────────────
BASE_MODEL="/path/to/Qwen2.5-Math-7B"

# Stage 1 base checkpoint (not domain-finetuned)
EAGLE_DIR="/path/to/checkpoints/stage1_base"
EPOCH=19

TASK="gsm8k"
TEMPERATURE=0.0
GPU="0"
MODEL_ID="base_transfer_${TASK}"
OUTPUT_DIR="results/${MODEL_ID}"
# ─────────────────────────────────────────────────────────────

export PYTHONPATH=$(pwd):$PYTHONPATH
mkdir -p "${OUTPUT_DIR}"

CKPT="${EAGLE_DIR}/state_${EPOCH}"
if [ ! -d "${CKPT}" ]; then
    echo "Error: checkpoint not found at ${CKPT}"
    exit 1
fi

echo "============================================================"
echo "EAGLE Base Transfer Evaluation"
echo "Base model : ${BASE_MODEL}"
echo "Draft ckpt : ${CKPT}"
echo "Task       : ${TASK}"
echo "============================================================"

CUDA_VISIBLE_DEVICES=${GPU} python -m eda.evaluation.gen_ea_answer_qwen2 \
    --base-model-path "${BASE_MODEL}" \
    --ea-model-path "${CKPT}" \
    --model-id "${MODEL_ID}" \
    --bench-name "${TASK}" \
    --temperature "${TEMPERATURE}" \
    --answer-file "${OUTPUT_DIR}/${MODEL_ID}.jsonl" \
    2>&1 | tee "${OUTPUT_DIR}/${MODEL_ID}.log"

echo ""
python eda/evaluation/extract_results.py "${OUTPUT_DIR}"
