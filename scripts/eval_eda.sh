#!/bin/bash
# ==================================================================
# Evaluate EDA EAGLE draft model (our method)
# ==================================================================
# Measures speculative decoding speedup (Accept Length) on benchmarks.
#
# Usage:
#   cd Efficient-Draft-Adaptation
#   bash scripts/eval_eda.sh
# ==================================================================

set -e

# ─── MODIFY THESE ────────────────────────────────────────────
# Target domain model (same as used in Stage 2 training)
BASE_MODEL="/path/to/Qwen2.5-Math-7B"

# Draft model checkpoint directory (contains state_19/)
EAGLE_DIR="/path/to/checkpoints/stage2_math"

# Epoch to evaluate (default: 19, the last epoch)
EPOCH=19

# Task / benchmark:
#   gsm8k | aime_2024 | svamp | hendrycks_math | math_qa   (Math domain)
#   humaneval | humaneval_plus | apps | bigcodebench | mbpp  (Code domain)
#   medmcqa | medqa_usmle | pubmedqa | mmlu_clinical         (Medical domain)
TASK="gsm8k"

# Temperature (0.0 for greedy, higher for sampling)
TEMPERATURE=0.0

# GPU to use
GPU="0"

# Name for saved result files
MODEL_ID="eda_math_${TASK}"

# Output directory
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
echo "EDA EAGLE Evaluation"
echo "Base model : ${BASE_MODEL}"
echo "Draft ckpt : ${CKPT}"
echo "Task       : ${TASK}"
echo "Temperature: ${TEMPERATURE}"
echo "GPU        : ${GPU}"
echo "============================================================"

CUDA_VISIBLE_DEVICES=${GPU} python -m eda.evaluation.gen_ea_answer_qwen2_eda \
    --base-model-path "${BASE_MODEL}" \
    --ea-model-path "${CKPT}" \
    --model-id "${MODEL_ID}" \
    --bench-name "${TASK}" \
    --temperature "${TEMPERATURE}" \
    --answer-file "${OUTPUT_DIR}/${MODEL_ID}.jsonl" \
    2>&1 | tee "${OUTPUT_DIR}/${MODEL_ID}.log"

echo ""
echo "Evaluation done. Extracting accept length stats..."

python eda/evaluation/extract_results.py "${OUTPUT_DIR}"
