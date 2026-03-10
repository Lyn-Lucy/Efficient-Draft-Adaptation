#!/bin/bash
# =============================================================
# Stage 1: Train the Base EDA Draft Model (on ShareGPT data)
# =============================================================
# This stage trains all components: shared experts + private experts + attention
# using the BASE model (e.g. Qwen2.5-7B) and general ShareGPT data.
#
# The resulting checkpoint carries general knowledge in the shared experts,
# which will be reused (frozen) in Stage 2 for domain adaptation.
#
# Usage:
#   cd Efficient-Draft-Adaptation
#   bash scripts/train_stage1_base.sh
# =============================================================

set -e

# ─── MODIFY THESE ────────────────────────────────────────────
# Path to the base LLM (e.g. Qwen2.5-7B or Qwen2.5-7B-Instruct)
BASE_MODEL="/path/to/Qwen2.5-7B"

# Path to extracted feature data (output of eda/data/allocation.py)
# Should be ShareGPT data extracted with the BASE_MODEL
TRAIN_DATA="/path/to/eagle_datas/sharegpt_base_mufp16"

# Where to save the checkpoint
CHECKPOINT_DIR="/path/to/checkpoints/stage1_base"

# GPUs to use (comma-separated)
GPUS="0,1,2,3,4,5,6,7"

# Number of private experts (must match Stage 2)
NUM_PRIVATE_EXPERTS=1
NUM_SHARED_EXPERTS=1
# ─────────────────────────────────────────────────────────────

export PYTHONPATH=$(pwd):$PYTHONPATH
mkdir -p "${CHECKPOINT_DIR}"

echo "============================================================"
echo "Stage 1: Base EDA Training"
echo "Base model : ${BASE_MODEL}"
echo "Train data : ${TRAIN_DATA}"
echo "Checkpoint : ${CHECKPOINT_DIR}"
echo "GPUs       : ${GPUS}"
echo "============================================================"

GPU_LIST=$(echo $GPUS | tr ',' '\n' | paste -sd,)
MASTER_PORT=29500

deepspeed --include localhost:${GPU_LIST} --master_port ${MASTER_PORT} \
    eda/train/main_deepspeed_eda.py \
    --basepath "${BASE_MODEL}" \
    --tmpdir "${TRAIN_DATA}" \
    --cpdir "${CHECKPOINT_DIR}" \
    --deepspeed_config eda/train/ds_config.json \
    2>&1 | tee "${CHECKPOINT_DIR}/train.log"

echo ""
echo "Stage 1 training complete. Checkpoint saved to: ${CHECKPOINT_DIR}"
