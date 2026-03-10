#!/bin/bash
# ==================================================================
# Stage 2: Domain Transfer — EDA Adaptation to Target Domain
# ==================================================================
# This stage loads the shared experts from Stage 1 (frozen) and only
# trains the private experts using domain-specific data.
#
# This achieves efficient adaptation with ~43% trainable parameters,
# while preserving the general capabilities in the shared experts.
#
# Usage:
#   cd Efficient-Draft-Adaptation
#   bash scripts/train_stage2_transfer.sh
#
# Run multiple domains in parallel by launching this script on
# different GPU groups with different DOMAIN / TARGET_MODEL settings.
# ==================================================================

set -e

# ─── MODIFY THESE ────────────────────────────────────────────
# Domain: math | code | medical (used only for naming)
DOMAIN="math"

# Path to the domain-specific TARGET model (the model you want to speed up)
#   Math   : Qwen2.5-Math-7B
#   Code   : Qwen2.5-Coder-7B-Instruct
#   Medical: Meditron3-Qwen2.5-7B (or your fine-tuned medical model)
TARGET_MODEL="/path/to/Qwen2.5-Math-7B"

# Path to domain-specific extracted features
# (output of eda/data/allocation.py with domain-generated data)
TRAIN_DATA="/path/to/eagle_datas/deepmath_generated_mufp16"

# Checkpoint from Stage 1 (shared experts will be loaded from here)
STAGE1_CHECKPOINT="/path/to/checkpoints/stage1_base/state_19"

# Where to save Stage 2 checkpoint
CHECKPOINT_DIR="/path/to/checkpoints/stage2_${DOMAIN}"

# GPUs to use (comma-separated)
GPUS="0,1,2,3,4,5,6,7"

# Master port (change if running multiple domains simultaneously)
MASTER_PORT=29600
# ─────────────────────────────────────────────────────────────

export PYTHONPATH=$(pwd):$PYTHONPATH
mkdir -p "${CHECKPOINT_DIR}"

echo "============================================================"
echo "Stage 2: EDA Transfer Training  [${DOMAIN}]"
echo "Target model     : ${TARGET_MODEL}"
echo "Train data       : ${TRAIN_DATA}"
echo "Stage1 ckpt      : ${STAGE1_CHECKPOINT}"
echo "Output checkpoint: ${CHECKPOINT_DIR}"
echo "GPUs             : ${GPUS}"
echo "============================================================"

GPU_LIST=$(echo $GPUS | tr ',' '\n' | paste -sd,)

deepspeed --include localhost:${GPU_LIST} --master_port ${MASTER_PORT} \
    eda/train/main_deepspeed_eda.py \
    --basepath "${TARGET_MODEL}" \
    --tmpdir "${TRAIN_DATA}" \
    --cpdir "${CHECKPOINT_DIR}" \
    --deepspeed_config eda/train/ds_config.json \
    --transfer \
    --pretrained_eda "${STAGE1_CHECKPOINT}" \
    --freeze_attention \
    2>&1 | tee "${CHECKPOINT_DIR}/train.log"

echo ""
echo "Stage 2 training complete. Checkpoint saved to: ${CHECKPOINT_DIR}"
