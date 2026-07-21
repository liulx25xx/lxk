#!/bin/bash
# ===========================================================================
# SFT Training Launch Script (Single-GPU, HuggingFace Trainer + LoRA)
#
# Usage:
#   bash train_sft.sh DOMAIN SEED N_TRAIN GPU_ID [SERVER]
#
# Examples:
#   bash train_sft.sh math 42 2000 2
#   bash train_sft.sh medicine 42 2000 5 <REDACTED_IP>
# ===========================================================================
set -euo pipefail

# --- Arguments ---
DOMAIN="${1:?Usage: train_sft.sh DOMAIN SEED N_TRAIN GPU_ID [SERVER]}"
SEED="${2:?Missing SEED}"
N_TRAIN="${3:?Missing N_TRAIN}"
GPU_ID="${4:?Missing GPU_ID}"
SERVER="${5:-}"  # Optional: run on remote server via ssh

# --- Paths (all absolute) ---
PROJECT_DIR="/path/to/workspace/project/emnlp/paper5"
PYTHON="/path/to/workspace/env/miniconda3/envs/openrlhf/bin/python"
SCRIPT="${PROJECT_DIR}/src/training/train_sft_clean.py"
DATA_PATH="${PROJECT_DIR}/data/sft_cot/${DOMAIN}/train.jsonl"
OUTPUT_DIR="${PROJECT_DIR}/outputs/${DOMAIN}/sft/seed${SEED}_n${N_TRAIN}"
LOG_DIR="${PROJECT_DIR}/logs"
LOG_FILE="${LOG_DIR}/sft_${DOMAIN}_s${SEED}_n${N_TRAIN}.log"

# --- Environment ---
ENV_VARS="HF_HOME=/path/to/workspace/cache/huggingface \
HUGGINGFACE_HUB_CACHE=/path/to/workspace/cache/huggingface/hub \
WANDB_MODE=disabled \
TOKENIZERS_PARALLELISM=false \
CUDA_VISIBLE_DEVICES=${GPU_ID}"

# --- Create dirs ---
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# --- Build command ---
CMD="env ${ENV_VARS} ${PYTHON} ${SCRIPT} \
    --domain ${DOMAIN} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --n_train ${N_TRAIN} \
    --seed ${SEED} \
    --num_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --max_length 2048 \
    --lora_rank 64 \
    --lora_alpha 128"

# --- Launch ---
echo "=============================================="
echo "SFT Training: ${DOMAIN} | seed=${SEED} | n=${N_TRAIN} | GPU=${GPU_ID}"
echo "Output: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"
echo "=============================================="

if [ -n "${SERVER}" ]; then
    echo "Running on remote server: ${SERVER}"
    ssh "${SERVER}" "bash -c '${CMD}'" 2>&1 | tee "${LOG_FILE}"
else
    eval "${CMD}" 2>&1 | tee "${LOG_FILE}"
fi

echo "SFT training finished: ${DOMAIN} seed=${SEED} n=${N_TRAIN}"
