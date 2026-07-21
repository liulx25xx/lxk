#!/bin/bash
# ===========================================================================
# GRPO Training Launch Script (Single-GPU, TRL 1.4.0)
#
# Usage:
#   bash train_grpo.sh DOMAIN SEED N_TRAIN GPU_ID [SERVER]
#
# Examples:
#   bash train_grpo.sh math 42 2000 0
#   bash train_grpo.sh medicine 42 2000 3 <REDACTED_IP>
# ===========================================================================
set -euo pipefail

# --- Arguments ---
DOMAIN="${1:?Usage: train_grpo.sh DOMAIN SEED N_TRAIN GPU_ID [SERVER]}"
SEED="${2:?Missing SEED}"
N_TRAIN="${3:?Missing N_TRAIN}"
GPU_ID="${4:?Missing GPU_ID}"
SERVER="${5:-}"  # Optional: run on remote server via ssh

# --- Paths (all absolute) ---
PROJECT_DIR="/path/to/workspace/project/emnlp/paper5"
PYTHON="/path/to/workspace/env/miniconda3/envs/openrlhf/bin/python"
SCRIPT="${PROJECT_DIR}/src/training/train_grpo_trl.py"
DATA_PATH="${PROJECT_DIR}/data/processed/${DOMAIN}/train.jsonl"
OUTPUT_DIR="${PROJECT_DIR}/outputs/${DOMAIN}/grpo/seed${SEED}_n${N_TRAIN}"
LOG_DIR="${PROJECT_DIR}/logs"
LOG_FILE="${LOG_DIR}/grpo_${DOMAIN}_s${SEED}_n${N_TRAIN}.log"

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
    --num_generations 8 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-7 \
    --temperature 1.0 \
    --beta 0.001 \
    --lora_rank 64 \
    --lora_alpha 128"

# --- Launch ---
echo "=============================================="
echo "GRPO Training: ${DOMAIN} | seed=${SEED} | n=${N_TRAIN} | GPU=${GPU_ID}"
echo "Output: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"
echo "=============================================="

if [ -n "${SERVER}" ]; then
    echo "Running on remote server: ${SERVER}"
    ssh "${SERVER}" "bash -c '${CMD}'" 2>&1 | tee "${LOG_FILE}"
else
    eval "${CMD}" 2>&1 | tee "${LOG_FILE}"
fi

echo "GRPO training finished: ${DOMAIN} seed=${SEED} n=${N_TRAIN}"
