#!/bin/bash
# ===========================================================================
# Model Evaluation Script (vLLM-based, greedy decoding)
#
# Usage:
#   bash eval_model.sh MODEL_PATH DOMAIN SPLIT GPU_ID [BASE_MODEL]
#
# Examples:
#   bash eval_model.sh outputs/math/grpo/seed42_n2000/final math test 0
#   bash eval_model.sh Qwen/Qwen2.5-7B-Instruct math test 0  # base model
#   bash eval_model.sh outputs/math/sft/seed42_n2000/final math test 1
# ===========================================================================
set -euo pipefail

# --- Arguments ---
MODEL_PATH="${1:?Usage: eval_model.sh MODEL_PATH DOMAIN SPLIT GPU_ID [BASE_MODEL]}"
DOMAIN="${2:?Missing DOMAIN}"
SPLIT="${3:?Missing SPLIT (test/val)}"
GPU_ID="${4:?Missing GPU_ID}"
BASE_MODEL="${5:-Qwen/Qwen2.5-7B-Instruct}"

# --- Paths (all absolute) ---
PROJECT_DIR="/path/to/workspace/project/emnlp/paper5"
PYTHON="/path/to/workspace/env/miniconda3/envs/openrlhf/bin/python"
SCRIPT="${PROJECT_DIR}/src/eval/evaluate.py"
TEST_DATA="${PROJECT_DIR}/data/processed/${DOMAIN}/${SPLIT}.jsonl"
LOG_DIR="${PROJECT_DIR}/logs"

# Make MODEL_PATH absolute if relative
if [[ "${MODEL_PATH}" != /* ]]; then
    MODEL_PATH="${PROJECT_DIR}/${MODEL_PATH}"
fi

# Determine output path from model path
MODEL_NAME=$(basename "$(dirname "${MODEL_PATH}")")
if [ "${MODEL_NAME}" = "final" ]; then
    MODEL_NAME=$(basename "$(dirname "$(dirname "${MODEL_PATH}")")")
fi
OUTPUT_DIR="${PROJECT_DIR}/eval_results/${DOMAIN}/${MODEL_NAME}"
OUTPUT_PATH="${OUTPUT_DIR}/${SPLIT}_results.json"
LOG_FILE="${LOG_DIR}/eval_${DOMAIN}_${MODEL_NAME}_${SPLIT}.log"

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
    --model_path ${MODEL_PATH} \
    --base_model ${BASE_MODEL} \
    --test_data ${TEST_DATA} \
    --output_path ${OUTPUT_PATH} \
    --mode greedy \
    --max_tokens 2048 \
    --tensor_parallel_size 1"

# --- Launch ---
echo "=============================================="
echo "Evaluation: ${DOMAIN}/${SPLIT}"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_PATH}"
echo "Log: ${LOG_FILE}"
echo "=============================================="

eval "${CMD}" 2>&1 | tee "${LOG_FILE}"

echo "Evaluation finished: ${DOMAIN}/${SPLIT} -> ${OUTPUT_PATH}"
