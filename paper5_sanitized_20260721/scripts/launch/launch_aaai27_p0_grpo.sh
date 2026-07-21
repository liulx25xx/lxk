#!/usr/bin/env bash
# Launch one AAAI-27 P0 GRPO run with explicit LR and seed provenance.
#
# Usage:
#   bash scripts/launch/launch_aaai27_p0_grpo.sh DOMAIN LR SEED GPU_ID [N_TRAIN]
#
# Required/optional environment variables:
#   PYTHON_BIN   Python executable for the training environment (default: python)
#   MODEL_NAME   Base model name/path (default: Qwen/Qwen2.5-7B-Instruct)
#   DATA_ROOT    Processed data root (default: <project>/data/processed)
#   OUTPUT_ROOT  Output root (default: <project>/outputs/aaai27_p0)
#   DRY_RUN      Set to 1 to print the command without launching training

set -euo pipefail

DOMAIN="${1:?Usage: launch_aaai27_p0_grpo.sh DOMAIN LR SEED GPU_ID [N_TRAIN]}"
LEARNING_RATE="${2:?Missing LR, expected 1e-6 or 2e-5}"
TRAIN_SEED="${3:?Missing training seed}"
GPU_ID="${4:?Missing GPU ID}"
N_TRAIN="${5:-2000}"

case "${DOMAIN}" in
  math|science|medicine|commonsense) ;;
  *) echo "Unsupported P0 domain: ${DOMAIN}" >&2; exit 2 ;;
esac

case "${LEARNING_RATE}" in
  1e-6|2e-5) ;;
  *) echo "P0 LR must be 1e-6 or 2e-5, got: ${LEARNING_RATE}" >&2; exit 2 ;;
esac

case "${TRAIN_SEED}" in
  42|123|456) ;;
  *) echo "P0 seed must be 42, 123, or 456, got: ${TRAIN_SEED}" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data/processed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/aaai27_p0}"
DATA_PATH="${DATA_ROOT}/${DOMAIN}/train.jsonl"

LR_TAG="${LEARNING_RATE//-/m}"
RUN_ID="${DOMAIN}_grpo_lr${LR_TAG}_seed${TRAIN_SEED}_n${N_TRAIN}"
OUTPUT_DIR="${OUTPUT_ROOT}/${DOMAIN}/grpo/${RUN_ID}"
LOG_DIR="${PROJECT_DIR}/logs/aaai27_p0"
LOG_PATH="${LOG_DIR}/${RUN_ID}.log"

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Training data not found: ${DATA_PATH}" >&2
  echo "Set DATA_ROOT to the restored processed-data directory." >&2
  exit 3
fi

if [[ -e "${OUTPUT_DIR}/final/adapter_config.json" ]]; then
  echo "Refusing to overwrite completed run: ${OUTPUT_DIR}" >&2
  exit 4
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

CMD=(
  "${PYTHON_BIN}" "${PROJECT_DIR}/src/training/train_grpo_trl.py"
  --domain "${DOMAIN}"
  --data_path "${DATA_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --n_train "${N_TRAIN}"
  --seed "${TRAIN_SEED}"
  --model_name "${MODEL_NAME}"
  --num_generations 8
  --per_device_train_batch_size 2
  --gradient_accumulation_steps 4
  --learning_rate "${LEARNING_RATE}"
  --temperature 1.0
  --beta 0.001
  --lora_rank 64
  --lora_alpha 128
)

echo "Run ID: ${RUN_ID}"
echo "CUDA device: ${GPU_ID}"
echo "Data: ${DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM=false

"${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
