#!/bin/bash
# ===========================================================================
# Pilot Launch Script: 4 training runs (Math + Medicine × GRPO + SFT)
#
# Launches all runs in separate tmux sessions on the current server.
# GRPO uses 1 GPU each, SFT uses 1 GPU each.
#
# Layout:
#   GPU 0: Math-GRPO
#   GPU 1: (reserved for GRPO vLLM generation overlap)
#   GPU 2: Math-SFT
#   GPU 3: Medicine-GRPO
#   GPU 4: (reserved for GRPO vLLM generation overlap)
#   GPU 5: Medicine-SFT
#
# Usage:
#   bash launch_pilot.sh [SERVER]
#
# Examples:
#   bash launch_pilot.sh              # run locally
#   bash launch_pilot.sh <REDACTED_IP>  # run on specific server
# ===========================================================================
set -euo pipefail

SERVER="${1:-}"

# --- Configuration ---
PROJECT_DIR="/path/to/workspace/project/emnlp/paper5"
PYTHON="/path/to/workspace/env/miniconda3/envs/openrlhf/bin/python"
GRPO_SCRIPT="${PROJECT_DIR}/src/training/train_grpo_trl.py"
SFT_SCRIPT="${PROJECT_DIR}/src/training/train_sft_clean.py"
LOG_DIR="${PROJECT_DIR}/logs"
N_TRAIN=2000
SEED=42

# --- Environment (inline for tmux) ---
ENV_PREFIX="env HF_HOME=/path/to/workspace/cache/huggingface HUGGINGFACE_HUB_CACHE=/path/to/workspace/cache/huggingface/hub WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false"

# --- Ensure directories exist ---
mkdir -p "${LOG_DIR}"
mkdir -p "${PROJECT_DIR}/outputs/math/grpo/seed${SEED}_n${N_TRAIN}"
mkdir -p "${PROJECT_DIR}/outputs/math/sft/seed${SEED}_n${N_TRAIN}"
mkdir -p "${PROJECT_DIR}/outputs/medicine/grpo/seed${SEED}_n${N_TRAIN}"
mkdir -p "${PROJECT_DIR}/outputs/medicine/sft/seed${SEED}_n${N_TRAIN}"

# --- Helper function ---
launch_tmux() {
    local SESSION_NAME="$1"
    local GPU_ID="$2"
    local CMD="$3"
    local LOG_FILE="$4"

    # Kill existing session if any
    tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true

    # Launch in new tmux session
    tmux new-session -d -s "${SESSION_NAME}" \
        "bash -c '${ENV_PREFIX} CUDA_VISIBLE_DEVICES=${GPU_ID} ${CMD} 2>&1 | tee ${LOG_FILE}; echo \"=== DONE ===\"; sleep 86400'"

    echo "  [OK] ${SESSION_NAME} launched on GPU ${GPU_ID} (tmux attach -t ${SESSION_NAME})"
}

# --- Build commands ---
MATH_GRPO_CMD="${PYTHON} ${GRPO_SCRIPT} \
    --domain math \
    --data_path ${PROJECT_DIR}/data/processed/math/train.jsonl \
    --output_dir ${PROJECT_DIR}/outputs/math/grpo/seed${SEED}_n${N_TRAIN} \
    --n_train ${N_TRAIN} --seed ${SEED} \
    --num_generations 8 --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
    --learning_rate 5e-7 --temperature 1.0 --beta 0.001 --lora_rank 64 --lora_alpha 128"

MATH_SFT_CMD="${PYTHON} ${SFT_SCRIPT} \
    --domain math \
    --data_path ${PROJECT_DIR}/data/sft_cot/math/train.jsonl \
    --output_dir ${PROJECT_DIR}/outputs/math/sft/seed${SEED}_n${N_TRAIN} \
    --n_train ${N_TRAIN} --seed ${SEED} \
    --num_epochs 3 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 --max_length 2048 --lora_rank 64 --lora_alpha 128"

MED_GRPO_CMD="${PYTHON} ${GRPO_SCRIPT} \
    --domain medicine \
    --data_path ${PROJECT_DIR}/data/processed/medicine/train.jsonl \
    --output_dir ${PROJECT_DIR}/outputs/medicine/grpo/seed${SEED}_n${N_TRAIN} \
    --n_train ${N_TRAIN} --seed ${SEED} \
    --num_generations 8 --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
    --learning_rate 5e-7 --temperature 1.0 --beta 0.001 --lora_rank 64 --lora_alpha 128"

MED_SFT_CMD="${PYTHON} ${SFT_SCRIPT} \
    --domain medicine \
    --data_path ${PROJECT_DIR}/data/sft_cot/medicine/train.jsonl \
    --output_dir ${PROJECT_DIR}/outputs/medicine/sft/seed${SEED}_n${N_TRAIN} \
    --n_train ${N_TRAIN} --seed ${SEED} \
    --num_epochs 3 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 --max_length 2048 --lora_rank 64 --lora_alpha 128"

# --- Launch ---
echo "=============================================="
echo "PILOT LAUNCH: Paper 5 - RLVR vs SFT"
echo "  N_TRAIN=${N_TRAIN}, SEED=${SEED}"
echo "  Domains: math, medicine"
echo "  Methods: GRPO, SFT"
echo "=============================================="

if [ -n "${SERVER}" ]; then
    echo "Launching on remote server: ${SERVER}"
    echo "(Note: tmux sessions will be on the remote server)"
    # For remote launch, wrap the whole thing in ssh
    ssh "${SERVER}" "bash -s" <<REMOTE_EOF
set -euo pipefail
mkdir -p ${LOG_DIR}
mkdir -p ${PROJECT_DIR}/outputs/math/grpo/seed${SEED}_n${N_TRAIN}
mkdir -p ${PROJECT_DIR}/outputs/math/sft/seed${SEED}_n${N_TRAIN}
mkdir -p ${PROJECT_DIR}/outputs/medicine/grpo/seed${SEED}_n${N_TRAIN}
mkdir -p ${PROJECT_DIR}/outputs/medicine/sft/seed${SEED}_n${N_TRAIN}

# Math GRPO - GPU 0
tmux kill-session -t pilot_math_grpo 2>/dev/null || true
tmux new-session -d -s pilot_math_grpo \
    "bash -c '${ENV_PREFIX} CUDA_VISIBLE_DEVICES=0 ${MATH_GRPO_CMD} 2>&1 | tee ${LOG_DIR}/grpo_math_s${SEED}_n${N_TRAIN}.log; echo === DONE ===; sleep 86400'"
echo "  [OK] pilot_math_grpo on GPU 0"

# Math SFT - GPU 2
tmux kill-session -t pilot_math_sft 2>/dev/null || true
tmux new-session -d -s pilot_math_sft \
    "bash -c '${ENV_PREFIX} CUDA_VISIBLE_DEVICES=2 ${MATH_SFT_CMD} 2>&1 | tee ${LOG_DIR}/sft_math_s${SEED}_n${N_TRAIN}.log; echo === DONE ===; sleep 86400'"
echo "  [OK] pilot_math_sft on GPU 2"

# Medicine GRPO - GPU 3
tmux kill-session -t pilot_med_grpo 2>/dev/null || true
tmux new-session -d -s pilot_med_grpo \
    "bash -c '${ENV_PREFIX} CUDA_VISIBLE_DEVICES=3 ${MED_GRPO_CMD} 2>&1 | tee ${LOG_DIR}/grpo_medicine_s${SEED}_n${N_TRAIN}.log; echo === DONE ===; sleep 86400'"
echo "  [OK] pilot_med_grpo on GPU 3"

# Medicine SFT - GPU 5
tmux kill-session -t pilot_med_sft 2>/dev/null || true
tmux new-session -d -s pilot_med_sft \
    "bash -c '${ENV_PREFIX} CUDA_VISIBLE_DEVICES=5 ${MED_SFT_CMD} 2>&1 | tee ${LOG_DIR}/sft_medicine_s${SEED}_n${N_TRAIN}.log; echo === DONE ===; sleep 86400'"
echo "  [OK] pilot_med_sft on GPU 5"

echo ""
echo "All 4 pilot runs launched. Monitor with:"
echo "  tmux ls"
echo "  tmux attach -t pilot_math_grpo"
echo "  tail -f ${LOG_DIR}/grpo_math_s${SEED}_n${N_TRAIN}.log"
REMOTE_EOF

else
    # Local launch
    launch_tmux "pilot_math_grpo" 0 "${MATH_GRPO_CMD}" "${LOG_DIR}/grpo_math_s${SEED}_n${N_TRAIN}.log"
    launch_tmux "pilot_math_sft"  2 "${MATH_SFT_CMD}"  "${LOG_DIR}/sft_math_s${SEED}_n${N_TRAIN}.log"
    launch_tmux "pilot_med_grpo"  3 "${MED_GRPO_CMD}"  "${LOG_DIR}/grpo_medicine_s${SEED}_n${N_TRAIN}.log"
    launch_tmux "pilot_med_sft"   5 "${MED_SFT_CMD}"   "${LOG_DIR}/sft_medicine_s${SEED}_n${N_TRAIN}.log"

    echo ""
    echo "All 4 pilot runs launched. Monitor with:"
    echo "  tmux ls"
    echo "  tmux attach -t pilot_math_grpo"
    echo "  tail -f ${LOG_DIR}/grpo_math_s${SEED}_n${N_TRAIN}.log"
fi

echo ""
echo "Expected GPU usage:"
echo "  GPU 0: Math-GRPO (~30GB VRAM)"
echo "  GPU 2: Math-SFT  (~20GB VRAM)"
echo "  GPU 3: Medicine-GRPO (~30GB VRAM)"
echo "  GPU 5: Medicine-SFT  (~20GB VRAM)"
