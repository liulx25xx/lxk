#!/bin/bash
# Launch OOD evaluations across multiple GPUs/servers.
#
# For each domain, evaluates 3 models: base, best SFT, best GRPO
# Total: 5 domains × 3 models = 15 evaluations
#
# Each eval takes ~5-15 min on 1 GPU (depending on dataset size).
# With 15 parallel GPUs, all finish in ~15 min.
#
# Usage: bash scripts/launch_ood_eval.sh

set -e

PROJECT=/path/to/workspace/project/emnlp/paper5
PYTHON=/path/to/workspace/env/miniconda3/envs/openrlhf/bin/python
EVAL_SCRIPT=${PROJECT}/src/eval/eval_ood.py

export HF_HOME=/path/to/workspace/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/path/to/workspace/cache/huggingface/hub
export HF_DATASETS_CACHE=/path/to/workspace/cache/huggingface/datasets
export TRITON_CACHE_DIR=/path/to/workspace/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/path/to/workspace/cache/torch_inductor
export TMPDIR=/path/to/workspace/cache/tmp

# ---------------------------------------------------------------------------
# Best models per domain (from eval_results/trained/trained_summary.json)
# ---------------------------------------------------------------------------

# Math
MATH_SFT="${PROJECT}/outputs/math/sft/seed123_n100/final"
MATH_GRPO="${PROJECT}/outputs/math/grpo/FIXED_seed42_gsm8k_lr2e5/final"

# Science
SCI_SFT="${PROJECT}/outputs/science/sft/seed456_n5000/final"
SCI_GRPO="${PROJECT}/outputs/science/grpo/FIXED_seed456_n2000_lr2e5/final"

# Medicine
MED_SFT="${PROJECT}/outputs/medicine/sft/seed42_n2000/final"
MED_GRPO="${PROJECT}/outputs/medicine/grpo/FIXED_seed123_n2000_lr2e5/final"

# Law
LAW_SFT="${PROJECT}/outputs/law/sft/seed42/final"
LAW_GRPO="${PROJECT}/outputs/law/grpo/FIXED_seed42_lr5e6/final"

# Commonsense
COMM_SFT="${PROJECT}/outputs/commonsense/sft/seed123/final"
COMM_GRPO="${PROJECT}/outputs/commonsense/grpo/FIXED_seed42_n2000_lr2e5/final"

# ---------------------------------------------------------------------------
# Helper: run one eval
# ---------------------------------------------------------------------------
run_eval() {
    local GPU=$1
    local DOMAIN=$2
    local MODEL_PATH=$3
    local MODEL_TAG=$4

    echo "[$(date +%H:%M:%S)] Starting: domain=${DOMAIN} tag=${MODEL_TAG} gpu=${GPU}"
    nohup ${PYTHON} ${EVAL_SCRIPT} \
        --gpu ${GPU} \
        --domain ${DOMAIN} \
        --model_path "${MODEL_PATH}" \
        --model_tag "${MODEL_TAG}" \
        --cleanup_merge \
        > ${PROJECT}/logs/ood_${DOMAIN}_${MODEL_TAG}.log 2>&1 &
    echo "  PID=$! log=logs/ood_${DOMAIN}_${MODEL_TAG}.log"
}

# ---------------------------------------------------------------------------
# Launch all 15 evaluations
# Assign GPUs: each eval gets its own GPU
# Adjust GPU assignments based on available GPUs
# ---------------------------------------------------------------------------

echo "============================================"
echo "OOD Evaluation Launcher"
echo "============================================"
echo ""
echo "This will launch 15 evals (5 domains × 3 models)."
echo "Assign GPU IDs below based on available GPUs."
echo ""

# Default: launch on local machine using GPUs 0-7
# Edit GPU numbers or SSH commands for multi-server setup.

mkdir -p ${PROJECT}/logs

# --- Math (3 evals) ---
run_eval 0 math base base
run_eval 1 math "${MATH_SFT}" sft_best
run_eval 2 math "${MATH_GRPO}" grpo_best

# --- Science (3 evals) ---
run_eval 3 science base base
run_eval 4 science "${SCI_SFT}" sft_best
run_eval 5 science "${SCI_GRPO}" grpo_best

# --- Medicine (3 evals) ---
run_eval 6 medicine base base
run_eval 7 medicine "${MED_SFT}" sft_best
# Need another GPU — reuse GPU 0 after math base finishes, or use separate server
# For now, queue it
run_eval 0 medicine "${MED_GRPO}" grpo_best

# --- Law (3 evals) ---
run_eval 1 law base base
run_eval 2 law "${LAW_SFT}" sft_best
run_eval 3 law "${LAW_GRPO}" grpo_best

# --- Commonsense (3 evals) ---
run_eval 4 commonsense base base
run_eval 5 commonsense "${COMM_SFT}" sft_best
run_eval 6 commonsense "${COMM_GRPO}" grpo_best

echo ""
echo "All 15 evaluations launched."
echo "Monitor with: tail -f ${PROJECT}/logs/ood_*.log"
echo "Check results: ls ${PROJECT}/eval_results/ood/"
echo ""
echo "Note: base model evals share the same model, so they can run"
echo "concurrently on different GPUs. LoRA models need merge first"
echo "(~60s) before vLLM can load them."
