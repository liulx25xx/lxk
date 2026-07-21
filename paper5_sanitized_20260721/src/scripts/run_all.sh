#!/usr/bin/env bash
# ===========================================================================
# Master pipeline: data preparation → training → evaluation → analysis
#
# Usage:
#   bash src/scripts/run_all.sh
#
# Set environment variables to customize:
#   GPUS=8               Number of GPUs (default: 8)
#   MODEL=...            Base model (default: Qwen/Qwen2.5-7B-Instruct)
#   DATA_SIZE=5000       Training data size (default: 5000)
#   OUTPUT_ROOT=outputs  Root output directory
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC_DIR="$PROJECT_DIR/src"

# Configuration
GPUS="${GPUS:-8}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DATA_SIZE="${DATA_SIZE:-5000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/huggingface}"

export WANDB_PROJECT="rlvr-vs-sft"
export TOKENIZERS_PARALLELISM=false

echo "============================================================"
echo "RLVR vs SFT: Master Pipeline"
echo "============================================================"
echo "  Model:     $MODEL"
echo "  Data size: $DATA_SIZE"
echo "  GPUs:      $GPUS"
echo "  Output:    $OUTPUT_ROOT"
echo "============================================================"

# ---------------------------------------------------------------------------
# Phase 1: Data Preparation
# ---------------------------------------------------------------------------
echo ""
echo "[Phase 1/5] Data Preparation"
echo "------------------------------------------------------------"

# 1a. Download and prepare all datasets
python "$SRC_DIR/data/prepare_datasets.py" \
    --output_dir "$PROJECT_DIR/data/raw" \
    --cache_dir "$CACHE_DIR"

# 1b. Create size and difficulty splits
python "$SRC_DIR/data/create_splits.py" \
    --raw_dir "$PROJECT_DIR/data/raw" \
    --output_dir "$PROJECT_DIR/data/splits" \
    --seed 42

# 1c. Format for SFT
python "$SRC_DIR/data/format_sft.py" \
    --input_dir "$PROJECT_DIR/data/splits" \
    --output_dir "$PROJECT_DIR/data/formatted/sft" \
    --sizes $DATA_SIZE

# 1d. Format for GRPO
python "$SRC_DIR/data/format_rlvr.py" \
    --input_dir "$PROJECT_DIR/data/splits" \
    --output_dir "$PROJECT_DIR/data/formatted/rlvr" \
    --sizes $DATA_SIZE

echo "[Phase 1] Done."

# ---------------------------------------------------------------------------
# Phase 2: Training — all 6 domains × 3 methods
# ---------------------------------------------------------------------------
echo ""
echo "[Phase 2/5] Training"
echo "------------------------------------------------------------"
bash "$SCRIPT_DIR/run_domain_comparison.sh"

echo "[Phase 2] Done."

# ---------------------------------------------------------------------------
# Phase 3: Evaluation
# ---------------------------------------------------------------------------
echo ""
echo "[Phase 3/5] Evaluation"
echo "------------------------------------------------------------"

DOMAINS="math science law medicine code commonsense"
METHODS="sft grpo dpo"

for method in $METHODS; do
    for domain in $DOMAINS; do
        MODEL_PATH="$OUTPUT_ROOT/${method}/${domain}/${DATA_SIZE}/final"
        if [ -d "$MODEL_PATH" ]; then
            echo "  Evaluating: $method / $domain / ${DATA_SIZE}"
            python "$SRC_DIR/eval/evaluate.py" \
                --model_path "$MODEL_PATH" \
                --base_model "$MODEL" \
                --test_dir "$PROJECT_DIR/data/raw" \
                --output_dir "$OUTPUT_ROOT/results/${method}/${domain}/${DATA_SIZE}" \
                --domains "$domain" \
                --mode greedy \
                --tensor_parallel_size "$GPUS"
        fi
    done
done

# Evaluate base model (no training)
for domain in $DOMAINS; do
    echo "  Evaluating: base / $domain"
    python "$SRC_DIR/eval/evaluate.py" \
        --model_path "$MODEL" \
        --test_dir "$PROJECT_DIR/data/raw" \
        --output_dir "$OUTPUT_ROOT/results/base/${domain}/${DATA_SIZE}" \
        --domains "$domain" \
        --mode greedy \
        --tensor_parallel_size "$GPUS"
done

echo "[Phase 3] Done."

# ---------------------------------------------------------------------------
# Phase 4: Analysis
# ---------------------------------------------------------------------------
echo ""
echo "[Phase 4/5] Analysis & Visualization"
echo "------------------------------------------------------------"

# Generate figures
python "$SRC_DIR/analysis/plot_frontier.py" \
    --results_dir "$OUTPUT_ROOT/results" \
    --output_dir "$PROJECT_DIR/figures"

# Statistical tests
python "$SRC_DIR/analysis/compute_statistics.py" \
    --results_dir "$OUTPUT_ROOT/results" \
    --output "$OUTPUT_ROOT/results/statistics_grpo_vs_sft.json" \
    --method_a grpo --method_b sft

python "$SRC_DIR/analysis/compute_statistics.py" \
    --results_dir "$OUTPUT_ROOT/results" \
    --output "$OUTPUT_ROOT/results/statistics_grpo_vs_dpo.json" \
    --method_a grpo --method_b dpo

# Generate LaTeX tables
python "$SRC_DIR/analysis/generate_tables.py" \
    --results_dir "$OUTPUT_ROOT/results" \
    --output_dir "$PROJECT_DIR/tables"

echo "[Phase 4] Done."

# ---------------------------------------------------------------------------
# Phase 5: Summary
# ---------------------------------------------------------------------------
echo ""
echo "[Phase 5/5] Summary"
echo "------------------------------------------------------------"
echo "  Figures:    $PROJECT_DIR/figures/"
echo "  Tables:     $PROJECT_DIR/tables/"
echo "  Results:    $OUTPUT_ROOT/results/"
echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "============================================================"
