#!/usr/bin/env bash
# ===========================================================================
# Data scaling ablation: train with 500 / 2000 / 5000 / 20000 instances
# per domain per method to study how the RLVR benefit changes with data size.
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC_DIR="$PROJECT_DIR/src"
CONFIG_DIR="$SRC_DIR/training/configs"

GPUS="${GPUS:-8}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs}"
DS_CONFIG="$CONFIG_DIR/ds_z2.json"

export WANDB_PROJECT="rlvr-vs-sft"
export TOKENIZERS_PARALLELISM=false

DOMAINS="${DOMAINS:-math science law medicine code commonsense}"
SIZES="500 2000 5000 20000"
METHODS="sft grpo"

echo "============================================================"
echo "Data Scaling Ablation"
echo "  Domains: $DOMAINS"
echo "  Sizes:   $SIZES"
echo "  Methods: $METHODS"
echo "============================================================"

# Step 1: Ensure all formatted data exists
echo ""
echo "[Step 1] Formatting data for all sizes..."
for size in $SIZES; do
    python "$SRC_DIR/data/format_sft.py" \
        --input_dir "$PROJECT_DIR/data/splits" \
        --output_dir "$PROJECT_DIR/data/formatted/sft" \
        --sizes "$size"

    python "$SRC_DIR/data/format_rlvr.py" \
        --input_dir "$PROJECT_DIR/data/splits" \
        --output_dir "$PROJECT_DIR/data/formatted/rlvr" \
        --sizes "$size"
done

# Step 2: Train all combinations
echo ""
echo "[Step 2] Training all combinations..."

for domain in $DOMAINS; do
    for size in $SIZES; do
        echo ""
        echo "--- $domain / $size ---"

        # SFT
        SFT_DATA="$PROJECT_DIR/data/formatted/sft/${domain}/${size}/train.jsonl"
        SFT_OUT="$OUTPUT_ROOT/sft/${domain}/${size}"

        if [ -f "$SFT_DATA" ] && [ ! -d "$SFT_OUT/final" ]; then
            echo "  [SFT] $domain @ $size"

            # Adjust epochs for smaller datasets to match compute
            if [ "$size" -le 500 ]; then
                EPOCHS=10
            elif [ "$size" -le 2000 ]; then
                EPOCHS=5
            else
                EPOCHS=3
            fi

            torchrun --nproc_per_node=$GPUS \
                "$SRC_DIR/training/train_sft.py" \
                --data_path "$SFT_DATA" \
                --output_dir "$SFT_OUT" \
                --model_name "$MODEL" \
                --num_epochs "$EPOCHS" \
                --per_device_batch_size 4 \
                --gradient_accumulation_steps 4 \
                --learning_rate 2e-5 \
                --deepspeed_config "$DS_CONFIG"
        fi

        # GRPO
        GRPO_DATA="$PROJECT_DIR/data/formatted/rlvr/${domain}/${size}/train.jsonl"
        GRPO_OUT="$OUTPUT_ROOT/grpo/${domain}/${size}"

        if [ -f "$GRPO_DATA" ] && [ ! -d "$GRPO_OUT/final" ]; then
            echo "  [GRPO] $domain @ $size"

            # Adjust steps proportionally to data size
            # Base: 5000 samples -> 4000 steps
            STEPS=$(( size * 4000 / 5000 ))
            [ "$STEPS" -lt 500 ] && STEPS=500

            torchrun --nproc_per_node=$GPUS \
                "$SRC_DIR/training/train_grpo.py" \
                --data_path "$GRPO_DATA" \
                --output_dir "$GRPO_OUT" \
                --model_name "$MODEL" \
                --num_train_steps "$STEPS" \
                --num_generations 8 \
                --per_device_batch_size 2 \
                --gradient_accumulation_steps 4 \
                --learning_rate 1e-6 \
                --deepspeed_config "$DS_CONFIG"
        fi
    done
done

# Step 3: Evaluate all
echo ""
echo "[Step 3] Evaluating all checkpoints..."

for domain in $DOMAINS; do
    for size in $SIZES; do
        for method in $METHODS; do
            MODEL_PATH="$OUTPUT_ROOT/${method}/${domain}/${size}/final"
            if [ -d "$MODEL_PATH" ]; then
                echo "  Eval: $method / $domain / $size"
                python "$SRC_DIR/eval/evaluate.py" \
                    --model_path "$MODEL_PATH" \
                    --base_model "$MODEL" \
                    --test_dir "$PROJECT_DIR/data/raw" \
                    --output_dir "$OUTPUT_ROOT/results/${method}/${domain}/${size}" \
                    --domains "$domain" \
                    --mode greedy \
                    --tensor_parallel_size "$GPUS"
            fi
        done
    done
done

echo ""
echo "============================================================"
echo "Data scaling ablation complete!"
echo "============================================================"
