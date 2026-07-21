#!/usr/bin/env bash
# ===========================================================================
# Run all 6 domains × 3 methods (SFT, GRPO, DPO) comparisons.
#
# Each run uses the SAME data, SAME LoRA config, SAME compute budget.
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC_DIR="$PROJECT_DIR/src"
CONFIG_DIR="$SRC_DIR/training/configs"

GPUS="${GPUS:-8}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DATA_SIZE="${DATA_SIZE:-5000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs}"
DS_CONFIG="$CONFIG_DIR/ds_z2.json"

export WANDB_PROJECT="rlvr-vs-sft"
export TOKENIZERS_PARALLELISM=false

DOMAINS="math science law medicine code commonsense"

echo "============================================================"
echo "Domain Comparison: 6 domains x 3 methods"
echo "  Data size: $DATA_SIZE"
echo "============================================================"

for domain in $DOMAINS; do
    echo ""
    echo "============================================================"
    echo "Domain: $domain"
    echo "============================================================"

    # ----- SFT -----
    SFT_DATA="$PROJECT_DIR/data/formatted/sft/${domain}/${DATA_SIZE}/train.jsonl"
    SFT_OUTPUT="$OUTPUT_ROOT/sft/${domain}/${DATA_SIZE}"

    if [ -f "$SFT_DATA" ] && [ ! -d "$SFT_OUTPUT/final" ]; then
        echo "  [SFT] Training $domain (${DATA_SIZE} samples)..."
        torchrun --nproc_per_node=$GPUS \
            "$SRC_DIR/training/train_sft.py" \
            --data_path "$SFT_DATA" \
            --output_dir "$SFT_OUTPUT" \
            --model_name "$MODEL" \
            --lora_rank 64 \
            --lora_alpha 128 \
            --num_epochs 3 \
            --per_device_batch_size 4 \
            --gradient_accumulation_steps 4 \
            --learning_rate 2e-5 \
            --deepspeed_config "$DS_CONFIG" \
            --max_length 2048
        echo "  [SFT] Done: $SFT_OUTPUT"
    else
        echo "  [SFT] Skipping (data missing or already trained)"
    fi

    # ----- GRPO -----
    GRPO_DATA="$PROJECT_DIR/data/formatted/rlvr/${domain}/${DATA_SIZE}/train.jsonl"
    GRPO_OUTPUT="$OUTPUT_ROOT/grpo/${domain}/${DATA_SIZE}"

    if [ -f "$GRPO_DATA" ] && [ ! -d "$GRPO_OUTPUT/final" ]; then
        echo "  [GRPO] Training $domain (${DATA_SIZE} prompts)..."
        torchrun --nproc_per_node=$GPUS \
            "$SRC_DIR/training/train_grpo.py" \
            --data_path "$GRPO_DATA" \
            --output_dir "$GRPO_OUTPUT" \
            --model_name "$MODEL" \
            --lora_rank 64 \
            --lora_alpha 128 \
            --num_generations 8 \
            --num_train_steps 4000 \
            --per_device_batch_size 2 \
            --gradient_accumulation_steps 4 \
            --learning_rate 1e-6 \
            --temperature 0.7 \
            --kl_coeff 0.01 \
            --deepspeed_config "$DS_CONFIG"
        echo "  [GRPO] Done: $GRPO_OUTPUT"
    else
        echo "  [GRPO] Skipping (data missing or already trained)"
    fi

    # ----- DPO -----
    # DPO requires preference pairs — generate them first if not present
    DPO_DATA="$PROJECT_DIR/data/formatted/dpo/${domain}/${DATA_SIZE}/train.jsonl"
    DPO_OUTPUT="$OUTPUT_ROOT/dpo/${domain}/${DATA_SIZE}"

    if [ ! -f "$DPO_DATA" ] && [ -f "$GRPO_DATA" ]; then
        echo "  [DPO] Generating preference pairs for $domain..."
        python "$SRC_DIR/data/format_dpo.py" \
            --input_dir "$PROJECT_DIR/data/formatted/rlvr" \
            --output_dir "$PROJECT_DIR/data/formatted/dpo" \
            --model_name "$MODEL" \
            --domains "$domain" \
            --sizes "$DATA_SIZE" \
            --n_samples 8 \
            --temperature 1.0 \
            --tensor_parallel_size "$GPUS"
    fi

    if [ -f "$DPO_DATA" ] && [ ! -d "$DPO_OUTPUT/final" ]; then
        echo "  [DPO] Training $domain (${DATA_SIZE} pairs)..."
        torchrun --nproc_per_node=$GPUS \
            "$SRC_DIR/training/train_dpo.py" \
            --data_path "$DPO_DATA" \
            --output_dir "$DPO_OUTPUT" \
            --model_name "$MODEL" \
            --lora_rank 64 \
            --lora_alpha 128 \
            --beta 0.1 \
            --num_train_steps 4000 \
            --per_device_batch_size 2 \
            --gradient_accumulation_steps 8 \
            --learning_rate 5e-7 \
            --deepspeed_config "$DS_CONFIG"
        echo "  [DPO] Done: $DPO_OUTPUT"
    else
        echo "  [DPO] Skipping (data missing or already trained)"
    fi
done

echo ""
echo "============================================================"
echo "All domain comparisons complete!"
echo "============================================================"
