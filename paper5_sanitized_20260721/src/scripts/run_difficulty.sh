#!/usr/bin/env bash
# ===========================================================================
# Difficulty split experiments: train on easy/medium/hard,
# evaluate within and across difficulty levels (OOD generalization).
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
DIFFICULTIES="easy medium hard"
METHODS="sft grpo"

echo "============================================================"
echo "Difficulty Split Experiments"
echo "  Domains:      $DOMAINS"
echo "  Difficulties: $DIFFICULTIES"
echo "  Methods:      $METHODS"
echo "============================================================"

# Step 1: Train on each difficulty split
echo ""
echo "[Step 1] Training on difficulty splits..."

for domain in $DOMAINS; do
    for diff in $DIFFICULTIES; do
        echo ""
        echo "--- $domain / $diff ---"

        # SFT
        SFT_DATA="$PROJECT_DIR/data/formatted/sft/${domain}/difficulty/${diff}/train.jsonl"
        SFT_OUT="$OUTPUT_ROOT/sft/${domain}/difficulty/${diff}"

        if [ -f "$SFT_DATA" ] && [ ! -d "$SFT_OUT/final" ]; then
            N_SAMPLES=$(wc -l < "$SFT_DATA")
            echo "  [SFT] $domain/$diff ($N_SAMPLES samples)"

            torchrun --nproc_per_node=$GPUS \
                "$SRC_DIR/training/train_sft.py" \
                --data_path "$SFT_DATA" \
                --output_dir "$SFT_OUT" \
                --model_name "$MODEL" \
                --num_epochs 5 \
                --per_device_batch_size 4 \
                --gradient_accumulation_steps 4 \
                --learning_rate 2e-5 \
                --deepspeed_config "$DS_CONFIG"
        fi

        # GRPO
        GRPO_DATA="$PROJECT_DIR/data/formatted/rlvr/${domain}/difficulty/${diff}/train.jsonl"
        GRPO_OUT="$OUTPUT_ROOT/grpo/${domain}/difficulty/${diff}"

        if [ -f "$GRPO_DATA" ] && [ ! -d "$GRPO_OUT/final" ]; then
            N_SAMPLES=$(wc -l < "$GRPO_DATA")
            echo "  [GRPO] $domain/$diff ($N_SAMPLES prompts)"

            STEPS=$(( N_SAMPLES * 4000 / 5000 ))
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

# Step 2: Evaluate — both in-domain and cross-difficulty (OOD)
echo ""
echo "[Step 2] Evaluating (in-domain + cross-difficulty OOD)..."

for domain in $DOMAINS; do
    for train_diff in $DIFFICULTIES; do
        for method in $METHODS; do
            MODEL_PATH="$OUTPUT_ROOT/${method}/${domain}/difficulty/${train_diff}/final"
            if [ ! -d "$MODEL_PATH" ]; then
                continue
            fi

            # Evaluate on all difficulty levels
            for eval_diff in $DIFFICULTIES; do
                EVAL_DATA="$PROJECT_DIR/data/raw/${domain}/test.jsonl"
                RESULT_DIR="$OUTPUT_ROOT/results/${method}/${domain}/difficulty/${train_diff}/eval_${eval_diff}"

                echo "  Eval: $method / $domain / train=$train_diff -> eval=$eval_diff"

                # Filter test data by difficulty for targeted eval
                python -c "
import json, sys
with open('$EVAL_DATA') as f:
    for line in f:
        rec = json.loads(line)
        if rec.get('difficulty') == '$eval_diff' or '$eval_diff' == 'all':
            sys.stdout.write(line)
" > "/tmp/eval_${domain}_${eval_diff}.jsonl"

                python "$SRC_DIR/eval/evaluate.py" \
                    --model_path "$MODEL_PATH" \
                    --base_model "$MODEL" \
                    --test_data "/tmp/eval_${domain}_${eval_diff}.jsonl" \
                    --output_path "$RESULT_DIR/results.json" \
                    --mode greedy \
                    --tensor_parallel_size "$GPUS"
            done
        done
    done
done

echo ""
echo "============================================================"
echo "Difficulty experiments complete!"
echo "============================================================"
