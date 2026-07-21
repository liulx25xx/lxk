#!/bin/bash
# Monitor length-confounded training and auto-eval when complete
# Run this on <internal-host>

set -e

export HF_HOME=/path/to/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/path/to/cache/huggingface/hub
export TRITON_CACHE_DIR=/path/to/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/path/to/cache/torch_inductor
export TMPDIR=/path/to/cache/tmp

PROJECT=/path/to/paper3_judge_rl
PYTHON=/path/to/env/judge_rl/bin/python
RESULT_DIR=$PROJECT/results/EXP-LENGTH_confounded

echo "[$(date)] Monitoring training at $RESULT_DIR"

# Wait for training to complete (check for final_model directory)
while true; do
    if [ -d "$RESULT_DIR/final_model" ]; then
        echo "[$(date)] Training complete! final_model found."
        break
    fi
    # Also check if process died (no GPU usage and no final_model)
    if [ -f "$RESULT_DIR/train.log" ]; then
        LAST_LINE=$(tail -1 "$RESULT_DIR/train.log" 2>/dev/null)
        if echo "$LAST_LINE" | grep -q "Training complete"; then
            echo "[$(date)] Training complete (log confirms)."
            break
        fi
    fi
    echo "[$(date)] Still training... waiting 60s"
    sleep 60
done

echo "[$(date)] Starting evaluation..."

# Determine adapter path
if [ -d "$RESULT_DIR/final_model" ]; then
    ADAPTER_PATH="$RESULT_DIR/final_model"
elif [ -d "$RESULT_DIR/checkpoints/checkpoint-300" ]; then
    ADAPTER_PATH="$RESULT_DIR/checkpoints/checkpoint-300"
else
    echo "ERROR: No model found!"
    exit 1
fi

echo "[$(date)] Using adapter: $ADAPTER_PATH"

# Run eval
CUDA_VISIBLE_DEVICES=0 $PYTHON $PROJECT/scripts/eval_judge.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --adapter_path "$ADAPTER_PATH" \
    --test_data $PROJECT/data/eval/rewardbench_test.json \
    --output_dir $RESULT_DIR/eval \
    --batch_size 4

echo "[$(date)] Eval complete!"

# Run length preference analysis (includes the new experiment)
$PYTHON $PROJECT/scripts/analyze_length_preference.py

echo "[$(date)] All done! Results at $RESULT_DIR/eval/"
