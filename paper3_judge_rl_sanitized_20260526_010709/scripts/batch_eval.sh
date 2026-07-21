#!/bin/bash
# Batch eval for all models that need evaluation
# Run on any server with access to NFS
# Usage: bash batch_eval.sh <GPU_ID>

GPU=${1:-0}
export HF_HOME=/path/to/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/path/to/cache/huggingface/hub
export TMPDIR=/path/to/cache/tmp
export CUDA_VISIBLE_DEVICES=$GPU

PYTHON=/path/to/env/judge_rl/bin/python
SCRIPT=/path/to/paper3_judge_rl/scripts/eval_judge.py
RESULTS=/path/to/paper3_judge_rl/results
TEST=/path/to/paper3_judge_rl/data/eval/rewardbench_test.json
BASE=Qwen/Qwen2.5-7B-Instruct

# Find all models that need eval (trained but no metrics.json)
for dir in $RESULTS/*/; do
    name=$(basename "$dir")
    train_log="$dir/train.log"
    eval_metrics="$dir/eval/metrics.json"
    
    # Skip if no training log or not complete
    [ ! -f "$train_log" ] && continue
    grep -q "Training complete" "$train_log" 2>/dev/null || continue
    
    # Skip if already evaluated
    [ -f "$eval_metrics" ] && continue
    
    # Find latest checkpoint
    ckpt_dir="$dir/checkpoints"
    [ ! -d "$ckpt_dir" ] && continue
    ckpt=$(ls -1d "$ckpt_dir"/checkpoint-* 2>/dev/null | sort -V | tail -1)
    [ -z "$ckpt" ] && continue
    
    echo "$(date '+%H:%M:%S') Evaluating: $name (checkpoint: $(basename $ckpt))"
    mkdir -p "$dir/eval"
    $PYTHON $SCRIPT \
        --model_path $BASE \
        --adapter_path "$ckpt" \
        --test_data $TEST \
        --output_dir "$dir/eval" \
        2>&1 | tail -1
    
    if [ -f "$eval_metrics" ]; then
        acc=$($PYTHON -c "import json; d=json.load(open('$eval_metrics')); print(f'Acc={d[\"accuracy\"]:.3f} Con={d[\"consistency\"]:.3f}')")
        echo "  -> $acc"
    else
        echo "  -> FAILED (no metrics.json)"
    fi
done

echo "$(date '+%H:%M:%S') Batch eval complete."
