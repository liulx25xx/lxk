#!/bin/bash
# Eval the length-confounded model
# Run on same server (244) after training completes
# This evaluates on the standard RewardBench test set
# Then we analyze the results for length preference

set -e

export HF_HOME=/path/to/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/path/to/cache/huggingface/hub
export TRITON_CACHE_DIR=/path/to/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/path/to/cache/torch_inductor
export TMPDIR=/path/to/cache/tmp

PROJECT=/path/to/paper3_judge_rl
PYTHON=/path/to/env/judge_rl/bin/python

echo "=== Evaluating EXP-LENGTH_confounded ==="

$PYTHON $PROJECT/scripts/eval_judge.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --adapter_path $PROJECT/results/EXP-LENGTH_confounded/final_model \
    --test_data $PROJECT/data/eval/rewardbench_test.json \
    --output_dir $PROJECT/results/EXP-LENGTH_confounded/eval \
    --batch_size 4

echo "=== Eval complete ==="
echo "=== Running length preference analysis ==="

$PYTHON $PROJECT/scripts/analyze_length_preference.py

echo "=== All done ==="
