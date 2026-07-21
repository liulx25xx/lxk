#!/bin/bash
cd /path/to/paper3_judge_rl
export HF_HOME=/path/to/cache/huggingface
export HF_HUB_CACHE=/path/to/cache/huggingface/hub
export TRANSFORMERS_CACHE=/path/to/cache/huggingface/hub
export TRITON_CACHE_DIR=/path/to/cache/triton
export TMPDIR=/path/to/cache/tmp
export HF_TOKEN=[REDACTED_TOKEN]]
PY=/path/to/env/judge_rl/bin/python

mkdir -p results/baseline_qwen3_8b/eval

# GPU 0: Qwen3-8B baseline (for A-selection rate measurement)
CUDA_VISIBLE_DEVICES=0 nohup $PY scripts/eval_judge.py \
  --model_path Qwen/Qwen3-8B \
  --test_data data/eval/rewardbench_test.json \
  --output_dir results/baseline_qwen3_8b/eval \
  --batch_size 4 --disable_thinking \
  --resume > results/baseline_qwen3_8b/eval/run_121.log 2>&1 &

echo "Qwen3 baseline eval launched on 121.26 GPU 0"
