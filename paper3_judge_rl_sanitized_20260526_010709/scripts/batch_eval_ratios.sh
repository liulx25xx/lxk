#!/bin/bash
# Eval all ratio models (checkpoint-200) in parallel
cd /path/to/paper3_judge_rl
export HF_HOME=/path/to/cache/huggingface
export HF_HUB_CACHE=/path/to/cache/huggingface/hub
export TRANSFORMERS_CACHE=/path/to/cache/huggingface/hub
export TRITON_CACHE_DIR=/path/to/cache/triton
export TMPDIR=/path/to/cache/tmp
export HF_TOKEN=[REDACTED_TOKEN]]
PY=/path/to/env/judge_rl/bin/python

pkill -u researcher -f train_judge_grpo 2>/dev/null
sleep 2

mkdir -p results/GRPO_ratio60/eval results/GRPO_ratio75/eval results/GRPO_ratio80/eval results/GRPO_ratio90/eval results/GRPO_ratio95/eval results/GRPO_balanced_nodupe/eval

CUDA_VISIBLE_DEVICES=2 nohup $PY scripts/eval_judge.py --model_path Qwen/Qwen2.5-7B-Instruct --adapter_path results/GRPO_ratio60/checkpoints/checkpoint-200 --test_data data/eval/rewardbench_test.json --output_dir results/GRPO_ratio60/eval --batch_size 8 > results/GRPO_ratio60/eval/run.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup $PY scripts/eval_judge.py --model_path Qwen/Qwen2.5-7B-Instruct --adapter_path results/GRPO_ratio75/checkpoints/checkpoint-200 --test_data data/eval/rewardbench_test.json --output_dir results/GRPO_ratio75/eval --batch_size 8 > results/GRPO_ratio75/eval/run.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup $PY scripts/eval_judge.py --model_path Qwen/Qwen2.5-7B-Instruct --adapter_path results/GRPO_ratio80/checkpoints/checkpoint-200 --test_data data/eval/rewardbench_test.json --output_dir results/GRPO_ratio80/eval --batch_size 8 > results/GRPO_ratio80/eval/run.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup $PY scripts/eval_judge.py --model_path Qwen/Qwen2.5-7B-Instruct --adapter_path results/GRPO_ratio90/checkpoints/checkpoint-200 --test_data data/eval/rewardbench_test.json --output_dir results/GRPO_ratio90/eval --batch_size 8 > results/GRPO_ratio90/eval/run.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup $PY scripts/eval_judge.py --model_path Qwen/Qwen2.5-7B-Instruct --adapter_path results/GRPO_ratio95/checkpoints/checkpoint-200 --test_data data/eval/rewardbench_test.json --output_dir results/GRPO_ratio95/eval --batch_size 8 > results/GRPO_ratio95/eval/run.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup $PY scripts/eval_judge.py --model_path Qwen/Qwen2.5-7B-Instruct --adapter_path results/GRPO_balanced_nodupe/checkpoints/checkpoint-200 --test_data data/eval/rewardbench_test.json --output_dir results/GRPO_balanced_nodupe/eval --batch_size 8 > results/GRPO_balanced_nodupe/eval/run.log 2>&1 &

echo "ALL 6 RATIO EVALS LAUNCHED (checkpoint-200, ~15min each)"
