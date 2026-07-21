#!/bin/bash
# Deep analysis evals on 119.14 — mechanistic depth experiments
cd /path/to/paper3_judge_rl
export HF_HOME=/path/to/cache/huggingface
export HF_HUB_CACHE=/path/to/cache/huggingface/hub
export TRANSFORMERS_CACHE=/path/to/cache/huggingface/hub
export TRITON_CACHE_DIR=/path/to/cache/triton
export TMPDIR=/path/to/cache/tmp
export HF_TOKEN=[REDACTED_TOKEN]]
PY=/path/to/env/judge_rl/bin/python

# GPU 0-1: Mistral training dynamics (checkpoint-100 and checkpoint-200)
mkdir -p results/GRPO_mistral7b_unbal/eval_step100 results/GRPO_mistral7b_unbal/eval_step200
CUDA_VISIBLE_DEVICES=0 nohup $PY scripts/eval_judge.py \
  --model_path mistralai/Mistral-7B-Instruct-v0.3 \
  --adapter_path results/GRPO_mistral7b_unbal/checkpoints/checkpoint-100 \
  --test_data data/eval/rewardbench_test.json \
  --output_dir results/GRPO_mistral7b_unbal/eval_step100 \
  --batch_size 4 > results/GRPO_mistral7b_unbal/eval_step100/run.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup $PY scripts/eval_judge.py \
  --model_path mistralai/Mistral-7B-Instruct-v0.3 \
  --adapter_path results/GRPO_mistral7b_unbal/checkpoints/checkpoint-200 \
  --test_data data/eval/rewardbench_test.json \
  --output_dir results/GRPO_mistral7b_unbal/eval_step200 \
  --batch_size 4 > results/GRPO_mistral7b_unbal/eval_step200/run.log 2>&1 &

# GPU 2: Mistral BASELINE (no adapter) — measure natural A-selection rate
mkdir -p results/baseline_mistral7b/eval
CUDA_VISIBLE_DEVICES=2 nohup $PY scripts/eval_judge.py \
  --model_path mistralai/Mistral-7B-Instruct-v0.3 \
  --test_data data/eval/rewardbench_test.json \
  --output_dir results/baseline_mistral7b/eval \
  --batch_size 4 > results/baseline_mistral7b/eval/run.log 2>&1 &

# GPU 3: Qwen3-8B BASELINE — measure natural A-selection rate  
mkdir -p results/baseline_qwen3_8b/eval
CUDA_VISIBLE_DEVICES=3 nohup $PY scripts/eval_judge.py \
  --model_path Qwen/Qwen3-8B \
  --test_data data/eval/rewardbench_test.json \
  --output_dir results/baseline_qwen3_8b/eval \
  --batch_size 4 --disable_thinking > results/baseline_qwen3_8b/eval/run.log 2>&1 &

# GPU 4-5: Qwen2.5 additional checkpoint evals (step-300, step-400) for finer dynamics
mkdir -p results/EXP-009_full_composite/eval_step300 results/EXP-009_full_composite/eval_step400
CUDA_VISIBLE_DEVICES=4 nohup $PY scripts/eval_judge.py \
  --model_path Qwen/Qwen2.5-7B-Instruct \
  --adapter_path results/EXP-009_full_composite/checkpoints/checkpoint-300 \
  --test_data data/eval/rewardbench_test.json \
  --output_dir results/EXP-009_full_composite/eval_step300 \
  --batch_size 8 > results/EXP-009_full_composite/eval_step300/run.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup $PY scripts/eval_judge.py \
  --model_path Qwen/Qwen2.5-7B-Instruct \
  --adapter_path results/EXP-009_full_composite/checkpoints/checkpoint-400 \
  --test_data data/eval/rewardbench_test.json \
  --output_dir results/EXP-009_full_composite/eval_step400 \
  --batch_size 8 > results/EXP-009_full_composite/eval_step400/run.log 2>&1 &

# GPU 6: Mistral BALANCED eval on UNBALANCED test (robustness check)
# Does balanced training make the model robust to confounded evaluation?
mkdir -p results/GRPO_mistral7b_balanced/eval_unbal_test
CUDA_VISIBLE_DEVICES=6 nohup $PY scripts/eval_judge.py \
  --model_path mistralai/Mistral-7B-Instruct-v0.3 \
  --adapter_path results/GRPO_mistral7b_balanced/final_model \
  --test_data data/eval/rewardbench_test.json \
  --swap_data data/eval/rewardbench_test.json \
  --output_dir results/GRPO_mistral7b_balanced/eval_unbal_test \
  --batch_size 4 > results/GRPO_mistral7b_balanced/eval_unbal_test/run.log 2>&1 &

# GPU 7: Mistral full-composite unbal (from earlier attempt, redo if failed)
mkdir -p results/GRPO_mistral7b_full_unbal/eval
CUDA_VISIBLE_DEVICES=7 nohup $PY scripts/eval_judge.py \
  --model_path mistralai/Mistral-7B-Instruct-v0.3 \
  --adapter_path results/GRPO_mistral7b_full_unbal/final_model \
  --test_data data/eval/rewardbench_test.json \
  --output_dir results/GRPO_mistral7b_full_unbal/eval \
  --batch_size 4 > results/GRPO_mistral7b_full_unbal/eval/run.log 2>&1 &

echo "ALL 8 DEPTH EVALS LAUNCHED on 119.14"
