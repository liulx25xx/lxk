#!/bin/bash
# Resume interrupted depth evals on 244 (GPU 4-7)
cd /path/to/paper3_judge_rl
export HF_HOME=/path/to/cache/huggingface
export HF_HUB_CACHE=/path/to/cache/huggingface/hub
export TRANSFORMERS_CACHE=/path/to/cache/huggingface/hub
export TRITON_CACHE_DIR=/path/to/cache/triton
export TMPDIR=/path/to/cache/tmp
export HF_TOKEN=[REDACTED_TOKEN]]
PY=/path/to/env/judge_rl/bin/python

# GPU 4: Resume Qwen2.5 step-300 (312/449 done)
CUDA_VISIBLE_DEVICES=4 nohup $PY scripts/eval_judge.py \
  --model_path Qwen/Qwen2.5-7B-Instruct \
  --adapter_path results/EXP-009_full_composite/checkpoints/checkpoint-300 \
  --test_data data/eval/rewardbench_test.json \
  --output_dir results/EXP-009_full_composite/eval_step300 \
  --batch_size 8 --resume > results/EXP-009_full_composite/eval_step300/run2.log 2>&1 &

# GPU 5: Resume Qwen2.5 step-400 (312/449 done)
CUDA_VISIBLE_DEVICES=5 nohup $PY scripts/eval_judge.py \
  --model_path Qwen/Qwen2.5-7B-Instruct \
  --adapter_path results/EXP-009_full_composite/checkpoints/checkpoint-400 \
  --test_data data/eval/rewardbench_test.json \
  --output_dir results/EXP-009_full_composite/eval_step400 \
  --batch_size 8 --resume > results/EXP-009_full_composite/eval_step400/run2.log 2>&1 &

# GPU 6: Resume Mistral step-100 (84/449 done)
CUDA_VISIBLE_DEVICES=6 nohup $PY scripts/eval_judge.py \
  --model_path mistralai/Mistral-7B-Instruct-v0.3 \
  --adapter_path results/GRPO_mistral7b_unbal/checkpoints/checkpoint-100 \
  --test_data data/eval/rewardbench_test.json \
  --output_dir results/GRPO_mistral7b_unbal/eval_step100 \
  --batch_size 4 --resume > results/GRPO_mistral7b_unbal/eval_step100/run2.log 2>&1 &

# GPU 7: Resume Mistral step-200 (88/449 done)
CUDA_VISIBLE_DEVICES=7 nohup $PY scripts/eval_judge.py \
  --model_path mistralai/Mistral-7B-Instruct-v0.3 \
  --adapter_path results/GRPO_mistral7b_unbal/checkpoints/checkpoint-200 \
  --test_data data/eval/rewardbench_test.json \
  --output_dir results/GRPO_mistral7b_unbal/eval_step200 \
  --batch_size 4 --resume > results/GRPO_mistral7b_unbal/eval_step200/run2.log 2>&1 &

echo "4 depth evals resumed on 244 (GPU 4-7)"
