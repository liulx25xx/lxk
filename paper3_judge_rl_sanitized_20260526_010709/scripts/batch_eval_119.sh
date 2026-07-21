#!/bin/bash
# Batch eval on <internal-host> — all pure eval, <1h total
cd /path/to/paper3_judge_rl
export HF_HOME=/path/to/cache/huggingface
export HF_HUB_CACHE=/path/to/cache/huggingface/hub
export TRANSFORMERS_CACHE=/path/to/cache/huggingface/hub
export TRITON_CACHE_DIR=/path/to/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/path/to/cache/torch_inductor
export TMPDIR=/path/to/cache/tmp
export HF_TOKEN=[REDACTED_TOKEN]]
PY=/path/to/env/judge_rl/bin/python

# GPU 0: Probe SHORTCUT model
mkdir -p results/probe_shortcut
CUDA_VISIBLE_DEVICES=0 nohup $PY scripts/probe_hidden_states.py \
  --model_path Qwen/Qwen2.5-7B-Instruct \
  --adapter_path results/EXP-009_full_composite/final_model \
  --test_data data/eval/rewardbench_test.json \
  --output_dir results/probe_shortcut \
  --model_name "Shortcut_GRPO_unbal" --max_samples 200 > results/probe_shortcut/run.log 2>&1 &

# GPU 1: Probe BALANCED model
mkdir -p results/probe_balanced
CUDA_VISIBLE_DEVICES=1 nohup $PY scripts/probe_hidden_states.py \
  --model_path Qwen/Qwen2.5-7B-Instruct \
  --adapter_path results/EXP-009b_full_balanced/final_model \
  --test_data data/eval/rewardbench_test.json \
  --output_dir results/probe_balanced \
  --model_name "Balanced_GRPO" --max_samples 200 > results/probe_balanced/run.log 2>&1 &

# GPU 2: Eval Qwen3-8B unbalanced (new seed, needs eval)
mkdir -p results/GRPO_qwen3_8b_unbalanced/eval
CUDA_VISIBLE_DEVICES=2 nohup $PY scripts/eval_judge.py \
  --model_path Qwen/Qwen3-8B \
  --adapter_path results/GRPO_qwen3_8b_unbalanced/final_model \
  --test_data data/eval/rewardbench_test.json \
  --output_dir results/GRPO_qwen3_8b_unbalanced/eval \
  --batch_size 4 --disable_thinking > results/GRPO_qwen3_8b_unbalanced/eval_run.log 2>&1 &

# GPU 3: Label variant NUMERIC on Mistral unbal
mkdir -p results/label_variant_numeric_mistral
CUDA_VISIBLE_DEVICES=3 nohup $PY scripts/eval_judge_label_variant.py \
  --model_path mistralai/Mistral-7B-Instruct-v0.3 \
  --adapter_path results/GRPO_mistral7b_unbal/final_model \
  --test_data data/eval/rewardbench_test.json \
  --output_dir results/label_variant_numeric_mistral \
  --variant numeric --batch_size 4 > results/label_variant_numeric_mistral/run.log 2>&1 &

# GPU 4: Label variant LEFTRIGHT on Mistral unbal
mkdir -p results/label_variant_leftright_mistral
CUDA_VISIBLE_DEVICES=4 nohup $PY scripts/eval_judge_label_variant.py \
  --model_path mistralai/Mistral-7B-Instruct-v0.3 \
  --adapter_path results/GRPO_mistral7b_unbal/final_model \
  --test_data data/eval/rewardbench_test.json \
  --output_dir results/label_variant_leftright_mistral \
  --variant leftright --batch_size 4 > results/label_variant_leftright_mistral/run.log 2>&1 &

# GPU 5: Anti-prompt on Mistral unbal (cross-model anti-prompt)
mkdir -p results/antiprompt_mistral
CUDA_VISIBLE_DEVICES=5 nohup $PY scripts/eval_judge_antiprompt.py \
  --model_path mistralai/Mistral-7B-Instruct-v0.3 \
  --adapter_path results/GRPO_mistral7b_unbal/final_model \
  --test_data data/eval/rewardbench_test.json \
  --output_dir results/antiprompt_mistral \
  --batch_size 4 > results/antiprompt_mistral/run.log 2>&1 &

# GPU 6: Eval Mistral DECISIVE unbal (another reward variant on Mistral)
mkdir -p results/GRPO_mistral7b_decisive_unbal/eval
CUDA_VISIBLE_DEVICES=6 nohup $PY scripts/eval_judge.py \
  --model_path mistralai/Mistral-7B-Instruct-v0.3 \
  --adapter_path results/GRPO_mistral7b_decisive_unbal/final_model \
  --test_data data/eval/rewardbench_test.json \
  --output_dir results/GRPO_mistral7b_decisive_unbal/eval \
  --batch_size 4 > results/GRPO_mistral7b_decisive_unbal/eval_run.log 2>&1 &

# GPU 7: Eval Mistral CALIB unbal (another reward variant)
mkdir -p results/GRPO_mistral7b_calib_unbal/eval
CUDA_VISIBLE_DEVICES=7 nohup $PY scripts/eval_judge.py \
  --model_path mistralai/Mistral-7B-Instruct-v0.3 \
  --adapter_path results/GRPO_mistral7b_calib_unbal/final_model \
  --test_data data/eval/rewardbench_test.json \
  --output_dir results/GRPO_mistral7b_calib_unbal/eval \
  --batch_size 4 > results/GRPO_mistral7b_calib_unbal/eval_run.log 2>&1 &

echo "ALL 8 LAUNCHED on 119.14"
