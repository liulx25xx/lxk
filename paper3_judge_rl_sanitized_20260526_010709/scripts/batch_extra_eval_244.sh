#!/bin/bash
# Extra evals on 244 — Qwen3 label variant + anti-prompt, all <20min
cd /path/to/paper3_judge_rl
export HF_HOME=/path/to/cache/huggingface
export HF_HUB_CACHE=/path/to/cache/huggingface/hub
export TRANSFORMERS_CACHE=/path/to/cache/huggingface/hub
export TRITON_CACHE_DIR=/path/to/cache/triton
export TMPDIR=/path/to/cache/tmp
export HF_TOKEN=[REDACTED_TOKEN]]
PY=/path/to/env/judge_rl/bin/python

# GPU 4: Qwen3 label variant NUMERIC
mkdir -p results/label_variant_numeric_qwen3
CUDA_VISIBLE_DEVICES=4 nohup $PY scripts/eval_judge_label_variant.py \
  --model_path Qwen/Qwen3-8B \
  --adapter_path results/GRPO_qwen3_8b_unbalanced/final_model \
  --test_data data/eval/rewardbench_test.json \
  --output_dir results/label_variant_numeric_qwen3 \
  --variant numeric --batch_size 4 > results/label_variant_numeric_qwen3/run.log 2>&1 &

# GPU 5: Qwen3 label variant LEFTRIGHT
mkdir -p results/label_variant_leftright_qwen3
CUDA_VISIBLE_DEVICES=5 nohup $PY scripts/eval_judge_label_variant.py \
  --model_path Qwen/Qwen3-8B \
  --adapter_path results/GRPO_qwen3_8b_unbalanced/final_model \
  --test_data data/eval/rewardbench_test.json \
  --output_dir results/label_variant_leftright_qwen3 \
  --variant leftright --batch_size 4 > results/label_variant_leftright_qwen3/run.log 2>&1 &

# GPU 6: Qwen3 anti-prompt
mkdir -p results/antiprompt_qwen3
CUDA_VISIBLE_DEVICES=6 nohup $PY scripts/eval_judge_antiprompt.py \
  --model_path Qwen/Qwen3-8B \
  --adapter_path results/GRPO_qwen3_8b_unbalanced/final_model \
  --test_data data/eval/rewardbench_test.json \
  --output_dir results/antiprompt_qwen3 \
  --batch_size 4 --disable_thinking > results/antiprompt_qwen3/run.log 2>&1 &

# GPU 7: Mistral balanced eval with standard swap (verify already-reported 81.7/70.8 is correct)
# Actually better: Qwen3 BALANCED eval with label variant to show balanced model is ALSO label-agnostic
mkdir -p results/label_variant_numeric_qwen3_bal
CUDA_VISIBLE_DEVICES=7 nohup $PY scripts/eval_judge_label_variant.py \
  --model_path Qwen/Qwen3-8B \
  --adapter_path results/GRPO_qwen3_8b_balanced/final_model \
  --test_data data/eval/rewardbench_test.json \
  --output_dir results/label_variant_numeric_qwen3_bal \
  --variant numeric --batch_size 4 > results/label_variant_numeric_qwen3_bal/run.log 2>&1 &

echo "4 extra evals launched on 244 (GPU 4-7), ~15-20min each"
