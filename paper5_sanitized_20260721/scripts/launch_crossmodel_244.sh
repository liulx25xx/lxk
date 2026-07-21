#!/bin/bash
# Paper 5 cross-model experiments moved from 121.26 to 244
cd /path/to/workspace/project/emnlp/paper5
export HF_HOME=/path/to/workspace/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/path/to/workspace/cache/huggingface/hub
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=<HF_TOKEN_REDACTED>
PY=/path/to/workspace/env/miniconda3/envs/openrlhf/bin/python

mkdir -p logs outputs/science/grpo outputs/medicine/grpo

# GPU 0: Science lr=5e-7 (Qwen2.5-7B, baseline comparison)
CUDA_VISIBLE_DEVICES=0 nohup $PY src/training/train_grpo_trl.py \
  --domain science \
  --data_path data/processed/science/rlvr_train_n2000.jsonl \
  --output_dir outputs/science/grpo/FIXED_seed42_n2000_lr5e7 \
  --n_train 2000 --seed 42 --num_generations 8 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 2 \
  --learning_rate 5e-7 --temperature 1.0 --beta 0.001 \
  --lora_rank 64 --lora_alpha 128 --max_steps 375 \
  > logs/244_sci_lr5e7.log 2>&1 &

# GPU 2: Mistral Medicine lr=2e-5 (cross-model)
CUDA_VISIBLE_DEVICES=2 nohup $PY src/training/train_grpo_trl.py \
  --domain medicine \
  --data_path data/processed/medicine/rlvr_train_n2000.jsonl \
  --output_dir outputs/medicine/grpo/mistral7b_seed42_n2000_lr2e5 \
  --model_name mistralai/Mistral-7B-Instruct-v0.3 \
  --n_train 2000 --seed 42 --num_generations 8 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 2 \
  --learning_rate 2e-5 --temperature 1.0 --beta 0.001 \
  --lora_rank 64 --lora_alpha 128 --max_steps 375 \
  > logs/244_mistral_med_lr2e5.log 2>&1 &

# GPU 3: DeepSeek Medicine lr=2e-5 (cross-model)
CUDA_VISIBLE_DEVICES=3 nohup $PY src/training/train_grpo_trl.py \
  --domain medicine \
  --data_path data/processed/medicine/rlvr_train_n2000.jsonl \
  --output_dir outputs/medicine/grpo/deepseek7b_seed42_n2000_lr2e5 \
  --model_name deepseek-ai/deepseek-llm-7b-chat \
  --n_train 2000 --seed 42 --num_generations 8 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 2 \
  --learning_rate 2e-5 --temperature 1.0 --beta 0.001 \
  --lora_rank 64 --lora_alpha 128 --max_steps 375 \
  > logs/244_deepseek_med_lr2e5.log 2>&1 &

echo "Paper 5 cross-model experiments launched on 244 (GPU 0,2,3)"
