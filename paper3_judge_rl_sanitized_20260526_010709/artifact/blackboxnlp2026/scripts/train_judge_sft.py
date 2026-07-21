"""
Paper 3: SFT Training for Judge Model (Baseline Comparison with GRPO)

Purpose: Train SFT baselines under unbalanced and position-balanced data.

Usage:
    # SFT on unbalanced data (all gold=A) — compare with GRPO unbalanced
    python train_judge_sft.py \
        --train_data ../data/train/judge_train.json \
        --output_dir ../results/SFT_unbalanced

    # SFT on balanced data (50% A / 50% B)
    python train_judge_sft.py \
        --train_data ../data/train/judge_train_balanced.json \
        --output_dir ../results/SFT_balanced

The script respects standard Hugging Face and CUDA environment variables.
"""

import argparse
import json
import os
import sys
import re
import time
from pathlib import Path
from datetime import datetime

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

PROJECT_ROOT = Path(__file__).parent.parent


def construct_sft_response(gold_label):
    """Construct a judge-style response that picks the gold label.
    
    We use a simple template that mimics real judge output format.
    This is the minimal SFT target — the model learns to output [[A]] or [[B]].
    """
    if gold_label == "A":
        return (
            "After carefully comparing both responses, I find that Assistant A "
            "provides a more helpful, relevant, and well-structured answer that "
            "better addresses the user's question. Assistant A demonstrates stronger "
            "reasoning and more appropriate handling of the topic.\n\n"
            "[[A]]"
        )
    elif gold_label == "B":
        return (
            "After carefully comparing both responses, I find that Assistant B "
            "provides a more helpful, relevant, and well-structured answer that "
            "better addresses the user's question. Assistant B demonstrates stronger "
            "reasoning and more appropriate handling of the topic.\n\n"
            "[[B]]"
        )
    else:
        return (
            "After carefully comparing both responses, I find that both assistants "
            "provide roughly equivalent answers. Neither clearly outperforms the other "
            "on the key criteria of helpfulness, relevance, and accuracy.\n\n"
            "[[C]]"
        )


def format_sft_dataset(train_data):
    """Convert judge_train.json to SFT chat format.
    
    Each sample becomes:
        messages = [
            {"role": "user", "content": judge_prompt},
            {"role": "assistant", "content": constructed_response}
        ]
    """
    messages_list = []
    for item in train_data:
        prompt = item["prompt"]
        gold_label = item["gold_label"]
        response = construct_sft_response(gold_label)
        
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        messages_list.append(messages)
    
    return Dataset.from_dict({"messages": messages_list})


def main():
    parser = argparse.ArgumentParser(description="SFT training for judge model baseline")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct",
                       help="Base model to train")
    parser.add_argument("--train_data", type=str,
                       default=str(PROJECT_ROOT / "data/train/judge_train.json"),
                       help="Training data (judge_train.json or judge_train_balanced.json)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for checkpoints and logs")
    parser.add_argument("--lr", type=float, default=5e-5,
                       help="Learning rate (higher than GRPO since SFT is more stable)")
    parser.add_argument("--max_steps", type=int, default=500,
                       help="Max training steps (same as GRPO for fair comparison)")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--lora_rank", type=int, default=64,
                       help="LoRA rank (same as GRPO)")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Max sequence length")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    args = parser.parse_args()

    # Set random seed
    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_dict = vars(args)
    config_dict["start_time"] = datetime.now().isoformat()
    config_dict["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    config_dict["trainer_type"] = "SFT"
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"=== Judge SFT Training ===")
    print(f"Model: {args.model_name}")
    print(f"Data: {args.train_data}")
    print(f"Output: {args.output_dir}")
    print(f"Steps: {args.max_steps}")
    print(f"LR: {args.lr}")
    print(f"Batch: {args.batch_size} x {args.gradient_accumulation_steps} = {args.batch_size * args.gradient_accumulation_steps}")
    print()

    # Load data
    print("Loading training data...")
    with open(args.train_data) as f:
        train_data = json.load(f)
    
    # Report data statistics
    from collections import Counter
    label_dist = Counter(item["gold_label"] for item in train_data)
    print(f"Training samples: {len(train_data)}")
    print(f"Label distribution: {dict(label_dist)}")
    print()

    # Format dataset for SFT
    dataset = format_sft_dataset(train_data)
    print(f"Dataset ready: {len(dataset)} samples")

    # LoRA config — SAME as GRPO for fair comparison
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,  # alpha = 2*r, same as GRPO
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    # SFT config
    training_config = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_length=args.max_length,
        save_steps=100,
        save_total_limit=3,
        logging_steps=10,
        report_to="none",
        seed=args.seed,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
    )

    print("Initializing SFTTrainer...")
    
    trainer = SFTTrainer(
        model=args.model_name,
        args=training_config,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    
    print(f"\n{'='*50}")
    print(f"Starting SFT training: {args.max_steps} steps")
    print(f"Data: {args.train_data}")
    print(f"Output: {output_dir}")
    print(f"{'='*50}\n")
    
    trainer.train()
    
    # Save final model
    final_model_dir = str(output_dir / "final_model")
    trainer.save_model(final_model_dir)
    print(f"\nTraining complete. Model saved to {final_model_dir}")
    
    # Save final config with end time
    config_dict["end_time"] = datetime.now().isoformat()
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)


if __name__ == "__main__":
    main()
