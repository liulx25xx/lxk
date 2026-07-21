"""
Paper 3: DPO Training for Judge Model (Algorithm Generalization Experiment)

Purpose: Test whether DPO (offline preference optimization) also exhibits position shortcut
amplification when trained on unbalanced data (all gold=A).

Key question: Is the position shortcut specific to GRPO, or a general phenomenon across
different RL/alignment algorithms?

Usage:
    # DPO on unbalanced data (all gold=A)
    python train_judge_dpo.py \
        --train_data ../data/train/judge_train.json \
        --output_dir ../results/DPO_unbalanced

    # DPO on balanced data (50% A / 50% B)
    python train_judge_dpo.py \
        --train_data ../data/train/judge_train_balanced.json \
        --output_dir ../results/DPO_balanced

The script respects standard Hugging Face and CUDA environment variables.
"""

import argparse
import json
import os
import sys
import random
import time
from pathlib import Path
from datetime import datetime

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig

PROJECT_ROOT = Path(__file__).parent.parent


def construct_chosen_response(gold_label):
    """Construct response that picks the correct label (chosen)."""
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


def construct_rejected_response(gold_label):
    """Construct response that picks the WRONG label (rejected)."""
    wrong_label = "B" if gold_label == "A" else "A"
    if wrong_label == "A":
        return (
            "After carefully comparing both responses, I find that Assistant A "
            "provides a more helpful, relevant, and well-structured answer that "
            "better addresses the user's question. Assistant A demonstrates stronger "
            "reasoning and more appropriate handling of the topic.\n\n"
            "[[A]]"
        )
    else:
        return (
            "After carefully comparing both responses, I find that Assistant B "
            "provides a more helpful, relevant, and well-structured answer that "
            "better addresses the user's question. Assistant B demonstrates stronger "
            "reasoning and more appropriate handling of the topic.\n\n"
            "[[B]]"
        )


def build_dpo_dataset(train_data):
    """Convert judge_train.json to DPO format.
    
    DPO requires: prompt, chosen, rejected (all as chat messages)
    - prompt: the user query (judge task)
    - chosen: assistant response that picks correct label
    - rejected: assistant response that picks wrong label
    """
    prompts = []
    chosens = []
    rejecteds = []
    
    for item in train_data:
        prompt_text = item["prompt"]
        gold_label = item["gold_label"]
        
        # Chat format
        prompt_msgs = [{"role": "user", "content": prompt_text}]
        chosen_msgs = [{"role": "assistant", "content": construct_chosen_response(gold_label)}]
        rejected_msgs = [{"role": "assistant", "content": construct_rejected_response(gold_label)}]
        
        prompts.append(prompt_msgs)
        chosens.append(chosen_msgs)
        rejecteds.append(rejected_msgs)
    
    return Dataset.from_dict({
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds,
    })


def main():
    parser = argparse.ArgumentParser(description="DPO training for judge model")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct",
                       help="Base model to train")
    parser.add_argument("--train_data", type=str,
                       default=str(PROJECT_ROOT / "data/train/judge_train.json"),
                       help="Training data JSON")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--lr", type=float, default=5e-6,
                       help="Learning rate (same as GRPO default for fair comparison)")
    parser.add_argument("--max_steps", type=int, default=500,
                       help="Max training steps")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--lora_rank", type=int, default=64,
                       help="LoRA rank (same as GRPO)")
    parser.add_argument("--beta", type=float, default=0.1,
                       help="DPO beta (KL penalty strength)")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Max sequence length")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    args = parser.parse_args()

    # Set random seed
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
    config_dict["trainer_type"] = "DPO"
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"=== Judge DPO Training ===")
    print(f"Model: {args.model_name}")
    print(f"Data: {args.train_data}")
    print(f"Output: {args.output_dir}")
    print(f"Steps: {args.max_steps}")
    print(f"LR: {args.lr}")
    print(f"Beta (KL penalty): {args.beta}")
    print(f"Batch: {args.batch_size} x {args.gradient_accumulation_steps} = {args.batch_size * args.gradient_accumulation_steps}")
    print()

    # Load data
    print("Loading training data...")
    with open(args.train_data) as f:
        train_data = json.load(f)
    
    from collections import Counter
    label_dist = Counter(item["gold_label"] for item in train_data)
    print(f"Training samples: {len(train_data)}")
    print(f"Label distribution: {dict(label_dist)}")
    print()

    # Build DPO dataset
    dataset = build_dpo_dataset(train_data)
    print(f"DPO dataset ready: {len(dataset)} pairs")

    # LoRA config — SAME as GRPO/SFT for fair comparison
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    # DPO config
    training_config = DPOConfig(
        output_dir=str(output_dir / "checkpoints"),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        beta=args.beta,
        max_length=args.max_length,
        save_steps=100,
        save_total_limit=3,
        logging_steps=10,
        report_to="none",
        seed=args.seed,
        bf16=True,
        gradient_checkpointing=True,
    )

    print("Initializing DPOTrainer...")
    
    trainer = DPOTrainer(
        model=args.model_name,
        args=training_config,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    
    print(f"\n{'='*50}")
    print(f"Starting DPO training: {args.max_steps} steps")
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
