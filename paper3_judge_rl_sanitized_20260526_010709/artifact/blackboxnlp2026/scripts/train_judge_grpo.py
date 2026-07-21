"""GRPO training for the pairwise judge experiments.

The historical CLI mode names are retained for compatibility. ``acc_consist``
adds a decisiveness proxy (not paired swap consistency), and ``acc_calib`` adds
a Brier-shaped fixed-confidence proxy (not an empirical calibration objective).

Usage:
    # EXP-006: accuracy only
    python train_judge_grpo.py --reward_mode accuracy --output_dir ../results/EXP-006

    # Accuracy + decisiveness proxy (legacy mode name)
    python train_judge_grpo.py --reward_mode acc_consist --output_dir ../results/EXP-007

    # Accuracy + fixed-confidence proxy (legacy mode name)
    python train_judge_grpo.py --reward_mode acc_calib --output_dir ../results/EXP-008

    # Accuracy + both proxies
    python train_judge_grpo.py --reward_mode full --output_dir ../results/EXP-009

    # Short pilot
    python train_judge_grpo.py --reward_mode full --max_steps 50 --output_dir ../results/pilot

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
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

PROJECT_ROOT = Path(__file__).parent.parent


# =============================================================================
# Reward Functions
# =============================================================================

def parse_judge_output(text):
    """Parse judge model output to extract choice and confidence.
    
    Supports two formats:
      - [[A, 0.85]] — choice with explicit confidence
      - [[A]]       — choice only (confidence defaults to 0.8/0.5)
    """
    # Try format with confidence: [[A, 0.85]] or [[A,0.85]]
    match = re.search(r'\[\[(A|B|C),?\s*([\d.]+)\]\]', text)
    if match:
        choice = match.group(1)
        confidence = float(match.group(2))
        confidence = max(0.5, min(1.0, confidence))  # clamp to [0.5, 1.0]
        return {"choice": choice, "confidence": confidence}
    
    # Fallback: [[A]] without confidence
    match = re.search(r'\[\[(A|B|C)\]\]', text)
    if match:
        choice = match.group(1)
        confidence = 0.8 if choice != "C" else 0.5
        return {"choice": choice, "confidence": confidence}
    
    # Unparseable → tie with low confidence
    return {"choice": "C", "confidence": 0.5}


def compute_accuracy_reward(judge_output, gold_label):
    """Accuracy: 1 if judge agrees with gold, 0 otherwise."""
    parsed = parse_judge_output(judge_output)
    return 1.0 if parsed["choice"] == gold_label else 0.0


def compute_consistency_reward(judge_output, swapped_output):
    """Consistency: 1 if position-swap invariant, 0 otherwise."""
    orig = parse_judge_output(judge_output)
    swap = parse_judge_output(swapped_output)
    
    # If original says A, swapped should say B (since positions flipped)
    flip_map = {"A": "B", "B": "A", "C": "C"}
    expected = flip_map[orig["choice"]]
    
    return 1.0 if swap["choice"] == expected else 0.0


def compute_fixed_confidence_proxy(judge_output, gold_label):
    """Brier-shaped proxy using emitted or fallback confidence."""
    parsed = parse_judge_output(judge_output)
    correct = 1.0 if parsed["choice"] == gold_label else 0.0
    brier = (parsed["confidence"] - correct) ** 2
    return 1.0 - brier


# =============================================================================
# Main Training
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct",
                       help="Base model to train")
    parser.add_argument(
        "--reward_mode",
        choices=["accuracy", "acc_consist", "acc_calib", "full"],
        default="full",
        help=("Reward mode. Legacy acc_consist=accuracy+decisiveness; "
              "acc_calib=accuracy+fixed-confidence proxy."),
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Accuracy reward weight")
    parser.add_argument("--beta", type=float, default=0.5, help="Consistency reward weight")
    parser.add_argument("--gamma", type=float, default=0.3, help="Calibration reward weight")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_data", type=str, 
                       default=str(PROJECT_ROOT / "data/train/judge_train.json"))
    parser.add_argument("--swap_data", type=str, 
                       default=str(PROJECT_ROOT / "data/train/judge_swap.json"))
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set random seed for reproducibility
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
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"=== Judge GRPO Training ===")
    print(f"Mode: {args.reward_mode}")
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    print(f"Steps: {args.max_steps}")
    print()

    # Load data
    print("Loading training data...")
    with open(args.train_data) as f:
        train_data = json.load(f)
    with open(args.swap_data) as f:
        swap_data = json.load(f)
    
    # Build dataset for GRPO
    # TRL GRPOTrainer requires a "prompt" column; extra columns are passed to reward_funcs via **kwargs
    prompts = [item["prompt"] for item in train_data]
    gold_labels = [item["gold_label"] for item in train_data]
    
    # Format prompts as chat messages for chat models
    formatted_prompts = [
        [{"role": "user", "content": p}] for p in prompts
    ]
    
    dataset = Dataset.from_dict({"prompt": formatted_prompts, "gold_label": gold_labels})
    print(f"Training samples: {len(dataset)}")

    # LoRA config
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    # GRPO config
    training_config = GRPOConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=3,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        num_generations=args.group_size,
        max_completion_length=1024,
        save_steps=100,
        logging_steps=10,
        report_to="none",
        seed=args.seed,
        bf16=True,
    )

    # Reward function (core of the paper)
    # TRL GRPOTrainer passes: completions (list[list[dict]]), plus dataset columns as **kwargs
    reward_mode = args.reward_mode
    alpha, beta, gamma = args.alpha, args.beta, args.gamma
    
    def reward_function(completions, gold_label, **kwargs):
        """Compute composite reward for a batch of judge completions.
        
        Args:
            completions: list[list[dict]] — each item is [{"role": "assistant", "content": "..."}]
            gold_label: list[str] — gold labels from dataset (passed via **kwargs by TRL)
        Returns:
            list[float] — reward per completion
        """
        rewards = []
        for i, completion in enumerate(completions):
            # Extract text from chat message format
            text = completion[0]["content"] if isinstance(completion, list) else str(completion)
            gold = gold_label[i]
            
            parsed = parse_judge_output(text)
            
            # Accuracy reward
            acc = 1.0 if parsed["choice"] == gold else 0.0
            
            if reward_mode == "accuracy":
                r = alpha * acc
            elif reward_mode == "acc_consist":
                # Decisiveness proxy: this does not evaluate a swapped pair.
                decisiveness = 0.5 if parsed["choice"] != "C" else 0.0
                r = alpha * acc + beta * decisiveness
            elif reward_mode == "acc_calib":
                conf = parsed["confidence"]
                correct = 1.0 if parsed["choice"] == gold else 0.0
                brier = (conf - correct) ** 2
                confidence_proxy = 1.0 - brier
                r = alpha * acc + gamma * confidence_proxy
            else:  # full
                decisiveness = 0.5 if parsed["choice"] != "C" else 0.0
                conf = parsed["confidence"]
                correct = 1.0 if parsed["choice"] == gold else 0.0
                brier = (conf - correct) ** 2
                confidence_proxy = 1.0 - brier
                r = alpha * acc + beta * decisiveness + gamma * confidence_proxy
            
            rewards.append(r)
        
        return rewards

    print("Initializing GRPOTrainer...")
    
    trainer = GRPOTrainer(
        model=args.model_name,
        reward_funcs=[reward_function],
        args=training_config,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    
    print(f"\n{'='*50}")
    print(f"Starting training: {args.reward_mode} mode, {args.max_steps} steps")
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
