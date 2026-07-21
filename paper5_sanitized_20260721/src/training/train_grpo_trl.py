"""
GRPO training using TRL 1.4.0 GRPOTrainer.

Clean implementation for Paper 5 "When Does RLVR Beat SFT?" experiments.
Uses verifiable rewards (exact-match) for math/MCQ/code/binary domains.

Usage:
    python train_grpo_trl.py \
        --domain math \
        --data_path /path/to/data/processed/math/train.jsonl \
        --output_dir /path/to/outputs/math/grpo/seed42_n2000/ \
        --n_train 2000 \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# Add parent dir for reward module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reward.rewards import compute_reward, get_reward_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Domain -> max_completion_length
DOMAIN_MAX_COMPLETION = {
    "math": 2048,
    "code": 2048,
    "science": 1024,
    "medicine": 1024,
    "law": 1024,
    "commonsense": 1024,
}


def load_rlvr_dataset(data_path: str, n_train: int | None = None) -> tuple[Dataset, list[dict]]:
    """
    Load RLVR-formatted prompts and create a TRL-compatible dataset.

    TRL GRPOTrainer expects dataset with "prompt" column containing
    list of message dicts (chat format).

    Returns:
        dataset: HuggingFace Dataset with "prompt" column
        records: raw records for gold answer lookup
    """
    records = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if n_train and n_train < len(records):
        records = records[:n_train]

    logger.info(f"Loaded {len(records)} prompts from {data_path}")

    # Convert to TRL format: prompt is a list of message dicts
    dataset_items = []
    for rec in records:
        # The prompt field may be raw text or already formatted
        question = rec.get("prompt") or rec.get("question", "")
        if isinstance(question, list):
            # Already in message format
            dataset_items.append({"prompt": question})
        elif question.startswith("<|im_start|>"):
            # Already has chat template applied — strip it to get raw user content
            # Format: <|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n
            import re
            match = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", question, re.DOTALL)
            if match:
                content = match.group(1).strip()
            else:
                content = question  # fallback
            dataset_items.append({
                "prompt": [{"role": "user", "content": content}]
            })
        else:
            dataset_items.append({
                "prompt": [{"role": "user", "content": question}]
            })

    dataset = Dataset.from_list(dataset_items)
    return dataset, records


def build_reward_function(records: list[dict], domain: str):
    """
    Build a reward function compatible with TRL GRPOTrainer.

    TRL 1.4.0 GRPOTrainer calls reward_funcs with:
        reward_fn(completions, **kwargs)
    where completions is a list of strings (generated text).

    We need to map completions back to their gold answers using the prompts.
    """
    # Build lookup: prompt text -> record
    # KEY FIX: TRL 1.4.0 passes prompts as [{"role":"user","content":"..."}]
    # so we must use the STRIPPED user content as key (without <|im_start|> tokens)
    prompt_to_record = {}
    import re as _re
    for rec in records:
        question = rec.get("prompt") or rec.get("question", "")
        if isinstance(question, list):
            # Already in message format
            key = question[-1]["content"] if question else ""
        elif isinstance(question, str) and "<|im_start|>" in question:
            # Strip chat template tokens to get raw user content
            match = _re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", question, _re.DOTALL)
            if match:
                key = match.group(1).strip()
            else:
                key = question
        else:
            key = question
        prompt_to_record[key] = rec

    reward_fn = get_reward_fn(domain)

    def trl_reward_fn(completions, **kwargs) -> list[float]:
        """Reward function for TRL GRPOTrainer.
        
        TRL 1.4.0 may pass completions as:
        - list[str]: plain text completions
        - list[list[dict]]: chat-format messages (list of message dicts per completion)
        """
        # kwargs may contain 'prompts' from TRL
        prompts = kwargs.get("prompts", [None] * len(completions))

        rewards = []
        for i, completion in enumerate(completions):
            # Extract text from completion (handle both str and chat format)
            if isinstance(completion, str):
                completion_text = completion
            elif isinstance(completion, list):
                # Chat format: extract assistant content
                completion_text = ""
                for msg in completion:
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        completion_text = msg.get("content", "")
                        break
                if not completion_text and completion:
                    # Fallback: use last message content
                    last = completion[-1]
                    completion_text = last.get("content", str(last)) if isinstance(last, dict) else str(last)
            else:
                completion_text = str(completion)

            # Try to find the gold answer
            prompt = prompts[i] if i < len(prompts) else None

            # Look up by prompt content
            rec = None
            if prompt is not None:
                if isinstance(prompt, list):
                    # Chat format prompt
                    for msg in prompt:
                        if isinstance(msg, dict) and msg.get("role") == "user":
                            key = msg.get("content", "")
                            rec = prompt_to_record.get(key)
                            if rec:
                                break
                    if rec is None and prompt:
                        key = prompt[-1]["content"] if isinstance(prompt[-1], dict) else str(prompt[-1])
                        rec = prompt_to_record.get(key)
                elif isinstance(prompt, str):
                    rec = prompt_to_record.get(prompt)

            if rec is None:
                # Fallback: try matching by index (records are in order)
                idx = i % len(records)
                rec = records[idx]

            gold = rec.get("gold_answer") or rec.get("answer", "")
            metadata = rec.get("metadata", {})

            r = reward_fn(completion_text, gold, metadata=metadata)
            rewards.append(float(r))

        return rewards

    return trl_reward_fn


def main():
    parser = argparse.ArgumentParser(description="GRPO Training (TRL 1.4.0)")
    parser.add_argument("--domain", type=str, required=True,
                        choices=["math", "science", "law", "medicine", "code", "commonsense"])
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_train", type=int, default=None,
                        help="Number of training samples to use (None = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max_steps (default: auto from n_train)")
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.001, help="KL coefficient")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)

    args = parser.parse_args()

    # Determine max_completion_length from domain
    max_completion_length = DOMAIN_MAX_COMPLETION.get(args.domain, 1024)

    # Load dataset
    dataset, records = load_rlvr_dataset(args.data_path, n_train=args.n_train)
    n_actual = len(dataset)

    # Compute max_steps: target ~3 epochs
    # effective_batch = per_device_batch_size * gradient_accumulation_steps = 8
    effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
    if args.max_steps is not None:
        max_steps = args.max_steps
    else:
        max_steps = (n_actual * 3) // effective_batch
        max_steps = max(max_steps, 100)  # minimum 100 steps

    logger.info(f"Domain: {args.domain}")
    logger.info(f"Training samples: {n_actual}")
    logger.info(f"Max steps: {max_steps} (~3 epochs with batch={effective_batch})")
    logger.info(f"Max completion length: {max_completion_length}")
    logger.info(f"Seed: {args.seed}")

    os.makedirs(args.output_dir, exist_ok=True)

    # TRL GRPOConfig
    config = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_completion_length=max_completion_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=max_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        beta=args.beta,
        temperature=args.temperature,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        max_grad_norm=1.0,
        dataloader_num_workers=2,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )

    # Build reward function
    reward_fn = build_reward_function(records, args.domain)

    # Initialize trainer
    logger.info("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Training complete. Final model saved to {final_dir}")

    # Save training config for reproducibility
    config_path = os.path.join(args.output_dir, "train_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Config saved to {config_path}")


if __name__ == "__main__":
    main()
