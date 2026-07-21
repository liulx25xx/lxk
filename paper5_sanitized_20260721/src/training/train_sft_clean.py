"""
SFT training with LoRA using HuggingFace Trainer.

Clean implementation for Paper 5 "When Does RLVR Beat SFT?" experiments.
Implements response-only loss masking (only trains on assistant tokens).

Usage:
    python train_sft_clean.py \
        --domain math \
        --data_path /path/to/data/sft_cot/math/train.jsonl \
        --output_dir /path/to/outputs/math/sft/seed42_n2000/ \
        --n_train 2000 \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Qwen2.5 chat template markers
ASSISTANT_START_TOKEN = "<|im_start|>assistant\n"
ASSISTANT_END_TOKEN = "<|im_end|>"


def load_sft_dataset(
    data_path: str,
    tokenizer,
    n_train: int | None = None,
    max_length: int = 2048,
) -> Dataset:
    """
    Load SFT-formatted JSONL with conversations field.

    Expected format per line:
    {"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

    or:
    {"prompt": "...", "response": "...", "domain": "..."}

    Returns tokenized dataset with labels masked on non-assistant tokens.
    """
    records = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if n_train and n_train < len(records):
        records = records[:n_train]

    logger.info(f"Loaded {len(records)} SFT records from {data_path}")

    # Normalize to conversations format
    normalized = []
    for rec in records:
        if "conversations" in rec:
            normalized.append(rec["conversations"])
        elif "messages" in rec:
            normalized.append(rec["messages"])
        elif "prompt" in rec and "response" in rec:
            normalized.append([
                {"role": "user", "content": rec["prompt"]},
                {"role": "assistant", "content": rec["response"]},
            ])
        else:
            logger.warning(f"Skipping record with unknown format: {list(rec.keys())}")
            continue

    # Tokenize with response-only masking
    # We encode the full conversation but mask labels for non-assistant tokens
    assistant_start_ids = tokenizer.encode(ASSISTANT_START_TOKEN, add_special_tokens=False)

    def tokenize_fn(conversations):
        text = tokenizer.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=False
        )
        encoded = tokenizer(
            text, truncation=True, max_length=max_length, padding=False
        )
        input_ids = encoded["input_ids"]

        # Create labels: -100 for all tokens except assistant response
        labels = [-100] * len(input_ids)

        # Find assistant response spans
        # Look for assistant_start_ids pattern in input_ids
        start_len = len(assistant_start_ids)
        i = 0
        while i < len(input_ids) - start_len:
            if input_ids[i:i + start_len] == assistant_start_ids:
                # Found assistant start, unmask from here until <|im_end|>
                resp_start = i + start_len
                # Find end token
                end_ids = tokenizer.encode(ASSISTANT_END_TOKEN, add_special_tokens=False)
                end_len = len(end_ids)
                resp_end = len(input_ids)  # default to end
                for j in range(resp_start, len(input_ids) - end_len + 1):
                    if input_ids[j:j + end_len] == end_ids:
                        resp_end = j
                        break
                # Unmask assistant tokens (include the end token too)
                for k in range(resp_start, min(resp_end + end_len, len(input_ids))):
                    labels[k] = input_ids[k]
                i = resp_end + end_len
            else:
                i += 1

        encoded["labels"] = labels
        return encoded

    processed = []
    for convs in normalized:
        try:
            item = tokenize_fn(convs)
            processed.append(item)
        except Exception as e:
            logger.debug(f"Skipping record due to tokenization error: {e}")
            continue

    dataset = Dataset.from_dict({
        "input_ids": [p["input_ids"] for p in processed],
        "attention_mask": [p["attention_mask"] for p in processed],
        "labels": [p["labels"] for p in processed],
    })
    return dataset


@dataclass
class SFTDataCollator:
    """Data collator that pads input_ids, attention_mask, and labels."""
    tokenizer: Any
    max_length: int = 2048

    def __call__(self, features: list[dict]) -> dict:
        # Pad to max length in batch
        max_len = min(
            max(len(f["input_ids"]) for f in features),
            self.max_length,
        )

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for f in features:
            input_ids = f["input_ids"][:max_len]
            attention_mask = f["attention_mask"][:max_len]
            labels = f["labels"][:max_len]

            # Pad to max_len
            pad_len = max_len - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len
                labels = labels + [-100] * pad_len

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def main():
    parser = argparse.ArgumentParser(description="SFT Training (Clean)")
    parser.add_argument("--domain", type=str, required=True,
                        choices=["math", "science", "law", "medicine", "code", "commonsense"])
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_train", type=int, default=None,
                        help="Number of training samples (None = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and tokenize dataset
    train_dataset = load_sft_dataset(
        args.data_path, tokenizer, n_train=args.n_train, max_length=args.max_length
    )
    logger.info(f"Tokenized dataset: {len(train_dataset)} samples")

    # Load model
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # LoRA config (same as GRPO for fair comparison)
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
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()

    # Data collator
    data_collator = SFTDataCollator(tokenizer=tokenizer, max_length=args.max_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting SFT training...")
    trainer.train()

    # Save final model
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"SFT training complete. Final model saved to {final_dir}")

    # Save config
    config_path = os.path.join(args.output_dir, "train_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Config saved to {config_path}")


if __name__ == "__main__":
    main()
