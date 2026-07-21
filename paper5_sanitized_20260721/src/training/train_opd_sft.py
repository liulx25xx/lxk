"""
OPD Phase 2: SFT on filtered on-policy solutions.

Takes filtered_pairs.jsonl from generate_opd_vllm.py and runs LoRA SFT.
Same config as Paper 5 SFT for fair comparison.

Usage:
    python train_opd_sft.py \
        --data_path /path/to/filtered_pairs.jsonl \
        --output_dir /path/to/outputs/opd/math/seed42/ \
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
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ASSISTANT_START_TOKEN = "<|im_start|>assistant\n"
ASSISTANT_END_TOKEN = "<|im_end|>"


class SFTDataCollator:
    """Data collator for SFT with padding."""
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        max_len = min(max(len(f["input_ids"]) for f in features), self.max_length)
        batch_input_ids, batch_attention_mask, batch_labels = [], [], []
        for f in features:
            input_ids = f["input_ids"][:max_len]
            attention_mask = f["attention_mask"][:max_len]
            labels = f["labels"][:max_len]
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


def prepare_sft_dataset(data_path: str, tokenizer, max_length: int = 2048) -> Dataset:
    """Load filtered pairs and tokenize for SFT with response-only masking."""
    sft_pairs = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                sft_pairs.append(json.loads(line))

    logger.info(f"Loaded {len(sft_pairs)} filtered pairs from {data_path}")

    assistant_start_ids = tokenizer.encode(ASSISTANT_START_TOKEN, add_special_tokens=False)
    end_ids = tokenizer.encode(ASSISTANT_END_TOKEN, add_special_tokens=False)

    processed = []
    for pair in sft_pairs:
        conversations = [
            {"role": "user", "content": pair["prompt"]},
            {"role": "assistant", "content": pair["response"]},
        ]
        text = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
        encoded = tokenizer(text, truncation=True, max_length=max_length, padding=False)
        input_ids = encoded["input_ids"]

        # Create labels: -100 for non-assistant tokens
        labels = [-100] * len(input_ids)
        start_len = len(assistant_start_ids)
        i = 0
        while i < len(input_ids) - start_len:
            if input_ids[i:i + start_len] == assistant_start_ids:
                resp_start = i + start_len
                end_len = len(end_ids)
                resp_end = len(input_ids)
                for j in range(resp_start, len(input_ids) - end_len + 1):
                    if input_ids[j:j + end_len] == end_ids:
                        resp_end = j
                        break
                for k in range(resp_start, min(resp_end + end_len, len(input_ids))):
                    labels[k] = input_ids[k]
                i = resp_end + end_len
            else:
                i += 1

        encoded["labels"] = labels
        processed.append(encoded)

    dataset = Dataset.from_dict({
        "input_ids": [p["input_ids"] for p in processed],
        "attention_mask": [p["attention_mask"] for p in processed],
        "labels": [p["labels"] for p in processed],
    })
    return dataset


def main():
    parser = argparse.ArgumentParser(description="OPD SFT Training (Phase 2)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to filtered_pairs.jsonl")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="/path/to/workspace/model/Qwen2.5-7B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=-1, help="Override epochs with max_steps")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = prepare_sft_dataset(args.data_path, tokenizer, max_length=args.max_length)
    logger.info(f"Tokenized dataset: {len(train_dataset)} samples")

    # Load model
    logger.info(f"Loading model for SFT: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()

    data_collator = SFTDataCollator(tokenizer=tokenizer, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs if args.max_steps <= 0 else 100,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    logger.info("Starting OPD SFT training...")
    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"OPD SFT complete. Model saved to {final_dir}")

    # Save config
    config_path = os.path.join(args.output_dir, "train_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "method": "opd",
            "data_path": args.data_path,
            "model_name": args.model_name,
            "seed": args.seed,
            "num_epochs": args.num_epochs,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "n_train_samples": len(train_dataset),
        }, f, indent=2)


if __name__ == "__main__":
    main()
