"""
SFT training with LoRA + DeepSpeed on Qwen2.5-7B-Instruct.

Fair comparison settings:
  - LoRA rank 64, alpha 128, all linear layers
  - 3 epochs (matched compute budget with GRPO ~4000 steps)
  - Cosine learning rate schedule
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_sft_dataset(data_path: str, tokenizer, max_length: int = 2048) -> Dataset:
    """Load SFT-formatted JSONL and tokenize for causal LM training."""
    records = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    logger.info(f"Loaded {len(records)} SFT records from {data_path}")

    def tokenize_conversation(example):
        conversations = example["conversations"]
        # Build text from conversations using Qwen chat template
        text = tokenizer.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=False
        )
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        # For causal LM, labels = input_ids (shifted internally by the model)
        encoded["labels"] = encoded["input_ids"].copy()

        # Mask system + user tokens (only train on assistant output)
        # Find the assistant response start token
        assistant_start = _find_assistant_start(
            encoded["input_ids"], tokenizer
        )
        if assistant_start > 0:
            encoded["labels"][:assistant_start] = [-100] * assistant_start

        return encoded

    dataset = Dataset.from_list(records)
    tokenized = dataset.map(
        tokenize_conversation,
        remove_columns=dataset.column_names,
        num_proc=4,
        desc="Tokenizing SFT data",
    )
    return tokenized


def _find_assistant_start(input_ids: list[int], tokenizer) -> int:
    """Find the position where the assistant's response begins."""
    # For Qwen2.5 chat template, look for the assistant turn marker
    text = tokenizer.decode(input_ids, skip_special_tokens=False)

    # Common markers for assistant turn in different templates
    markers = ["<|im_start|>assistant\n", "<|assistant|>", "### Assistant:"]
    for marker in markers:
        marker_ids = tokenizer.encode(marker, add_special_tokens=False)
        # Search for the LAST occurrence (in case system also matches)
        last_pos = -1
        for i in range(len(input_ids) - len(marker_ids) + 1):
            if input_ids[i : i + len(marker_ids)] == marker_ids:
                last_pos = i + len(marker_ids)
        if last_pos > 0:
            return last_pos

    # Fallback: mask first 30% of tokens
    return len(input_ids) // 3


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class SFTConfig:
    """Configuration for SFT training."""
    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_length: int = 2048

    # LoRA
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training
    num_epochs: int = 3
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    bf16: bool = True
    gradient_checkpointing: bool = True

    # DeepSpeed
    deepspeed_config: str | None = None

    # Logging
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200

    @classmethod
    def from_yaml(cls, path: str) -> "SFTConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def train_sft(
    config: SFTConfig,
    data_path: str,
    output_dir: str,
    eval_data_path: str | None = None,
    resume_from: str | None = None,
):
    """Run SFT training."""
    logger.info(f"Starting SFT training")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Data: {data_path}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  LoRA rank={config.lora_rank}, alpha={config.lora_alpha}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if config.gradient_checkpointing:
        model.enable_input_require_grads()

    # Load data
    train_dataset = load_sft_dataset(data_path, tokenizer, config.max_length)
    eval_dataset = None
    if eval_data_path:
        eval_dataset = load_sft_dataset(eval_data_path, tokenizer, config.max_length)

    # Data collator
    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=config.max_length,
        label_pad_token_id=-100,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=config.eval_steps if eval_dataset else None,
        report_to="wandb",
        run_name=f"sft_{Path(data_path).parent.parent.name}_{Path(data_path).parent.name}",
        deepspeed=config.deepspeed_config,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    if resume_from:
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    # Save final model
    trainer.save_model(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))

    logger.info(f"SFT training complete. Model saved to {output_dir}/final")


def main():
    parser = argparse.ArgumentParser(description="SFT Training")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to SFT-formatted JSONL training data")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--eval_data_path", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)

    # Override config params via CLI
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--per_device_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--deepspeed_config", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=None)

    args = parser.parse_args()

    # Load config
    if args.config:
        config = SFTConfig.from_yaml(args.config)
    else:
        config = SFTConfig()

    # CLI overrides
    for param in [
        "model_name", "lora_rank", "lora_alpha", "num_epochs",
        "per_device_batch_size", "gradient_accumulation_steps",
        "learning_rate", "deepspeed_config", "max_length",
    ]:
        val = getattr(args, param)
        if val is not None:
            setattr(config, param, val)

    train_sft(
        config=config,
        data_path=args.data_path,
        output_dir=args.output_dir,
        eval_data_path=args.eval_data_path,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
