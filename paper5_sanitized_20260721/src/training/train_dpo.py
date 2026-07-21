"""
DPO (Direct Preference Optimization) training with TRL.

Uses pre-generated preference pairs (chosen/rejected) from format_dpo.py.

Fair comparison settings:
  - LoRA rank 64, alpha 128 (same as SFT and GRPO)
  - Training steps matched to compute budget
  - beta = 0.1 (standard DPO temperature)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class DPOTrainConfig:
    """Configuration for DPO training."""
    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_length: int = 2048
    max_prompt_length: int = 512

    # LoRA (must match SFT/GRPO for fair comparison)
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # DPO
    beta: float = 0.1                 # DPO temperature parameter
    loss_type: str = "sigmoid"        # "sigmoid" or "hinge"

    # Training
    num_train_steps: int = 4000       # matched compute with GRPO
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-7
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500

    # DeepSpeed
    deepspeed_config: str | None = None

    @classmethod
    def from_yaml(cls, path: str) -> "DPOTrainConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def load_dpo_dataset(data_path: str) -> Dataset:
    """Load DPO-formatted JSONL."""
    records = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                records.append({
                    "prompt": rec["prompt"],
                    "chosen": rec["chosen"],
                    "rejected": rec["rejected"],
                })
    logger.info(f"Loaded {len(records)} DPO pairs from {data_path}")
    return Dataset.from_list(records)


def train_dpo(
    config: DPOTrainConfig,
    data_path: str,
    output_dir: str,
    eval_data_path: str | None = None,
):
    """Run DPO training using TRL."""
    from trl import DPOConfig, DPOTrainer

    logger.info(f"Starting DPO training")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Data: {data_path}")
    logger.info(f"  Beta: {config.beta}")

    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True, padding_side="left"
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

    # Reference model (frozen copy)
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        trust_remote_code=True,
    )

    # Apply LoRA to policy model
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

    # Load data
    train_dataset = load_dpo_dataset(data_path)
    eval_dataset = load_dpo_dataset(eval_data_path) if eval_data_path else None

    # DPO training config
    dpo_config = DPOConfig(
        output_dir=output_dir,
        max_steps=config.num_train_steps,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=config.eval_steps if eval_dataset else None,
        beta=config.beta,
        loss_type=config.loss_type,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        report_to="wandb",
        run_name=f"dpo_{Path(data_path).parent.parent.name}_{Path(data_path).parent.name}",
        deepspeed=config.deepspeed_config,
    )

    # Initialize DPO trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Train
    trainer.train()

    # Save final model
    trainer.save_model(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))

    logger.info(f"DPO training complete. Model saved to {output_dir}/final")


def main():
    parser = argparse.ArgumentParser(description="DPO Training")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--eval_data_path", type=str, default=None)

    # CLI overrides
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--num_train_steps", type=int, default=None)
    parser.add_argument("--per_device_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--deepspeed_config", type=str, default=None)

    args = parser.parse_args()

    if args.config:
        config = DPOTrainConfig.from_yaml(args.config)
    else:
        config = DPOTrainConfig()

    for param in [
        "model_name", "lora_rank", "lora_alpha", "beta",
        "num_train_steps", "per_device_batch_size",
        "gradient_accumulation_steps", "learning_rate", "deepspeed_config",
    ]:
        val = getattr(args, param, None)
        if val is not None:
            setattr(config, param, val)

    train_dpo(
        config=config,
        data_path=args.data_path,
        output_dir=args.output_dir,
        eval_data_path=args.eval_data_path,
    )


if __name__ == "__main__":
    main()
