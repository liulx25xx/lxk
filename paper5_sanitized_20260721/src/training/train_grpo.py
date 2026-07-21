"""
GRPO (Group Relative Policy Optimization) training with verifiable rewards.

Uses TRL's GRPOTrainer or a custom implementation compatible with veRL.
GRPO generates G rollouts per prompt, scores each with the domain reward,
computes group-relative advantages, and updates the policy via clipped PPO.

Fair comparison settings:
  - LoRA rank 64, alpha 128, all linear layers (same as SFT)
  - ~4000 GRPO steps with G=8 (matched compute with SFT 3 epochs)
  - Verifiable rewards only (no neural reward model)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Add parent dir for reward module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reward.rewards import compute_reward, get_reward_fn


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_prompt_length: int = 512
    max_response_length: int = 2048

    # LoRA (must match SFT for fair comparison)
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # GRPO
    num_generations: int = 8       # G: rollouts per prompt
    num_train_steps: int = 4000    # total GRPO update steps
    per_device_batch_size: int = 2  # prompts per device per step
    gradient_accumulation_steps: int = 4
    temperature: float = 0.7
    top_p: float = 0.95

    # PPO clipping (DAPO-style asymmetric)
    clip_eps_high: float = 0.28
    clip_eps_low: float = 0.18
    kl_coeff: float = 0.01        # KL penalty coefficient

    # Optimization
    learning_rate: float = 1e-6
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Reward
    use_format_reward: bool = True
    format_reward_weight: float = 0.1

    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500

    # DeepSpeed
    deepspeed_config: str | None = None

    @classmethod
    def from_yaml(cls, path: str) -> "GRPOConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# GRPO Training Loop
# ---------------------------------------------------------------------------

def load_prompts(data_path: str) -> list[dict]:
    """Load RLVR-formatted prompts."""
    records = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} prompts from {data_path}")
    return records


def compute_grpo_advantages(rewards: list[float], eps: float = 1e-8) -> list[float]:
    """
    Compute group-relative advantages: A_i = (R_i - mean(R)) / (std(R) + eps).

    This is the key GRPO idea: normalize within each group of G rollouts
    per prompt, so the model learns from relative quality differences.
    """
    if not rewards:
        return []

    mean_r = sum(rewards) / len(rewards)
    var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
    std_r = var_r ** 0.5

    return [(r - mean_r) / (std_r + eps) for r in rewards]


def train_grpo(
    config: GRPOConfig,
    data_path: str,
    output_dir: str,
    eval_data_path: str | None = None,
):
    """
    Main GRPO training function.

    Uses TRL's GRPOTrainer when available, falls back to custom loop.
    """
    logger.info(f"Starting GRPO training")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Data: {data_path}")
    logger.info(f"  Output: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Try to use TRL's GRPOTrainer (preferred)
    try:
        return _train_grpo_trl(config, data_path, output_dir, eval_data_path)
    except ImportError:
        logger.warning("TRL GRPOTrainer not available, using custom implementation")
        return _train_grpo_custom(config, data_path, output_dir, eval_data_path)


def _train_grpo_trl(
    config: GRPOConfig,
    data_path: str,
    output_dir: str,
    eval_data_path: str | None = None,
):
    """GRPO training using TRL library."""
    from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer

    # Load prompts
    records = load_prompts(data_path)

    # Create dataset with prompt column
    dataset = Dataset.from_list([{"prompt": r["prompt"]} for r in records])

    # Build a lookup for gold answers
    gold_lookup = {r["prompt"]: r for r in records}

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

    # Define reward function for TRL
    def reward_function(completions: list[str], prompts: list[str] | None = None, **kwargs) -> list[float]:
        """Reward function compatible with TRL GRPOTrainer."""
        rewards = []
        for i, completion in enumerate(completions):
            prompt = prompts[i] if prompts else ""
            rec = gold_lookup.get(prompt, {})
            gold = rec.get("gold_answer", "")
            domain = rec.get("domain", "math")
            metadata = rec.get("metadata", {})
            r = compute_reward(
                completion, gold, domain,
                metadata=metadata,
                use_format_reward=config.use_format_reward,
                format_weight=config.format_reward_weight,
            )
            rewards.append(r)
        return rewards

    # TRL GRPO config
    trl_config = TRLGRPOConfig(
        output_dir=output_dir,
        max_steps=config.num_train_steps,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        num_generations=config.num_generations,
        temperature=config.temperature,
        max_completion_length=config.max_response_length,
        report_to="wandb",
        run_name=f"grpo_{Path(data_path).parent.parent.name}_{Path(data_path).parent.name}",
    )

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        args=trl_config,
        train_dataset=dataset,
        reward_funcs=reward_function,
        processing_class=tokenizer,
    )

    # Train
    trainer.train()

    # Save
    trainer.save_model(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    logger.info(f"GRPO training complete. Model saved to {output_dir}/final")


def _train_grpo_custom(
    config: GRPOConfig,
    data_path: str,
    output_dir: str,
    eval_data_path: str | None = None,
):
    """
    Custom GRPO training loop (fallback when TRL GRPOTrainer is not available).

    This implements the core GRPO algorithm:
    1. For each prompt, generate G completions
    2. Score each with the reward function
    3. Compute group-relative advantages
    4. Update policy with clipped PPO objective
    """
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from vllm import LLM, SamplingParams

    # Load prompts
    records = load_prompts(data_path)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_model = AutoModelForCausalLM.from_pretrained(
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
    policy_model = get_peft_model(policy_model, lora_config)
    policy_model.print_trainable_parameters()

    if config.gradient_checkpointing:
        policy_model.enable_input_require_grads()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_model.to(device)

    # Reference model (frozen copy for KL penalty)
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        trust_remote_code=True,
    ).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Optimizer
    optimizer = AdamW(
        policy_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_train_steps)

    # vLLM for fast generation (separate process for efficiency)
    # In practice, use the policy model weights periodically synced to vLLM
    generation_params = SamplingParams(
        n=config.num_generations,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_response_length,
    )

    # Training loop
    import wandb
    wandb.init(
        project="rlvr-vs-sft",
        name=f"grpo_custom_{Path(data_path).parent.parent.name}_{Path(data_path).parent.name}",
    )

    global_step = 0
    epoch = 0
    total_batch_size = config.per_device_batch_size * config.gradient_accumulation_steps

    while global_step < config.num_train_steps:
        epoch += 1
        # Shuffle data each epoch
        import random
        random.shuffle(records)

        for batch_start in range(0, len(records), total_batch_size):
            if global_step >= config.num_train_steps:
                break

            batch_records = records[batch_start : batch_start + total_batch_size]
            if not batch_records:
                continue

            # === 1. Generate G completions per prompt ===
            prompts = [r["prompt"] for r in batch_records]

            # Use model.generate() for completions (slower but simpler than vLLM sync)
            all_completions = []
            all_rewards = []
            all_advantages = []

            for rec in batch_records:
                prompt = rec["prompt"]
                input_ids = tokenizer(
                    prompt, return_tensors="pt", truncation=True,
                    max_length=config.max_prompt_length
                ).input_ids.to(device)

                # Generate G completions
                with torch.no_grad():
                    outputs = policy_model.generate(
                        input_ids.expand(config.num_generations, -1),
                        max_new_tokens=config.max_response_length,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                completions = [
                    tokenizer.decode(o[input_ids.shape[1]:], skip_special_tokens=True)
                    for o in outputs
                ]

                # === 2. Score with reward function ===
                rewards = [
                    compute_reward(
                        comp, rec["gold_answer"], rec["domain"],
                        metadata=rec.get("metadata", {}),
                        use_format_reward=config.use_format_reward,
                        format_weight=config.format_reward_weight,
                    )
                    for comp in completions
                ]

                # === 3. Compute group-relative advantages ===
                advantages = compute_grpo_advantages(rewards)

                all_completions.append(completions)
                all_rewards.append(rewards)
                all_advantages.append(advantages)

            # === 4. Policy gradient update ===
            policy_model.train()
            total_loss = 0.0
            n_updates = 0

            for rec_idx, rec in enumerate(batch_records):
                prompt = rec["prompt"]
                prompt_ids = tokenizer(
                    prompt, return_tensors="pt", truncation=True,
                    max_length=config.max_prompt_length
                ).input_ids.to(device)

                for g_idx in range(config.num_generations):
                    advantage = all_advantages[rec_idx][g_idx]
                    if abs(advantage) < 1e-8:
                        continue  # Skip zero-advantage samples

                    completion = all_completions[rec_idx][g_idx]
                    full_text = prompt + completion
                    full_ids = tokenizer(
                        full_text, return_tensors="pt", truncation=True,
                        max_length=config.max_prompt_length + config.max_response_length,
                    ).input_ids.to(device)

                    # Forward pass through policy
                    policy_logits = policy_model(full_ids).logits
                    ref_logits = ref_model(full_ids).logits

                    # Compute log probs for response tokens only
                    response_start = prompt_ids.shape[1]
                    response_ids = full_ids[:, response_start:]
                    policy_logprobs = F.log_softmax(policy_logits[:, response_start - 1:-1, :], dim=-1)
                    ref_logprobs = F.log_softmax(ref_logits[:, response_start - 1:-1, :], dim=-1)

                    # Gather log probs for actual tokens
                    policy_lp = policy_logprobs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)
                    ref_lp = ref_logprobs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)

                    # Ratio and KL
                    ratio = torch.exp(policy_lp - ref_lp)
                    kl = (ratio - 1) - torch.log(ratio)

                    # Clipped surrogate objective (DAPO asymmetric clipping)
                    adv_tensor = torch.full_like(ratio, advantage)
                    surr1 = ratio * adv_tensor
                    clip_high = 1.0 + config.clip_eps_high
                    clip_low = 1.0 - config.clip_eps_low
                    surr2 = torch.clamp(ratio, clip_low, clip_high) * adv_tensor

                    # Loss = -min(surr1, surr2) + kl_coeff * kl
                    loss = -torch.min(surr1, surr2).mean() + config.kl_coeff * kl.mean()
                    loss = loss / config.gradient_accumulation_steps

                    loss.backward()
                    total_loss += loss.item()
                    n_updates += 1

            # Gradient step
            if n_updates > 0:
                torch.nn.utils.clip_grad_norm_(
                    policy_model.parameters(), config.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            # Logging
            if global_step % config.logging_steps == 0:
                avg_reward = sum(
                    sum(r) / len(r) for r in all_rewards
                ) / max(len(all_rewards), 1)
                logger.info(
                    f"Step {global_step}/{config.num_train_steps} | "
                    f"Loss: {total_loss:.4f} | "
                    f"Avg Reward: {avg_reward:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )
                wandb.log({
                    "step": global_step,
                    "loss": total_loss,
                    "avg_reward": avg_reward,
                    "lr": scheduler.get_last_lr()[0],
                })

            # Save checkpoint
            if global_step % config.save_steps == 0:
                ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                policy_model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                logger.info(f"Saved checkpoint to {ckpt_dir}")

    # Save final
    final_dir = os.path.join(output_dir, "final")
    policy_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"GRPO training complete. Model saved to {final_dir}")
    wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GRPO (RLVR) Training")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--eval_data_path", type=str, default=None)

    # CLI overrides
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--num_generations", type=int, default=None)
    parser.add_argument("--num_train_steps", type=int, default=None)
    parser.add_argument("--per_device_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--kl_coeff", type=float, default=None)
    parser.add_argument("--deepspeed_config", type=str, default=None)

    args = parser.parse_args()

    if args.config:
        config = GRPOConfig.from_yaml(args.config)
    else:
        config = GRPOConfig()

    for param in [
        "model_name", "lora_rank", "lora_alpha", "num_generations",
        "num_train_steps", "per_device_batch_size", "gradient_accumulation_steps",
        "learning_rate", "temperature", "kl_coeff", "deepspeed_config",
    ]:
        val = getattr(args, param, None)
        if val is not None:
            setattr(config, param, val)

    train_grpo(
        config=config,
        data_path=args.data_path,
        output_dir=args.output_dir,
        eval_data_path=args.eval_data_path,
    )


if __name__ == "__main__":
    main()
