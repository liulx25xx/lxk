"""
On-Policy Distillation (OPD) / Rejection Sampling Fine-Tuning.

For Paper 5: "When Does RLVR Beat SFT?"
OPD = generate responses from current policy → filter correct ones → SFT on them.
This is the "self-distillation" approach used by Qwen3, DeepSeek-V4, etc.

Pipeline:
    1. Load base model
    2. For each training prompt, generate K responses (using vLLM for speed)
    3. Check each response with domain-specific reward (exact-match verifier)
    4. Keep only prompts where at least 1 response is correct
    5. SFT on the (prompt, correct_response) pairs

Usage:
    python train_opd.py \
        --domain medicine \
        --data_path /path/to/data/processed/medicine/rlvr_train_n2000.jsonl \
        --output_dir /path/to/outputs/medicine/opd/seed42_n2000/ \
        --n_train 2000 \
        --seed 42 \
        --num_generations 8
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

# Add parent dir for reward module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reward.rewards import get_reward_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ASSISTANT_START_TOKEN = "<|im_start|>assistant\n"
ASSISTANT_END_TOKEN = "<|im_end|>"


def load_prompts(data_path: str, n_train: int | None = None) -> list[dict]:
    """Load RLVR-formatted prompts."""
    records = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if n_train and n_train < len(records):
        records = records[:n_train]
    logger.info(f"Loaded {len(records)} prompts from {data_path}")
    return records


def extract_user_content(rec: dict) -> str:
    """Extract user question from record."""
    question = rec.get("prompt") or rec.get("question", "")
    if isinstance(question, list):
        for msg in question:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg["content"]
        return question[-1]["content"] if question else ""
    elif question.startswith("<|im_start|>"):
        import re
        match = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", question, re.DOTALL)
        if match:
            return match.group(1).strip()
    return question


def generate_and_filter(
    model_name: str,
    records: list[dict],
    domain: str,
    num_generations: int = 8,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    seed: int = 42,
) -> list[dict]:
    """
    Generate K responses per prompt, filter to keep correct ones.
    Returns list of {"prompt": str, "response": str} for SFT.
    
    Uses HuggingFace generate (not vLLM) for simplicity on single GPU.
    """
    reward_fn = get_reward_fn(domain)

    logger.info(f"Loading model {model_name} for generation...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model.eval()

    sft_pairs = []
    total_correct = 0
    total_generated = 0

    for idx, rec in enumerate(records):
        user_content = extract_user_content(rec)
        gold = rec.get("gold_answer") or rec.get("answer", "")
        metadata = rec.get("metadata", {})

        # Build prompt
        messages = [{"role": "user", "content": user_content}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        # Generate K responses
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                num_return_sequences=num_generations,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode and check each response
        prompt_len = inputs["input_ids"].shape[1]
        correct_responses = []

        for gen in outputs:
            response_ids = gen[prompt_len:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

            r = reward_fn(response_text, gold, metadata=metadata)
            total_generated += 1
            if r > 0.5:
                correct_responses.append(response_text)
                total_correct += 1

        # Keep the first correct response (if any)
        if correct_responses:
            sft_pairs.append({
                "prompt": user_content,
                "response": correct_responses[0],
                "domain": domain,
                "n_correct": len(correct_responses),
                "n_generated": num_generations,
            })

        if (idx + 1) % 50 == 0:
            logger.info(
                f"[{idx+1}/{len(records)}] Pairs so far: {len(sft_pairs)}, "
                f"Overall correct rate: {total_correct}/{total_generated} "
                f"({100*total_correct/max(1,total_generated):.1f}%)"
            )

    logger.info(
        f"Generation complete: {len(sft_pairs)}/{len(records)} prompts have correct responses "
        f"({100*len(sft_pairs)/max(1,len(records)):.1f}%)"
    )
    logger.info(f"Total correct: {total_correct}/{total_generated} ({100*total_correct/max(1,total_generated):.1f}%)")

    # Clean up generation model to free VRAM
    del model
    torch.cuda.empty_cache()

    return sft_pairs


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


def prepare_sft_dataset(sft_pairs: list[dict], tokenizer, max_length: int = 2048) -> Dataset:
    """Tokenize filtered pairs for SFT with response-only masking."""
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
    parser = argparse.ArgumentParser(description="OPD (On-Policy Distillation) Training")
    parser.add_argument("--domain", type=str, required=True,
                        choices=["math", "science", "law", "medicine", "code", "commonsense"])
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_train", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num_generations", type=int, default=8,
                        help="K responses per prompt for rejection sampling")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip generation, load existing filtered data")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    filtered_path = os.path.join(args.output_dir, "filtered_pairs.jsonl")

    # Phase 1: Generate and filter
    if args.skip_generation and os.path.exists(filtered_path):
        logger.info(f"Loading existing filtered pairs from {filtered_path}")
        sft_pairs = []
        with open(filtered_path) as f:
            for line in f:
                if line.strip():
                    sft_pairs.append(json.loads(line))
    else:
        records = load_prompts(args.data_path, n_train=args.n_train)
        sft_pairs = generate_and_filter(
            model_name=args.model_name,
            records=records,
            domain=args.domain,
            num_generations=args.num_generations,
            max_new_tokens=1024 if args.domain != "math" else 2048,
            temperature=args.temperature,
            seed=args.seed,
        )
        # Save filtered pairs
        with open(filtered_path, "w") as f:
            for pair in sft_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(sft_pairs)} filtered pairs to {filtered_path}")

    if len(sft_pairs) < 10:
        logger.error(f"Only {len(sft_pairs)} pairs after filtering. Too few for SFT. Aborting.")
        sys.exit(1)

    # Phase 2: SFT on filtered data
    logger.info(f"Phase 2: SFT on {len(sft_pairs)} filtered pairs")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = prepare_sft_dataset(sft_pairs, tokenizer, max_length=args.max_length)
    logger.info(f"Tokenized dataset: {len(train_dataset)} samples")

    # Load model for SFT
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
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()

    data_collator = SFTDataCollator(tokenizer=tokenizer, max_length=args.max_length)

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
    logger.info(f"OPD training complete. Final model saved to {final_dir}")

    # Save config
    config_path = os.path.join(args.output_dir, "train_config.json")
    with open(config_path, "w") as f:
        config_dict = vars(args)
        config_dict["n_filtered_pairs"] = len(sft_pairs)
        config_dict["method"] = "opd"
        json.dump(config_dict, f, indent=2)
    logger.info(f"Config saved to {config_path}")


if __name__ == "__main__":
    main()
