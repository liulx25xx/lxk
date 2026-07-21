"""
Fast on-policy generation using vLLM for OPD (On-Policy Distillation).

Generates K responses per prompt using vLLM batch inference, then filters
correct ones using domain-specific reward functions.

Usage:
    python generate_opd_vllm.py \
        --domain math \
        --data_path /path/to/rlvr_train_n2000.jsonl \
        --output_path /path/to/outputs/opd/math/filtered_pairs.jsonl \
        --model_name /path/to/Qwen2.5-7B-Instruct \
        --num_generations 8 \
        --temperature 0.7 \
        --gpu_memory_utilization 0.90
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reward.rewards import get_reward_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_user_content(rec: dict) -> str:
    """Extract user question from record."""
    question = rec.get("prompt") or rec.get("question", "")
    if isinstance(question, list):
        for msg in question:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg["content"]
        return question[-1]["content"] if question else ""
    elif isinstance(question, str) and "<|im_start|>" in question:
        match = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", question, re.DOTALL)
        if match:
            return match.group(1).strip()
    return question


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="/path/to/workspace/model/Qwen2.5-7B-Instruct")
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--n_train", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Load data
    records = []
    with open(args.data_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if args.n_train and args.n_train < len(records):
        records = records[:args.n_train]
    logger.info(f"Loaded {len(records)} prompts from {args.data_path}")

    # Build prompts for vLLM (chat format)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    prompts = []
    for rec in records:
        user_content = extract_user_content(rec)
        messages = [{"role": "user", "content": user_content}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt_text)

    logger.info(f"Built {len(prompts)} prompts. Starting vLLM generation with K={args.num_generations}...")

    # vLLM generation
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_name,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
        dtype="bfloat16",
    )

    sampling_params = SamplingParams(
        n=args.num_generations,
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_tokens,
    )

    outputs = llm.generate(prompts, sampling_params)
    logger.info(f"Generation complete. Filtering correct responses...")

    # Filter correct responses
    reward_fn = get_reward_fn(args.domain)
    sft_pairs = []
    total_correct = 0
    total_generated = 0

    for idx, (rec, output) in enumerate(zip(records, outputs)):
        gold = rec.get("gold_answer") or rec.get("answer", "")
        metadata = rec.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}

        user_content = extract_user_content(rec)
        correct_responses = []

        for completion in output.outputs:
            response_text = completion.text
            total_generated += 1
            r = reward_fn(response_text, gold, metadata=metadata)
            if r > 0.5:
                correct_responses.append(response_text)
                total_correct += 1

        # Keep first correct response
        if correct_responses:
            sft_pairs.append({
                "prompt": user_content,
                "response": correct_responses[0],
                "domain": args.domain,
                "gold_answer": gold,
                "n_correct": len(correct_responses),
                "n_generated": args.num_generations,
            })

    coverage = len(sft_pairs) / max(1, len(records))
    accuracy = total_correct / max(1, total_generated)
    logger.info(f"RESULTS: {len(sft_pairs)}/{len(records)} prompts have correct responses ({100*coverage:.1f}%)")
    logger.info(f"Overall accuracy: {total_correct}/{total_generated} ({100*accuracy:.1f}%)")

    # Save filtered pairs
    with open(args.output_path, "w") as f:
        for pair in sft_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(sft_pairs)} filtered pairs to {args.output_path}")

    # Save stats
    stats_path = args.output_path.replace(".jsonl", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            "domain": args.domain,
            "n_prompts": len(records),
            "n_filtered": len(sft_pairs),
            "coverage": coverage,
            "accuracy": accuracy,
            "total_correct": total_correct,
            "total_generated": total_generated,
            "num_generations": args.num_generations,
            "temperature": args.temperature,
        }, f, indent=2)


if __name__ == "__main__":
    main()
