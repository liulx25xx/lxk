"""
Evaluate any checkpoint on any benchmark.

Supports:
  - All 6 domains (math, science, law, medicine, code, commonsense)
  - Multiple evaluation modes: greedy, sampling (pass@k), majority voting
  - Batch inference via vLLM for speed
  - Automatic LoRA adapter merging for inference
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reward.rewards import compute_reward, get_reward_fn, extract_math_answer, extract_mcq_answer
from eval.metrics import (
    compute_accuracy,
    compute_pass_at_k,
    compute_domain_metrics,
    aggregate_results,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates for evaluation (consistent with training)
# ---------------------------------------------------------------------------

EVAL_PROMPTS = {
    "math": (
        "Solve the following math problem step by step. "
        "Show your work and put your final numerical answer after '####'.\n\n"
        "Problem: {question}\n\n"
        "Solution:"
    ),
    "science": (
        "Answer the following science question. Think step by step and "
        "select the correct answer.\n\n"
        "{question}\n\n"
        "Answer:"
    ),
    "law": (
        "Analyze the following legal question carefully. "
        "Provide your reasoning and final answer.\n\n"
        "{question}\n\n"
        "Answer:"
    ),
    "medicine": (
        "You are given a medical question. Reason through the clinical "
        "scenario step by step and select the best answer.\n\n"
        "{question}\n\n"
        "Answer:"
    ),
    "code": (
        "Complete the following Python function. "
        "Think about the approach first, then write the code.\n\n"
        "{question}\n\n"
        "Solution:\n```python"
    ),
    "commonsense": (
        "Answer the following question using common sense reasoning. "
        "Think step by step.\n\n"
        "{question}\n\n"
        "Answer:"
    ),
}


def format_eval_question(rec: dict) -> str:
    """Format a test record into a prompt for evaluation."""
    domain = rec["domain"]
    template = EVAL_PROMPTS.get(domain, EVAL_PROMPTS["commonsense"])

    # Format MCQ choices
    question = rec["question"]
    choices = rec.get("choices")
    if choices:
        question += "\n\nOptions:\n"
        for i, c in enumerate(choices):
            question += f"  {chr(ord('A') + i)}. {c}\n"

    return template.format(question=question.strip())


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_for_eval(
    model_path: str,
    base_model: str | None = None,
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
) -> LLM:
    """
    Load a model for evaluation via vLLM.

    If model_path points to a LoRA adapter, merges it with the base model first.
    """
    adapter_config_path = os.path.join(model_path, "adapter_config.json")

    if os.path.exists(adapter_config_path):
        # LoRA adapter — merge first
        logger.info(f"Detected LoRA adapter at {model_path}, merging with base model...")
        from peft import PeftModel, PeftConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer

        peft_config = PeftConfig.from_pretrained(model_path)
        base_model_name = base_model or peft_config.base_model_name_or_path

        base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
        merged_model = model.merge_and_unload()

        # Save merged model to temp dir
        merged_dir = os.path.join(model_path, "merged")
        os.makedirs(merged_dir, exist_ok=True)
        merged_model.save_pretrained(merged_dir)
        AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True).save_pretrained(merged_dir)

        model_path = merged_dir
        logger.info(f"Merged model saved to {merged_dir}")

    # Load with vLLM
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=max_model_len,
    )
    return llm


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model_path: str,
    test_data_path: str,
    output_path: str,
    base_model: str | None = None,
    mode: str = "greedy",            # "greedy", "sample", "majority"
    n_samples: int = 1,              # for pass@k or majority voting
    temperature: float = 0.0,
    max_tokens: int = 2048,
    tensor_parallel_size: int = 1,
    batch_size: int = 128,
) -> dict:
    """
    Evaluate a model on a test dataset.

    Returns a dict with overall and per-subdomain metrics.
    """
    logger.info(f"Evaluating {model_path}")
    logger.info(f"  Test data: {test_data_path}")
    logger.info(f"  Mode: {mode}, n_samples: {n_samples}")

    # Load test data
    records = []
    with open(test_data_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    logger.info(f"  {len(records)} test records loaded")

    # Load model
    llm = load_model_for_eval(
        model_path, base_model=base_model,
        tensor_parallel_size=tensor_parallel_size,
    )

    # Format prompts
    prompts = [format_eval_question(r) for r in records]

    # Set sampling params
    if mode == "greedy":
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            n=1,
        )
    elif mode in ("sample", "majority"):
        sampling_params = SamplingParams(
            temperature=temperature if temperature > 0 else 0.7,
            top_p=0.95,
            max_tokens=max_tokens,
            n=n_samples,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Generate
    logger.info(f"  Generating responses...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.time() - start_time
    logger.info(f"  Generation took {gen_time:.1f}s ({len(records)/gen_time:.1f} records/s)")

    # Score
    results = []
    for rec, output in zip(records, outputs):
        domain = rec["domain"]
        gold = rec["answer"]
        metadata = rec.get("metadata", {})

        if mode == "greedy":
            response_text = output.outputs[0].text
            reward = compute_reward(
                response_text, gold, domain,
                metadata=metadata, use_format_reward=False,
            )
            results.append({
                "id": rec["id"],
                "domain": domain,
                "subdomain": rec.get("subdomain", ""),
                "difficulty": rec.get("difficulty"),
                "gold": gold,
                "prediction": response_text,
                "correct": reward > 0.5,
                "reward": reward,
            })

        elif mode == "majority":
            # Majority voting across n_samples
            responses = [o.text for o in output.outputs]
            rewards = [
                compute_reward(r, gold, domain, metadata=metadata, use_format_reward=False)
                for r in responses
            ]
            # Extract answers and vote
            if domain == "math":
                answers = [extract_math_answer(r) for r in responses]
            else:
                answers = [extract_mcq_answer(r) for r in responses]

            # Count votes (filter None)
            from collections import Counter
            valid_answers = [a for a in answers if a is not None]
            if valid_answers:
                majority = Counter(valid_answers).most_common(1)[0][0]
                majority_correct = compute_reward(
                    f"The answer is {majority}", gold, domain,
                    metadata=metadata, use_format_reward=False,
                ) > 0.5
            else:
                majority = None
                majority_correct = False

            results.append({
                "id": rec["id"],
                "domain": domain,
                "subdomain": rec.get("subdomain", ""),
                "difficulty": rec.get("difficulty"),
                "gold": gold,
                "prediction": majority,
                "correct": majority_correct,
                "reward": float(majority_correct),
                "n_correct": sum(1 for r in rewards if r > 0.5),
                "n_total": len(rewards),
            })

        elif mode == "sample":
            # pass@k evaluation
            responses = [o.text for o in output.outputs]
            rewards = [
                compute_reward(r, gold, domain, metadata=metadata, use_format_reward=False)
                for r in responses
            ]
            n_correct = sum(1 for r in rewards if r > 0.5)
            results.append({
                "id": rec["id"],
                "domain": domain,
                "subdomain": rec.get("subdomain", ""),
                "difficulty": rec.get("difficulty"),
                "gold": gold,
                "predictions": responses,
                "rewards": rewards,
                "n_correct": n_correct,
                "n_total": len(responses),
            })

    # Compute aggregate metrics
    metrics = compute_domain_metrics(results, mode=mode, n_samples=n_samples)

    # Save results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "config": {
                "model_path": model_path,
                "test_data": test_data_path,
                "mode": mode,
                "n_samples": n_samples,
                "temperature": temperature,
            },
            "metrics": metrics,
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"  Results saved to {output_path}")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluation Results: {model_path}")
    logger.info(f"{'='*60}")
    for key, value in metrics.get("overall", {}).items():
        logger.info(f"  {key}: {value:.4f}")
    for subdomain, sub_metrics in metrics.get("by_subdomain", {}).items():
        logger.info(f"  [{subdomain}] accuracy: {sub_metrics.get('accuracy', 0):.4f} ({sub_metrics.get('count', 0)} samples)")
    if "by_difficulty" in metrics:
        for diff, diff_metrics in metrics["by_difficulty"].items():
            logger.info(f"  [difficulty={diff}] accuracy: {diff_metrics.get('accuracy', 0):.4f}")

    return metrics


# ---------------------------------------------------------------------------
# Multi-benchmark evaluation
# ---------------------------------------------------------------------------

def evaluate_all_benchmarks(
    model_path: str,
    test_dir: str,
    output_dir: str,
    base_model: str | None = None,
    domains: list[str] | None = None,
    mode: str = "greedy",
    n_samples: int = 1,
    temperature: float = 0.0,
    tensor_parallel_size: int = 1,
) -> dict:
    """Evaluate a model on all available test sets."""
    test_path = Path(test_dir)
    all_domains = domains or ["math", "science", "law", "medicine", "code", "commonsense"]

    all_metrics = {}
    for domain in all_domains:
        test_file = test_path / domain / "test.jsonl"
        if not test_file.exists():
            logger.warning(f"  No test file for {domain}: {test_file}")
            continue

        out_file = os.path.join(output_dir, f"{domain}_results.json")
        metrics = evaluate(
            model_path=model_path,
            test_data_path=str(test_file),
            output_path=out_file,
            base_model=base_model,
            mode=mode,
            n_samples=n_samples,
            temperature=temperature,
            tensor_parallel_size=tensor_parallel_size,
        )
        all_metrics[domain] = metrics

    # Aggregate across domains
    summary = aggregate_results(all_metrics)

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model or LoRA adapter")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model (for LoRA adapters)")
    parser.add_argument("--test_data", type=str, default=None,
                        help="Path to specific test JSONL file")
    parser.add_argument("--test_dir", type=str, default="data/raw",
                        help="Directory containing domain test sets")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output file for results")
    parser.add_argument("--output_dir", type=str, default="results/eval",
                        help="Output directory for multi-benchmark results")
    parser.add_argument("--domains", nargs="+", default=None)
    parser.add_argument("--mode", choices=["greedy", "sample", "majority"],
                        default="greedy")
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    if args.test_data:
        # Single benchmark evaluation
        output_path = args.output_path or "results/eval/results.json"
        evaluate(
            model_path=args.model_path,
            test_data_path=args.test_data,
            output_path=output_path,
            base_model=args.base_model,
            mode=args.mode,
            n_samples=args.n_samples,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            batch_size=args.batch_size,
        )
    else:
        # All benchmarks
        evaluate_all_benchmarks(
            model_path=args.model_path,
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            base_model=args.base_model,
            domains=args.domains,
            mode=args.mode,
            n_samples=args.n_samples,
            temperature=args.temperature,
            tensor_parallel_size=args.tensor_parallel_size,
        )


if __name__ == "__main__":
    main()
