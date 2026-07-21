"""
Zero-shot base model evaluation on all 6 domain test sets.

Evaluates Qwen2.5-7B-Instruct with greedy decoding (T=0) and scores using
domain-specific reward functions. Outputs per-sample results + summary + difficulty stratification.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reward.rewards import (
    math_reward,
    mcq_reward,
    code_reward,
    binary_reward,
    extract_math_answer,
    extract_mcq_answer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reward dispatch — handles law's mixed answer types
# ---------------------------------------------------------------------------

def law_reward(output: str, gold: str, **kwargs) -> float:
    """
    Law domain reward: handles both binary (Yes/No) and categorical answers.
    For binary questions, uses binary_reward logic.
    For categorical (generic/descriptive/fanciful/etc), uses exact match on extracted answer.
    """
    gold_lower = gold.strip().lower()

    # Binary yes/no
    if gold_lower in ("yes", "no"):
        return binary_reward(output, gold, **kwargs)

    # Categorical answers — extract from output
    output_lower = output.strip().lower()

    # Check if the gold answer appears in the output (case insensitive)
    if gold_lower in output_lower:
        return 1.0

    # For longer gold answers (like full sentences), check containment
    if len(gold_lower) > 20:
        # Check if a significant portion of the gold answer is in output
        gold_words = set(gold_lower.split())
        output_words = set(output_lower.split())
        overlap = gold_words & output_words
        if len(overlap) / max(len(gold_words), 1) > 0.8:
            return 1.0

    return 0.0


def math_reward_fixed(output: str, gold: str, **kwargs) -> float:
    """
    Math reward that handles both numeric gold answers (GSM8K: "42")
    and full-solution gold answers (MATH: "...\\boxed{42}...").
    Extracts the answer from gold text if needed.
    """
    # If gold looks like a full solution (contains \boxed or is very long), extract answer first
    gold_clean = gold.strip()
    if len(gold_clean) > 20 or "\\boxed" in gold_clean or "####" in gold_clean:
        extracted = extract_math_answer(gold_clean)
        if extracted:
            gold_clean = extracted
    return math_reward(output, gold_clean, **kwargs)


REWARD_DISPATCH = {
    "math": math_reward_fixed,
    "code": code_reward,
    "science": mcq_reward,
    "medicine": mcq_reward,
    "commonsense": mcq_reward,
    "law": law_reward,
}


# ---------------------------------------------------------------------------
# Code test case lookup from raw data
# ---------------------------------------------------------------------------

def build_code_test_lookup(raw_code_path: str) -> dict:
    """Build a mapping from record ID to test metadata from raw code data."""
    lookup = {}
    with open(raw_code_path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = rec.get("id", "")
            meta = rec.get("metadata", {})
            if meta.get("test") or meta.get("test_list"):
                lookup[rid] = meta
    logger.info(f"Built code test lookup with {len(lookup)} entries")
    return lookup


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_base_eval(
    model_name: str,
    test_files: dict[str, str],
    output_dir: str,
    raw_code_path: str | None = None,
    tensor_parallel_size: int = 4,
    max_model_len: int = 4096,
    max_tokens_math_code: int = 2048,
    max_tokens_mcq: int = 1024,
    code_timeout: int = 10,
    subsample_limit: int | None = None,
):
    """Run zero-shot evaluation on all domains."""
    from vllm import LLM, SamplingParams

    os.makedirs(output_dir, exist_ok=True)

    # Build code test lookup if raw data available
    code_test_lookup = {}
    if raw_code_path and os.path.exists(raw_code_path):
        code_test_lookup = build_code_test_lookup(raw_code_path)

    # Load model once
    logger.info(f"Loading model: {model_name} with tp={tensor_parallel_size}")
    t0 = time.time()
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.90,
    )
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    summary = {}

    # Process domains in order: small first to validate pipeline
    domain_order = ["medicine", "code", "science", "math", "law", "commonsense"]
    domains_to_eval = [d for d in domain_order if d in test_files]

    for domain in domains_to_eval:
        test_file = test_files[domain]
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating domain: {domain}")
        logger.info(f"Test file: {test_file}")

        # Load test data
        records = []
        with open(test_file) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        # Filter out records with empty gold answers (e.g. HellaSwag test set)
        n_before = len(records)
        records = [r for r in records if r["gold_answer"].strip()]
        if len(records) < n_before:
            logger.info(f"Filtered out {n_before - len(records)} records with empty gold answers")

        if subsample_limit and len(records) > subsample_limit:
            import random
            random.seed(42)
            records = random.sample(records, subsample_limit)
            logger.info(f"Subsampled to {subsample_limit} records")
        else:
            logger.info(f"Loaded {len(records)} records (with valid gold answers)")

        # Prompts already have chat template
        prompts = [r["prompt"] for r in records]

        # Set max_tokens based on domain
        if domain in ("math", "code"):
            max_tokens = max_tokens_math_code
        else:
            max_tokens = max_tokens_mcq

        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
        )

        # Generate
        logger.info(f"Generating {len(prompts)} completions (max_tokens={max_tokens})...")
        t1 = time.time()
        outputs = llm.generate(prompts, sampling_params)
        gen_time = time.time() - t1
        logger.info(f"Generation: {gen_time:.1f}s ({len(records)/max(gen_time, 0.1):.1f} samples/s)")

        # Score
        reward_fn = REWARD_DISPATCH[domain]
        results = []
        n_correct = 0

        for rec, output in zip(records, outputs):
            response_text = output.outputs[0].text
            gold = rec["gold_answer"]
            metadata = rec.get("metadata", {})

            # For code: inject test cases from raw data
            if domain == "code":
                rid = rec["id"]
                if rid in code_test_lookup:
                    metadata = {**metadata, **code_test_lookup[rid]}
                reward = reward_fn(response_text, gold, metadata=metadata, timeout=code_timeout)
            else:
                reward = reward_fn(response_text, gold)

            correct = reward > 0.5
            if correct:
                n_correct += 1

            results.append({
                "id": rec["id"],
                "domain": domain,
                "subdomain": metadata.get("subdomain", ""),
                "gold": gold,
                "prediction": response_text,
                "reward": reward,
                "correct": correct,
            })

        accuracy = n_correct / max(len(results), 1)
        logger.info(f"  {domain}: accuracy={accuracy:.4f} ({n_correct}/{len(results)})")

        # Per-subdomain breakdown
        subdomain_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in results:
            sd = r["subdomain"] or "unknown"
            subdomain_stats[sd]["total"] += 1
            if r["correct"]:
                subdomain_stats[sd]["correct"] += 1

        for sd, stats in sorted(subdomain_stats.items()):
            sd_acc = stats["correct"] / max(stats["total"], 1)
            logger.info(f"    {sd}: {sd_acc:.4f} ({stats['correct']}/{stats['total']})")

        # Save per-sample results
        results_path = os.path.join(output_dir, f"base_{domain}.jsonl")
        with open(results_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"  Saved per-sample results to {results_path}")

        # Build summary entry
        subdomain_summary = {}
        for sd, stats in sorted(subdomain_stats.items()):
            subdomain_summary[sd] = {
                "accuracy": stats["correct"] / max(stats["total"], 1),
                "n": stats["total"],
                "correct": stats["correct"],
            }

        summary[domain] = {
            "accuracy": accuracy,
            "n": len(results),
            "correct": n_correct,
            "subdomains": subdomain_summary,
        }

    # Save summary
    summary_path = os.path.join(output_dir, "base_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSummary saved to {summary_path}")

    # Build difficulty stratification
    stratification = {}
    for domain in domains_to_eval:
        results_path = os.path.join(output_dir, f"base_{domain}.jsonl")
        if not os.path.exists(results_path):
            continue

        easy_ids = []
        hard_ids = []
        with open(results_path) as f:
            for line in f:
                r = json.loads(line)
                if r["correct"]:
                    easy_ids.append(r["id"])
                else:
                    hard_ids.append(r["id"])

        total = len(easy_ids) + len(hard_ids)
        stratification[domain] = {
            "easy_ids": easy_ids,
            "hard_ids": hard_ids,
            "easy_count": len(easy_ids),
            "hard_count": len(hard_ids),
            "total": total,
            "easy_pct": len(easy_ids) / max(total, 1),
            "hard_pct": len(hard_ids) / max(total, 1),
        }

    strat_path = os.path.join(output_dir, "difficulty_stratification.json")
    with open(strat_path, "w") as f:
        json.dump(stratification, f, indent=2, ensure_ascii=False)
    logger.info(f"Difficulty stratification saved to {strat_path}")

    # Print final summary table
    logger.info(f"\n{'='*60}")
    logger.info("BASE MODEL EVALUATION SUMMARY (Qwen2.5-7B-Instruct, zero-shot)")
    logger.info(f"{'='*60}")
    logger.info(f"{'Domain':<15} {'Accuracy':>10} {'Correct':>10} {'Total':>10} {'Easy%':>10}")
    logger.info(f"{'-'*55}")
    for domain in domains_to_eval:
        if domain in summary:
            s = summary[domain]
            strat = stratification.get(domain, {})
            easy_pct = strat.get("easy_pct", 0)
            logger.info(
                f"{domain:<15} {s['accuracy']:>10.4f} {s['correct']:>10d} {s['n']:>10d} {easy_pct:>10.1%}"
            )
    logger.info(f"{'='*60}")

    return summary, stratification


def main():
    parser = argparse.ArgumentParser(description="Base model zero-shot evaluation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size")
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJECT_ROOT / "eval_results"))
    parser.add_argument("--data_dir", type=str,
                        default=str(PROJECT_ROOT / "data" / "processed"))
    parser.add_argument("--raw_code_path", type=str,
                        default=str(PROJECT_ROOT / "data" / "raw" / "code" / "test.jsonl"))
    parser.add_argument("--domains", nargs="+", default=None,
                        help="Specific domains to evaluate (default: all)")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Subsample large datasets to this many records")
    parser.add_argument("--code_timeout", type=int, default=10)
    args = parser.parse_args()

    # Build test file paths
    all_domains = ["math", "code", "science", "medicine", "law", "commonsense"]
    domains = args.domains or all_domains

    test_files = {}
    for domain in domains:
        test_file = os.path.join(args.data_dir, domain, "rlvr_test.jsonl")
        if os.path.exists(test_file):
            test_files[domain] = test_file
        else:
            logger.warning(f"Test file not found for {domain}: {test_file}")

    if not test_files:
        logger.error("No test files found!")
        sys.exit(1)

    logger.info(f"Domains to evaluate: {list(test_files.keys())}")

    # Add file handler for logging
    log_dir = PROJECT_ROOT / "logs"
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(log_dir / "base_eval.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)

    run_base_eval(
        model_name=args.model,
        test_files=test_files,
        output_dir=args.output_dir,
        raw_code_path=args.raw_code_path,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        subsample_limit=args.subsample,
        code_timeout=args.code_timeout,
    )


if __name__ == "__main__":
    main()
