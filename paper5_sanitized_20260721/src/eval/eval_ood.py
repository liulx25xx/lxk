"""
Evaluate models on OOD (out-of-distribution) test sets.

Supports:
  - Base model (Qwen2.5-7B-Instruct)
  - LoRA adapter models (auto-merged before eval)
  - All 5 OOD domains: math, science, medicine, law, commonsense

Usage:
    python eval_ood.py --gpu 0 --domain math --model_path base
    python eval_ood.py --gpu 1 --domain medicine --model_path /path/to/adapter/final/
    python eval_ood.py --gpu 2 --domain science --model_path /path/to/adapter/final/ --model_tag grpo_lr2e5
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reward.rewards import math_reward, mcq_reward, binary_reward, extract_math_answer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
OOD_DATA_DIR = PROJECT_ROOT / "data" / "ood"
OOD_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "ood"
MERGED_CACHE_DIR = PROJECT_ROOT / "merged_models_cache"


# ---------------------------------------------------------------------------
# Reward dispatch (matches eval_trained_models.py)
# ---------------------------------------------------------------------------

def law_reward_ood(output: str, gold: str, **kwargs) -> float:
    """Law OOD uses MMLU (MCQ), so always use mcq_reward."""
    return mcq_reward(output, gold, **kwargs)


def math_reward_fixed(output: str, gold: str, **kwargs) -> float:
    """Math reward handling boxed gold answers."""
    gold_clean = gold.strip()
    if len(gold_clean) > 20 or "\\boxed" in gold_clean or "####" in gold_clean:
        extracted = extract_math_answer(gold_clean)
        if extracted:
            gold_clean = extracted
    return math_reward(output, gold_clean, **kwargs)


# OOD reward dispatch: MMLU-Law is MCQ (not binary like LegalBench)
REWARD_DISPATCH_OOD = {
    "math": math_reward_fixed,
    "science": mcq_reward,        # ARC-Challenge is MCQ
    "medicine": mcq_reward,       # MMLU-Med is MCQ
    "law": law_reward_ood,        # MMLU-Law is MCQ (NOT binary)
    "commonsense": mcq_reward,    # WinoGrande is MCQ (A/B)
}


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def merge_adapter(adapter_path: str, merged_dir: str) -> str:
    """Merge LoRA adapter into base model."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(config_path) as f:
        config = json.load(f)
    base_model_name = config.get("base_model_name_or_path", BASE_MODEL)

    logger.info(f"  Merging adapter: {adapter_path}")
    logger.info(f"  Base model: {base_model_name}")

    t0 = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(merged_dir)

    logger.info(f"  Merged in {time.time()-t0:.1f}s → {merged_dir}")

    del merged, model, base_model
    gc.collect()
    torch.cuda.empty_cache()
    return merged_dir


def get_model_path(model_path_arg: str, model_tag: str) -> str:
    """Resolve model path: 'base' → base model name, adapter → merged path."""
    if model_path_arg.lower() == "base":
        return BASE_MODEL

    # It's an adapter path — merge it
    adapter_path = model_path_arg.rstrip("/")
    merged_dir = str(MERGED_CACHE_DIR / f"ood_merge_{model_tag}")
    os.makedirs(MERGED_CACHE_DIR, exist_ok=True)

    if os.path.exists(os.path.join(merged_dir, "config.json")):
        logger.info(f"  Using cached merge: {merged_dir}")
        return merged_dir

    merge_adapter(adapter_path, merged_dir)
    return merged_dir


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_ood(
    model_path: str,
    domain: str,
    max_tokens: int = 2048,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.50,
) -> dict:
    """Evaluate model on OOD test set."""
    from vllm import LLM, SamplingParams

    # Load OOD data
    ood_file = OOD_DATA_DIR / f"{domain}_ood.jsonl"
    if not ood_file.exists():
        raise FileNotFoundError(f"OOD data not found: {ood_file}. Run prepare_ood_data.py first.")

    records = []
    with open(ood_file) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    logger.info(f"  OOD test samples: {len(records)}")
    if not records:
        return {"accuracy": 0, "n_test": 0, "correct": 0, "per_sample": []}

    # Load model
    logger.info(f"  Loading model: {model_path}")
    t0 = time.time()
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    logger.info(f"  Model loaded in {time.time()-t0:.1f}s")

    # Generate
    prompts = [r["prompt"] for r in records]
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens)

    logger.info(f"  Generating {len(prompts)} completions...")
    t1 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.time() - t1
    logger.info(f"  Generation: {gen_time:.1f}s ({len(records)/max(gen_time,0.1):.1f} samples/s)")

    # Score
    reward_fn = REWARD_DISPATCH_OOD[domain]
    n_correct = 0
    per_sample = []

    for rec, output in zip(records, outputs):
        response_text = output.outputs[0].text
        gold = rec["gold_answer"]

        reward = reward_fn(response_text, gold)
        correct = reward > 0.5
        if correct:
            n_correct += 1

        per_sample.append({
            "id": rec["id"],
            "gold": gold,
            "prediction": response_text[:500],
            "reward": reward,
            "correct": correct,
            "source": rec.get("source", ""),
            "subdomain": rec.get("metadata", {}).get("subdomain", ""),
        })

    accuracy = n_correct / max(len(records), 1)

    # Cleanup
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "accuracy": accuracy,
        "n_test": len(records),
        "correct": n_correct,
        "per_sample": per_sample,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate models on OOD test sets")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument("--domain", type=str, required=True,
                        choices=["math", "science", "medicine", "law", "commonsense"],
                        help="OOD domain to evaluate")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to LoRA adapter dir (or 'base' for base model)")
    parser.add_argument("--model_tag", type=str, default=None,
                        help="Short tag for this model (used in output filename)")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Max generation tokens (default: 2048 for math, 1024 for MCQ)")
    parser.add_argument("--gpu_mem", type=float, default=0.50,
                        help="vLLM gpu_memory_utilization (default: 0.50)")
    parser.add_argument("--output_dir", type=str, default=str(OOD_RESULTS_DIR))
    parser.add_argument("--cleanup_merge", action="store_true",
                        help="Delete merged model after eval to save disk")
    args = parser.parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("HF_HOME", "/path/to/workspace/cache/huggingface")

    # Determine model tag
    if args.model_tag:
        model_tag = args.model_tag
    elif args.model_path.lower() == "base":
        model_tag = "base"
    else:
        # Derive tag from path
        parts = Path(args.model_path).parts
        model_tag = "_".join(parts[-3:]) if len(parts) >= 3 else Path(args.model_path).stem

    # Determine max tokens
    if args.max_tokens:
        max_tokens = args.max_tokens
    elif args.domain == "math":
        max_tokens = 2048
    else:
        max_tokens = 1024

    # Setup logging
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(log_dir / f"eval_ood_{args.domain}_{model_tag}_gpu{args.gpu}.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info(f"OOD Eval: domain={args.domain}, model={args.model_path}, tag={model_tag}")

    # Resolve model path
    actual_model_path = get_model_path(args.model_path, model_tag)

    # Evaluate
    os.makedirs(args.output_dir, exist_ok=True)
    result = evaluate_ood(
        model_path=actual_model_path,
        domain=args.domain,
        max_tokens=max_tokens,
        gpu_memory_utilization=args.gpu_mem,
    )

    # Build result record
    result_record = {
        "domain": args.domain,
        "model_tag": model_tag,
        "model_path": args.model_path,
        "accuracy": result["accuracy"],
        "n_test": result["n_test"],
        "correct": result["correct"],
        "source": result["per_sample"][0]["source"] if result["per_sample"] else "",
    }

    # Subdomain breakdown
    subdomain_stats = {}
    for s in result["per_sample"]:
        sd = s.get("subdomain", "unknown")
        if sd not in subdomain_stats:
            subdomain_stats[sd] = {"correct": 0, "total": 0}
        subdomain_stats[sd]["total"] += 1
        if s["correct"]:
            subdomain_stats[sd]["correct"] += 1
    for sd, stats in subdomain_stats.items():
        stats["accuracy"] = stats["correct"] / max(stats["total"], 1)
    result_record["subdomain_breakdown"] = subdomain_stats

    # Save result
    result_path = os.path.join(args.output_dir, f"{args.domain}_{model_tag}_ood.json")
    with open(result_path, "w") as f:
        json.dump(result_record, f, indent=2)
    logger.info(f"  Result saved: {result_path}")

    # Save per-sample
    samples_path = os.path.join(args.output_dir, f"{args.domain}_{model_tag}_ood_samples.jsonl")
    with open(samples_path, "w") as f:
        for s in result["per_sample"]:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"OOD RESULT: {args.domain} | {model_tag}")
    logger.info(f"  Accuracy: {result['accuracy']:.4f} ({result['correct']}/{result['n_test']})")
    for sd, stats in sorted(subdomain_stats.items()):
        logger.info(f"  {sd}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
    logger.info(f"{'='*60}")

    print(f"\nOOD RESULT: {args.domain} | {model_tag} | "
          f"acc={result['accuracy']:.4f} ({result['correct']}/{result['n_test']})")

    # Cleanup merged model if requested
    if args.cleanup_merge and args.model_path.lower() != "base":
        merged_dir = str(MERGED_CACHE_DIR / f"ood_merge_{model_tag}")
        if os.path.exists(merged_dir):
            shutil.rmtree(merged_dir)
            logger.info(f"  Cleaned merged model: {merged_dir}")


if __name__ == "__main__":
    main()
