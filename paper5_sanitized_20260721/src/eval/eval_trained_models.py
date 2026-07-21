"""
Evaluate trained LoRA models on domain test sets.

For each model:
1. Merge LoRA adapter into base model
2. Use vLLM for fast greedy generation
3. Score with domain-specific reward functions
4. Save results incrementally

Usage:
    python eval_trained_models.py --gpu 0 --domain medicine
    python eval_trained_models.py --gpu 1 --domain math --subdomain gsm8k
    python eval_trained_models.py --gpu 2 --domain science
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
import shutil
from pathlib import Path

import torch

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reward.rewards import math_reward, mcq_reward, code_reward, binary_reward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reward dispatch (same as base_eval.py)
# ---------------------------------------------------------------------------

def law_reward(output: str, gold: str, **kwargs) -> float:
    """Law domain: handles binary + categorical answers."""
    gold_lower = gold.strip().lower()
    if gold_lower in ("yes", "no"):
        return binary_reward(output, gold, **kwargs)
    output_lower = output.strip().lower()
    if gold_lower in output_lower:
        return 1.0
    if len(gold_lower) > 20:
        gold_words = set(gold_lower.split())
        output_words = set(output_lower.split())
        overlap = gold_words & output_words
        if len(overlap) / max(len(gold_words), 1) > 0.8:
            return 1.0
    return 0.0


def math_reward_fixed(output: str, gold: str, **kwargs) -> float:
    """Math reward handling both numeric and boxed gold answers."""
    from reward.rewards import extract_math_answer
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
# Model discovery
# ---------------------------------------------------------------------------

def discover_models(outputs_dir: str, domain: str | None = None) -> list[dict]:
    """Find all trained models with final/ directory."""
    models = []
    outputs_path = Path(outputs_dir)

    domains = [domain] if domain else ["math", "medicine", "science", "code", "law", "commonsense"]

    for d in domains:
        domain_dir = outputs_path / d
        if not domain_dir.exists():
            continue
        for method in ["grpo", "sft"]:
            method_dir = domain_dir / method
            if not method_dir.exists():
                continue
            for run_dir in sorted(method_dir.iterdir()):
                final_dir = run_dir / "final"
                if final_dir.exists() and (final_dir / "adapter_config.json").exists():
                    # Parse run name for seed and n
                    name = run_dir.name
                    seed = 42
                    n_train = 0
                    seed_match = re.search(r"(?:^|[_-])(?:s|seed)(\d+)(?:[_-]|$)", name)
                    if seed_match:
                        seed = int(seed_match.group(1))
                    # Extract n
                    import re
                    n_match = re.search(r"n(\d+)", name)
                    if n_match:
                        n_train = int(n_match.group(1))

                    models.append({
                        "domain": d,
                        "method": method,
                        "run_name": name,
                        "seed": seed,
                        "n_train": n_train,
                        "adapter_path": str(final_dir),
                    })

    return models


# ---------------------------------------------------------------------------
# Merge and evaluate
# ---------------------------------------------------------------------------

def merge_adapter(adapter_path: str, merged_dir: str) -> str:
    """Merge LoRA adapter into base model and save."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Read base model from adapter config
    config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(config_path) as f:
        config = json.load(f)
    base_model_name = config.get("base_model_name_or_path", "Qwen/Qwen2.5-7B-Instruct")

    logger.info(f"  Merging: {adapter_path}")
    logger.info(f"  Base model: {base_model_name}")

    t0 = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir)

    # Copy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(merged_dir)

    logger.info(f"  Merged in {time.time()-t0:.1f}s → {merged_dir}")

    # Free memory aggressively
    del merged, model, base_model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # Force Python to release all CUDA tensors
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                del obj
        except:
            pass
    gc.collect()
    torch.cuda.empty_cache()

    return merged_dir


def evaluate_model(
    model_path: str,
    test_file: str,
    domain: str,
    subdomain_filter: str | None = None,
    max_tokens: int = 1024,
    max_model_len: int = 4096,
) -> dict:
    """Evaluate a model on a test set using vLLM."""
    from vllm import LLM, SamplingParams

    # Load test data
    records = []
    with open(test_file) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            # Filter by subdomain if specified
            if subdomain_filter:
                sd = rec.get("metadata", {}).get("subdomain", "")
                if sd != subdomain_filter:
                    continue
            # Skip empty gold answers
            if not rec.get("gold_answer", "").strip():
                continue
            records.append(rec)

    logger.info(f"  Test samples: {len(records)} (subdomain={subdomain_filter or 'all'})")

    if not records:
        return {"accuracy": 0, "n_test": 0, "correct": 0}

    # Load model with vLLM
    logger.info(f"  Loading model with vLLM: {model_path}")
    t0 = time.time()
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.50,
    )
    logger.info(f"  Model loaded in {time.time()-t0:.1f}s")

    # Prepare prompts
    prompts = [r["prompt"] for r in records]

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
    )

    # Generate
    logger.info(f"  Generating {len(prompts)} completions...")
    t1 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.time() - t1
    logger.info(f"  Generation: {gen_time:.1f}s ({len(records)/max(gen_time,0.1):.1f} samples/s)")

    # Score
    reward_fn = REWARD_DISPATCH[domain]
    n_correct = 0
    per_sample = []

    for rec, output in zip(records, outputs):
        response_text = output.outputs[0].text
        gold = rec["gold_answer"]
        metadata = rec.get("metadata", {})

        if domain == "code":
            reward = reward_fn(response_text, gold, metadata=metadata, timeout=10)
        else:
            reward = reward_fn(response_text, gold)

        correct = reward > 0.5
        if correct:
            n_correct += 1

        per_sample.append({
            "id": rec["id"],
            "gold": gold,
            "prediction": response_text[:500],  # truncate for storage
            "reward": reward,
            "correct": correct,
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
    parser = argparse.ArgumentParser(description="Evaluate trained LoRA models")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument("--domain", type=str, default=None,
                        help="Evaluate only this domain (default: all)")
    parser.add_argument("--subdomain", type=str, default=None,
                        help="Filter test set to this subdomain (e.g. gsm8k)")
    parser.add_argument("--method", type=str, default=None,
                        help="Filter by method: grpo or sft")
    parser.add_argument("--run", type=str, default=None,
                        help="Evaluate only this specific run name")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Override max_tokens (default: 2048 for math/code, 1024 for MCQ)")
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJECT_ROOT / "eval_results" / "trained"))
    parser.add_argument("--merged_cache", type=str,
                        default=str(PROJECT_ROOT / "merged_models_cache"))
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip models that already have results")
    args = parser.parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Setup logging to file
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(log_dir / f"eval_trained_gpu{args.gpu}.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)

    # Discover models
    outputs_dir = str(PROJECT_ROOT / "outputs")
    models = discover_models(outputs_dir, domain=args.domain)

    if args.method:
        models = [m for m in models if m["method"] == args.method]
    if args.run:
        models = [m for m in models if m["run_name"] == args.run]

    logger.info(f"Found {len(models)} models to evaluate")
    for m in models:
        logger.info(f"  {m['domain']}/{m['method']}/{m['run_name']}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.merged_cache, exist_ok=True)

    # Test files
    data_dir = PROJECT_ROOT / "data" / "processed"

    results_all = []

    for i, model_info in enumerate(models):
        domain = model_info["domain"]
        method = model_info["method"]
        run_name = model_info["run_name"]
        adapter_path = model_info["adapter_path"]

        # Determine output filename
        result_name = f"{domain}_{method}_{run_name}"
        result_path = os.path.join(args.output_dir, f"{result_name}.json")

        if args.skip_existing and os.path.exists(result_path):
            logger.info(f"[{i+1}/{len(models)}] SKIP (exists): {result_name}")
            # Load existing result
            with open(result_path) as f:
                results_all.append(json.load(f))
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"[{i+1}/{len(models)}] Evaluating: {domain}/{method}/{run_name}")
        logger.info(f"{'='*60}")

        # Determine test file and subdomain filter
        test_file = str(data_dir / domain / "rlvr_test.jsonl")
        if not os.path.exists(test_file):
            logger.error(f"  Test file not found: {test_file}")
            continue

        # For math models trained on GSM8K, only eval on GSM8K subdomain
        subdomain_filter = args.subdomain
        if domain == "math" and "gsm8k" in run_name.lower():
            subdomain_filter = "gsm8k"
        elif domain == "math" and method == "grpo":
            # All math GRPO were trained on GSM8K data
            subdomain_filter = "gsm8k"
        elif domain == "math" and method == "sft":
            # SFT also trained on GSM8K CoT, eval on GSM8K
            subdomain_filter = "gsm8k"

        # Determine max_tokens
        if args.max_tokens:
            max_tokens = args.max_tokens
        elif domain in ("math", "code"):
            max_tokens = 2048
        else:
            max_tokens = 1024

        # Merge adapter
        merged_dir = os.path.join(args.merged_cache, f"{domain}_{method}_{run_name}")
        if not os.path.exists(os.path.join(merged_dir, "config.json")):
            try:
                merge_adapter(adapter_path, merged_dir)
            except Exception as e:
                logger.error(f"  Merge failed: {e}")
                continue
        else:
            logger.info(f"  Using cached merge: {merged_dir}")

        # Evaluate
        try:
            result = evaluate_model(
                model_path=merged_dir,
                test_file=test_file,
                domain=domain,
                subdomain_filter=subdomain_filter,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error(f"  Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Build result record
        result_record = {
            "domain": domain,
            "method": method,
            "run_name": run_name,
            "seed": model_info["seed"],
            "n_train": model_info["n_train"],
            "accuracy": result["accuracy"],
            "n_test": result["n_test"],
            "correct": result["correct"],
            "model_path": adapter_path,
            "subdomain_evaluated": subdomain_filter or "all",
        }
        results_all.append(result_record)

        # Save individual result (without per_sample for compact storage)
        with open(result_path, "w") as f:
            json.dump(result_record, f, indent=2)
        logger.info(f"  RESULT: accuracy={result['accuracy']:.4f} ({result['correct']}/{result['n_test']})")
        logger.info(f"  Saved: {result_path}")

        # Save per-sample results separately
        per_sample_path = os.path.join(args.output_dir, f"{result_name}_samples.jsonl")
        with open(per_sample_path, "w") as f:
            for s in result.get("per_sample", []):
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        # Clean up merged model to save disk
        if os.path.exists(merged_dir):
            shutil.rmtree(merged_dir)
            logger.info(f"  Cleaned merged model cache")

    # Save combined summary
    summary_path = os.path.join(args.output_dir, "trained_summary.json")
    # Load any existing results and merge
    existing = []
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            existing = json.load(f)

    # Merge: update existing with new, keyed by domain+method+run_name
    def _result_key(r):
        return f"{r.get('domain','?')}_{r.get('method','?')}_{r.get('run_name', r.get('run','?'))}"
    existing_keys = {_result_key(r) for r in existing}
    for r in results_all:
        key = _result_key(r)
        if key not in existing_keys:
            existing.append(r)
        else:
            # Update existing
            for j, er in enumerate(existing):
                if _result_key(er) == key:
                    existing[j] = r
                    break

    with open(summary_path, "w") as f:
        json.dump(existing, f, indent=2)
    logger.info(f"\nSummary updated: {summary_path} ({len(existing)} total results)")

    # Print table
    logger.info(f"\n{'='*70}")
    logger.info("EVALUATION RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"{'Domain':<12} {'Method':<6} {'Run':<25} {'Acc':>8} {'N':>6}")
    logger.info(f"{'-'*70}")
    for r in sorted(results_all, key=lambda x: (x['domain'], x['method'], x.get('n_train', 0))):
        logger.info(f"{r['domain']:<12} {r['method']:<6} {r.get('run_name', r.get('run','?')):<25} {r['accuracy']:>8.4f} {r['n_test']:>6}")


if __name__ == "__main__":
    main()
