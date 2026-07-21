#!/usr/bin/env python3
"""
Batch evaluate all unevaluated models. Handles LoRA, Full FT, and OPD models.
Evaluates a single model per invocation (specify by index into the queue).

Usage:
    # Evaluate model at index 0 in the queue on GPU 0
    CUDA_VISIBLE_DEVICES=0 python eval_batch_v2.py --index 0
    
    # List all models needing eval
    python eval_batch_v2.py --list
"""
import argparse
import gc
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reward.rewards import math_reward, mcq_reward, code_reward, binary_reward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def law_reward(output: str, gold: str, **kwargs) -> float:
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


def get_all_unevaluated_models():
    """Find all models with final/ but no eval result."""
    outputs_dir = PROJECT_ROOT / "outputs"
    eval_dir = PROJECT_ROOT / "eval_results" / "trained"
    
    models = []
    
    domains = ["math", "medicine", "science", "code", "law", "commonsense"]
    methods = ["grpo", "sft", "opd"]  # Include OPD
    
    for domain in domains:
        for method in methods:
            method_dir = outputs_dir / domain / method
            if not method_dir.exists():
                continue
            for run_dir in sorted(method_dir.iterdir()):
                final_dir = run_dir / "final"
                if not final_dir.exists():
                    continue
                
                has_adapter = (final_dir / "adapter_config.json").exists()
                has_model = (final_dir / "config.json").exists()
                
                if not has_adapter and not has_model:
                    continue
                
                name = run_dir.name
                result_name = f"{domain}_{method}_{name}"
                result_path = eval_dir / f"{result_name}.json"
                
                if result_path.exists():
                    continue  # Already evaluated
                
                # Parse metadata
                seed_match = re.search(r"(?:^|[_-])(?:s|seed)(\d+)(?:[_-]|$)", name)
                seed = int(seed_match.group(1)) if seed_match else 42
                n_match = re.search(r"n(\d+)", name)
                n_train = int(n_match.group(1)) if n_match else 0
                
                is_fullft = has_model and not has_adapter
                
                models.append({
                    "domain": domain,
                    "method": method,
                    "run_name": name,
                    "seed": seed,
                    "n_train": n_train,
                    "final_path": str(final_dir),
                    "result_name": result_name,
                    "is_fullft": is_fullft,
                })
    
    # Sort by priority: medicine first, then by method
    priority = {
        "medicine_grpo_seed42_n2000_fullft": 0,
        "medicine_grpo_seed42_n2000_lr5e6": 1,
        "medicine_grpo_seed42_n2000_dapo": 2,
        "medicine_opd_seed42_n2000": 3,
        "math_grpo_seed42_mathfull_n2000": 4,
    }
    
    def sort_key(m):
        p = priority.get(m["result_name"], 100)
        domain_order = {"medicine": 0, "math": 1, "science": 2, "law": 3, "commonsense": 4, "code": 5}
        return (p, domain_order.get(m["domain"], 99), m["method"], m["run_name"])
    
    models.sort(key=sort_key)
    return models


def evaluate_single_model(model_info: dict):
    """Evaluate a single model."""
    domain = model_info["domain"]
    method = model_info["method"]
    run_name = model_info["run_name"]
    final_path = model_info["final_path"]
    result_name = model_info["result_name"]
    is_fullft = model_info["is_fullft"]
    
    eval_dir = PROJECT_ROOT / "eval_results" / "trained"
    eval_dir.mkdir(parents=True, exist_ok=True)
    result_path = eval_dir / f"{result_name}.json"
    
    if result_path.exists():
        logger.info(f"SKIP (already exists): {result_name}")
        return json.load(open(result_path))
    
    logger.info(f"{'='*60}")
    logger.info(f"Evaluating: {domain}/{method}/{run_name}")
    logger.info(f"  Full FT: {is_fullft}")
    logger.info(f"{'='*60}")
    
    # Determine model path for vLLM
    merged_cache = PROJECT_ROOT / "merged_models_cache"
    merged_cache.mkdir(exist_ok=True)
    
    if is_fullft:
        # Full FT: model is directly usable
        model_path = final_path
        logger.info(f"  Full FT model, loading directly: {model_path}")
    else:
        # LoRA: need to merge
        merged_dir = str(merged_cache / result_name)
        if not os.path.exists(os.path.join(merged_dir, "config.json")):
            logger.info(f"  Merging LoRA adapter...")
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            config_path = os.path.join(final_path, "adapter_config.json")
            with open(config_path) as f:
                config = json.load(f)
            base_model_name = config.get("base_model_name_or_path", "Qwen/Qwen2.5-7B-Instruct")
            
            t0 = time.time()
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base_model, final_path)
            merged = model.merge_and_unload()
            merged.save_pretrained(merged_dir)
            
            tokenizer = AutoTokenizer.from_pretrained(final_path, trust_remote_code=True)
            tokenizer.save_pretrained(merged_dir)
            
            logger.info(f"  Merged in {time.time()-t0:.1f}s")
            del merged, model, base_model
            gc.collect()
            torch.cuda.empty_cache()
        else:
            logger.info(f"  Using cached merge: {merged_dir}")
        model_path = merged_dir
    
    # Determine test file and subdomain filter
    data_dir = PROJECT_ROOT / "data" / "processed"
    test_file = str(data_dir / domain / "rlvr_test.jsonl")
    
    if not os.path.exists(test_file):
        logger.error(f"  Test file not found: {test_file}")
        return None
    
    subdomain_filter = None
    if domain == "math":
        if "mathfull" in run_name.lower():
            subdomain_filter = None  # eval on full test (both GSM8K and MATH)
        else:
            subdomain_filter = "gsm8k"  # GSM8K-trained → eval on GSM8K
    
    max_tokens = 2048 if domain in ("math", "code") else 1024
    
    # Load test data
    records = []
    with open(test_file) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if subdomain_filter:
                sd = rec.get("metadata", {}).get("subdomain", "")
                if sd != subdomain_filter:
                    continue
            if not rec.get("gold_answer", "").strip():
                continue
            records.append(rec)
    
    logger.info(f"  Test samples: {len(records)} (subdomain={subdomain_filter or 'all'})")
    
    if not records:
        return {"accuracy": 0, "n_test": 0, "correct": 0}
    
    # Load model with vLLM
    from vllm import LLM, SamplingParams
    
    logger.info(f"  Loading vLLM: {model_path}")
    t0 = time.time()
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
    )
    logger.info(f"  Loaded in {time.time()-t0:.1f}s")
    
    prompts = [r["prompt"] for r in records]
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens)
    
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
            "prediction": response_text[:500],
            "reward": reward,
            "correct": correct,
        })
    
    accuracy = n_correct / max(len(records), 1)
    
    # Cleanup vLLM
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    
    # Save result
    result_record = {
        "domain": domain,
        "method": method,
        "run_name": run_name,
        "seed": model_info["seed"],
        "n_train": model_info["n_train"],
        "accuracy": accuracy,
        "n_test": len(records),
        "correct": n_correct,
        "model_path": final_path,
        "subdomain_evaluated": subdomain_filter or "all",
        "is_fullft": is_fullft,
    }
    
    with open(result_path, "w") as f:
        json.dump(result_record, f, indent=2)
    
    # Save per-sample
    per_sample_path = eval_dir / f"{result_name}_samples.jsonl"
    with open(per_sample_path, "w") as f:
        for s in per_sample:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    logger.info(f"  RESULT: accuracy={accuracy:.4f} ({n_correct}/{len(records)})")
    logger.info(f"  Saved: {result_path}")
    
    # Cleanup merged model
    if not is_fullft:
        merged_dir = str(merged_cache / result_name)
        if os.path.exists(merged_dir):
            shutil.rmtree(merged_dir)
            logger.info(f"  Cleaned merged model cache")
    
    return result_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List all models needing eval")
    parser.add_argument("--index", type=int, default=None, help="Evaluate model at this index")
    parser.add_argument("--start", type=int, default=None, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    args = parser.parse_args()
    
    models = get_all_unevaluated_models()
    
    if args.list or (args.index is None and args.start is None):
        print(f"\n{len(models)} models needing evaluation:")
        print(f"{'Idx':<4} {'Domain':<12} {'Method':<6} {'Run Name':<30} {'FullFT':<6}")
        print("-" * 60)
        for i, m in enumerate(models):
            print(f"{i:<4} {m['domain']:<12} {m['method']:<6} {m['run_name']:<30} {m['is_fullft']}")
        return
    
    if args.index is not None:
        if args.index >= len(models):
            logger.info(f"Index {args.index} out of range (only {len(models)} models)")
            return
        evaluate_single_model(models[args.index])
    elif args.start is not None:
        end = args.end or len(models)
        for i in range(args.start, min(end, len(models))):
            logger.info(f"\n[{i+1}/{len(models)}] Evaluating model {i}...")
            result = evaluate_single_model(models[i])
            if result:
                logger.info(f"  → {result['domain']}/{result['method']}/{result['run_name']}: {result['accuracy']:.4f}")


if __name__ == "__main__":
    main()
