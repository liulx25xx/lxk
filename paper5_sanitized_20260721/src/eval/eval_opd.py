"""
Evaluate OPD models: in-domain + cross-domain (OOD).

Usage:
    python eval_opd.py --gpu 0 --domain math --adapter_path /path/to/final/ --model_tag opd_math_s42
    python eval_opd.py --gpu 1 --domain science --adapter_path /path/to/final/ --eval_type both
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reward.rewards import math_reward, mcq_reward, binary_reward, extract_math_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL = "/path/to/workspace/model/Qwen2.5-7B-Instruct"
MERGED_CACHE_DIR = PROJECT_ROOT / "merged_models_cache"


def math_reward_fixed(output: str, gold: str, **kwargs) -> float:
    gold_clean = gold.strip()
    if len(gold_clean) > 20 or "\\boxed" in gold_clean or "####" in gold_clean:
        extracted = extract_math_answer(gold_clean)
        if extracted:
            gold_clean = extracted
    return math_reward(output, gold_clean, **kwargs)


REWARD_DISPATCH = {
    "math": math_reward_fixed,
    "science": mcq_reward,
    "medicine": mcq_reward,
    "law": mcq_reward,  # OOD law uses MMLU (MCQ)
    "commonsense": mcq_reward,
}


def merge_adapter(adapter_path: str, merged_dir: str) -> str:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(config_path) as f:
        config = json.load(f)
    base_model_name = config.get("base_model_name_or_path", BASE_MODEL)

    logger.info(f"Merging adapter: {adapter_path} (base: {base_model_name})")
    t0 = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(merged_dir)

    logger.info(f"Merged in {time.time()-t0:.1f}s → {merged_dir}")
    del merged, model, base_model
    gc.collect()
    torch.cuda.empty_cache()
    return merged_dir


def evaluate_on_test(
    model_path: str,
    test_file: str,
    domain: str,
    max_tokens: int = 1024,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.50,
) -> dict:
    from vllm import LLM, SamplingParams

    records = []
    with open(test_file) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        return {"accuracy": 0, "n_test": 0, "correct": 0}

    logger.info(f"  Test: {test_file} ({len(records)} samples)")
    t0 = time.time()
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    logger.info(f"  Model loaded in {time.time()-t0:.1f}s")

    prompts = [r["prompt"] for r in records]
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens)

    t1 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    logger.info(f"  Generated in {time.time()-t1:.1f}s")

    reward_fn = REWARD_DISPATCH[domain]
    n_correct = 0
    for rec, output in zip(records, outputs):
        response_text = output.outputs[0].text
        gold = rec["gold_answer"]
        r = reward_fn(response_text, gold)
        if r > 0.5:
            n_correct += 1

    accuracy = n_correct / len(records)
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return {"accuracy": accuracy, "n_test": len(records), "correct": n_correct}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to adapter final/ dir")
    parser.add_argument("--train_domain", type=str, required=True, help="Domain this model was trained on")
    parser.add_argument("--model_tag", type=str, required=True, help="Tag for result files")
    parser.add_argument("--eval_type", type=str, default="both", choices=["indomain", "ood", "both"])
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.50)
    args = parser.parse_args()

    # Only set CUDA_VISIBLE_DEVICES if not already set by the environment
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Merge adapter
    merged_dir = str(MERGED_CACHE_DIR / f"opd_merge_{args.model_tag}")
    os.makedirs(MERGED_CACHE_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(merged_dir, "config.json")):
        merge_adapter(args.adapter_path, merged_dir)
    else:
        logger.info(f"Using cached merge: {merged_dir}")

    results = {"model_tag": args.model_tag, "train_domain": args.train_domain, "method": "opd"}

    # In-domain eval
    if args.eval_type in ("indomain", "both"):
        test_file = PROJECT_ROOT / "data" / "processed" / args.train_domain / "rlvr_test.jsonl"
        if test_file.exists():
            logger.info(f"=== In-domain eval: {args.train_domain} ===")
            res = evaluate_on_test(
                merged_dir, str(test_file), args.train_domain,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            results["indomain"] = res
            logger.info(f"  In-domain accuracy: {res['accuracy']*100:.1f}% ({res['correct']}/{res['n_test']})")

    # Cross-domain eval
    if args.eval_type in ("ood", "both"):
        ood_domains = ["math", "science", "medicine"]
        for ood_domain in ood_domains:
            if ood_domain == args.train_domain:
                continue  # skip same domain
            ood_file = PROJECT_ROOT / "data" / "ood" / f"{ood_domain}_ood.jsonl"
            if not ood_file.exists():
                # Try processed test set as fallback
                ood_file = PROJECT_ROOT / "data" / "processed" / ood_domain / "rlvr_test.jsonl"
            if ood_file.exists():
                logger.info(f"=== Cross-domain eval: {args.train_domain} → {ood_domain} ===")
                res = evaluate_on_test(
                    merged_dir, str(ood_file), ood_domain,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                )
                results[f"ood_{ood_domain}"] = res
                logger.info(f"  {args.train_domain}→{ood_domain}: {res['accuracy']*100:.1f}%")

    # Save results
    results_dir = PROJECT_ROOT / "eval_results" / "opd"
    os.makedirs(results_dir, exist_ok=True)
    result_file = results_dir / f"{args.model_tag}.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {result_file}")

    # Cleanup merged model to save disk
    import shutil
    if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)
        logger.info(f"Cleaned up merged model: {merged_dir}")


if __name__ == "__main__":
    main()
