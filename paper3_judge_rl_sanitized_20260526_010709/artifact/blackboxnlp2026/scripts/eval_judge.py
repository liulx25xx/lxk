"""
Paper 3: Evaluate Judge Model

Evaluates a judge model (base or RL-trained) on RewardBench test set.
Computes:
  - Accuracy: agreement with gold labels
  - Consistency: position-swap invariance (original vs swapped pair)
  - Fixed-confidence proxy: Brier-shaped score using emitted or fallback confidence

Features:
  - Batched generation (much faster than one-at-a-time)
  - Incremental saving (results saved after each batch)
  - Resume support (skips already-evaluated instances)
  - Auto-detects swap data if not specified

Usage:
    # Base model evaluation
    python eval_judge.py \
        --model_path Qwen/Qwen2.5-7B-Instruct \
        --test_data ../data/eval/rewardbench_test.json \
        --output_dir ../results/baseline_qwen7b

    # RL-trained model (LoRA adapter)
    python eval_judge.py \
        --model_path Qwen/Qwen2.5-7B-Instruct \
        --adapter_path ../results/EXP-006_accuracy_only/final_model \
        --test_data ../data/eval/rewardbench_test.json \
        --output_dir ../results/EXP-006_accuracy_only/eval

    # Resume from crash
    python eval_judge.py \
        --model_path Qwen/Qwen2.5-7B-Instruct \
        --test_data ../data/eval/rewardbench_test.json \
        --output_dir ../results/baseline_qwen7b --resume
"""

import argparse
import json
import os
import re
import sys
import time
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def setup_logging(output_dir):
    """Setup logging to both file and stdout."""
    log_file = output_dir / "eval.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(__name__)


def parse_judge_output(text):
    """Parse judge model output to extract choice and confidence."""
    # Try format with confidence: [[A, 0.85]]
    match = re.search(r'\[\[(A|B|C),?\s*([\d.]+)\]\]', text)
    if match:
        choice = match.group(1)
        confidence = float(match.group(2))
        confidence = max(0.5, min(1.0, confidence))
        return {"choice": choice, "confidence": confidence, "has_confidence": True}

    # Fallback: [[A]] without confidence
    match = re.search(r'\[\[(A|B|C)\]\]', text)
    if match:
        choice = match.group(1)
        # Default confidence: high for A/B, neutral for tie
        confidence = 0.8 if choice != "C" else 0.5
        return {"choice": choice, "confidence": confidence, "has_confidence": False}

    # Parse failure — couldn't find [[X]] pattern
    return {"choice": "PARSE_FAIL", "confidence": 0.5, "has_confidence": False}


def batch_generate(model, tokenizer, prompts, max_new_tokens=512, temperature=0.1, disable_thinking=False):
    """Generate responses for a batch of prompts using left-padding."""
    # Prepare inputs with chat template
    texts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        # For Qwen3/3.5 thinking models: disable thinking to get direct answers
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if disable_thinking:
            template_kwargs["enable_thinking"] = False
        try:
            text = tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            # Fallback: model doesn't support enable_thinking parameter
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(text)

    # Tokenize with left padding for batch generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=4096).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only new tokens for each sequence
    responses = []
    input_len = inputs['input_ids'].shape[1]
    for i in range(len(prompts)):
        new_tokens = outputs[i][input_len:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        responses.append(response)

    return responses


def compute_metrics(results):
    """Compute aggregate metrics from list of result dicts."""
    total = len(results)
    if total == 0:
        return {}

    correct = sum(1 for r in results if r["is_correct"])
    consistent = sum(1 for r in results if r["is_consistent"])
    parse_fails_orig = sum(1 for r in results if r["predicted"] == "PARSE_FAIL")
    parse_fails_swap = sum(1 for r in results if r["swap_predicted"] == "PARSE_FAIL")
    brier_scores = [r["brier_score"] for r in results if r["predicted"] != "PARSE_FAIL"]

    avg_brier = sum(brier_scores) / len(brier_scores) if brier_scores else 1.0
    has_confidence_count = sum(1 for r in results if r.get("has_confidence", False))

    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"correct": 0, "consistent": 0, "total": 0, "brier": []}
        categories[cat]["total"] += 1
        if r["is_correct"]:
            categories[cat]["correct"] += 1
        if r["is_consistent"]:
            categories[cat]["consistent"] += 1
        if r["predicted"] != "PARSE_FAIL":
            categories[cat]["brier"].append(r["brier_score"])

    category_metrics = {}
    for cat, stats in sorted(categories.items()):
        category_metrics[cat] = {
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
            "consistency": stats["consistent"] / stats["total"] if stats["total"] > 0 else 0,
            "avg_brier": sum(stats["brier"]) / len(stats["brier"]) if stats["brier"] else 1.0,
            "n": stats["total"],
        }

    return {
        "n_samples": total,
        "accuracy": correct / total,
        "consistency": consistent / total,
        "avg_brier_score": avg_brier,
        "fixed_confidence_proxy": 1.0 - avg_brier,
        # Retained so historical result consumers continue to load new files.
        "calibration_score": 1.0 - avg_brier,
        "parse_failures_orig": parse_fails_orig,
        "parse_failures_swap": parse_fails_swap,
        "has_confidence_count": has_confidence_count,
        "per_category": category_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate judge model on RewardBench")
    parser.add_argument("--model_path", required=True, help="Base model path or HF model ID")
    parser.add_argument("--adapter_path", default=None, help="LoRA adapter path (if evaluating RL model)")
    parser.add_argument("--test_data", required=True, help="Path to test JSON")
    parser.add_argument("--swap_data", default=None, help="Path to swap test JSON (auto-detect if not given)")
    parser.add_argument("--output_dir", required=True, help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens to generate")
    parser.add_argument("--disable_thinking", action="store_true", help="Disable thinking mode for Qwen3/3.5 models")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples (for debugging)")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
    parser.add_argument("--resume", action="store_true", help="Resume from partial results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("Judge Model Evaluation")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Adapter: {args.adapter_path or 'None (base model)'}")
    logger.info(f"Test data: {args.test_data}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Output: {args.output_dir}")

    # Auto-detect swap data
    if args.swap_data is None:
        test_path = Path(args.test_data)
        swap_path = test_path.parent / test_path.name.replace("test.", "test_swap.")
        if swap_path.exists():
            args.swap_data = str(swap_path)
            logger.info(f"Auto-detected swap data: {args.swap_data}")
        else:
            logger.error(f"Could not find swap data at {swap_path}. Use --swap_data to specify.")
            sys.exit(1)

    # Load test data
    with open(args.test_data) as f:
        test_data = json.load(f)
    with open(args.swap_data) as f:
        swap_data = json.load(f)

    if args.max_samples:
        test_data = test_data[:args.max_samples]
        swap_data = swap_data[:args.max_samples]

    assert len(test_data) == len(swap_data), f"Mismatch: {len(test_data)} vs {len(swap_data)}"
    logger.info(f"Test samples: {len(test_data)}")

    # Resume: load existing partial results
    results_file = output_dir / "eval_results.json"
    existing_results = []
    start_idx = 0
    if args.resume and results_file.exists():
        with open(results_file) as f:
            existing_results = json.load(f)
        start_idx = len(existing_results)
        logger.info(f"Resuming from index {start_idx} ({start_idx} already done)")
        if start_idx >= len(test_data):
            logger.info("All samples already evaluated. Computing final metrics.")
            metrics = compute_metrics(existing_results)
            metrics["model"] = args.model_path
            metrics["adapter"] = args.adapter_path
            with open(output_dir / "metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            print_summary(metrics, logger)
            return

    # Load model
    logger.info("Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.adapter_path:
        logger.info(f"Loading LoRA adapter from {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()

    model.eval()
    logger.info(f"Model loaded in {time.time()-t0:.1f}s")

    # Set padding for batch generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Run evaluation in batches
    results = existing_results.copy()
    total = len(test_data)
    batch_size = args.batch_size
    start_time = time.time()

    for batch_start in range(start_idx, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_items = test_data[batch_start:batch_end]
        batch_swap = swap_data[batch_start:batch_end]

        # Generate for original pairs
        orig_prompts = [item["prompt"] for item in batch_items]
        orig_responses = batch_generate(
            model, tokenizer, orig_prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            disable_thinking=args.disable_thinking,
        )

        # Generate for swapped pairs
        swap_prompts = [item["prompt"] for item in batch_swap]
        swap_responses = batch_generate(
            model, tokenizer, swap_prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            disable_thinking=args.disable_thinking,
        )

        # Process batch results
        for i, (item, swap_item, resp_orig, resp_swap) in enumerate(
            zip(batch_items, batch_swap, orig_responses, swap_responses)
        ):
            parsed_orig = parse_judge_output(resp_orig)
            parsed_swap = parse_judge_output(resp_swap)

            gold = item["gold_label"]
            is_correct = parsed_orig["choice"] == gold

            # Consistency: if original says A, swap should say B (and vice versa)
            flip_map = {"A": "B", "B": "A", "C": "C", "PARSE_FAIL": "PARSE_FAIL"}
            expected_swap = flip_map.get(parsed_orig["choice"], "PARSE_FAIL")
            is_consistent = parsed_swap["choice"] == expected_swap

            # Brier score
            conf = parsed_orig["confidence"]
            correct_binary = 1.0 if is_correct else 0.0
            brier = (conf - correct_binary) ** 2

            result = {
                "id": item.get("original_id", batch_start + i),
                "category": item.get("category", "unknown"),
                "gold_label": gold,
                "predicted": parsed_orig["choice"],
                "confidence": parsed_orig["confidence"],
                "has_confidence": parsed_orig["has_confidence"],
                "swap_predicted": parsed_swap["choice"],
                "is_correct": is_correct,
                "is_consistent": is_consistent,
                "brier_score": brier,
                "response_orig": resp_orig[:800],
                "response_swap": resp_swap[:800],
            }
            results.append(result)

        # Incremental save after each batch
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Progress log
        done = len(results)
        elapsed = time.time() - start_time
        speed = elapsed / (done - start_idx) if done > start_idx else 0
        remaining = speed * (total - done)
        acc = sum(1 for r in results if r["is_correct"]) / done
        con = sum(1 for r in results if r["is_consistent"]) / done
        logger.info(
            f"[{done}/{total}] Acc={acc:.3f} Consist={con:.3f} "
            f"Speed={speed:.1f}s/sample ETA={remaining/60:.1f}min"
        )

    # Compute and save final metrics
    metrics = compute_metrics(results)
    metrics["model"] = args.model_path
    metrics["adapter"] = args.adapter_path
    metrics["elapsed_seconds"] = time.time() - start_time + (
        existing_results and 0 or 0  # rough; doesn't track previous runs
    )

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print_summary(metrics, logger)
    logger.info(f"Results saved to {output_dir}")


def print_summary(metrics, logger):
    """Print evaluation summary."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Model: {metrics.get('model', 'unknown')}")
    logger.info(f"Adapter: {metrics.get('adapter', 'None')}")
    logger.info(f"Samples: {metrics['n_samples']}")
    logger.info(f"Accuracy:    {metrics['accuracy']:.4f}")
    logger.info(f"Consistency: {metrics['consistency']:.4f}")
    logger.info(
        "Fixed-confidence proxy: "
        f"{metrics['fixed_confidence_proxy']:.4f} "
        f"(1-Brier; emitted confidence in "
        f"{metrics['has_confidence_count']}/{metrics['n_samples']} instances)"
    )
    logger.info(f"Parse failures (orig/swap): {metrics['parse_failures_orig']}/{metrics['parse_failures_swap']}")
    logger.info(f"Instances with confidence: {metrics['has_confidence_count']}/{metrics['n_samples']}")
    logger.info("")
    logger.info("Per-category:")
    for cat, m in sorted(metrics.get("per_category", {}).items()):
        logger.info(f"  {cat:30s} acc={m['accuracy']:.3f} consist={m['consistency']:.3f} n={m['n']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
