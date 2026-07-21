"""
Paper 3: Evaluate Judge with Alternative Position Labels

Tests whether the position shortcut depends on the specific labels "A"/"B"
or latches onto ANY positional marker (1/2, Left/Right, First/Second).

Key: We replace A/B labels in both the prompt AND the output parsing,
then map back to A/B for consistency computation.
"""

import argparse
import json
import os
import re
import sys
import time
import logging
from pathlib import Path

os.environ.setdefault('HF_HOME', '/path/to/cache/huggingface')
os.environ.setdefault('TRITON_CACHE_DIR', '/path/to/cache/triton')
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', '/path/to/cache/torch_inductor')
os.environ.setdefault('TMPDIR', '/path/to/cache/tmp')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# Label variant configurations
LABEL_VARIANTS = {
    "numeric": {
        "old_labels": [("Assistant A", "Assistant 1"), ("Assistant B", "Assistant 2"),
                       ("assistant A", "assistant 1"), ("assistant B", "assistant 2"),
                       ('"[[A]]"', '"[[1]]"'), ('"[[B]]"', '"[[2]]"'), ('"[[C]]"', '"[[tie]]"'),
                       ("[[A]]", "[[1]]"), ("[[B]]", "[[2]]"), ("[[C]]", "[[tie]]")],
        "parse_pattern": r'\[\[(1|2|tie),?\s*([\d.]*)\]\]',
        "parse_map": {"1": "A", "2": "B", "tie": "C"},
        "desc": "Response 1/2 instead of A/B",
    },
    "leftright": {
        "old_labels": [("Assistant A", "Assistant Left"), ("Assistant B", "Assistant Right"),
                       ("assistant A", "assistant Left"), ("assistant B", "assistant Right"),
                       ('"[[A]]"', '"[[Left]]"'), ('"[[B]]"', '"[[Right]]"'), ('"[[C]]"', '"[[tie]]"'),
                       ("[[A]]", "[[Left]]"), ("[[B]]", "[[Right]]"), ("[[C]]", "[[tie]]")],
        "parse_pattern": r'\[\[(Left|Right|tie),?\s*([\d.]*)\]\]',
        "parse_map": {"Left": "A", "Right": "B", "tie": "C"},
        "desc": "Response Left/Right instead of A/B",
    },
}


def setup_logging(output_dir):
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


def transform_prompt(prompt, variant_config):
    """Replace A/B labels with variant labels in the prompt."""
    result = prompt
    for old, new in variant_config["old_labels"]:
        result = result.replace(old, new)
    return result


def parse_judge_output(text, variant_config):
    """Parse judge output with variant-specific pattern, map back to A/B."""
    pattern = variant_config["parse_pattern"]
    parse_map = variant_config["parse_map"]

    match = re.search(pattern, text)
    if match:
        raw_choice = match.group(1)
        choice = parse_map.get(raw_choice, "PARSE_FAIL")
        conf_str = match.group(2)
        confidence = float(conf_str) if conf_str else 0.8
        confidence = max(0.5, min(1.0, confidence))
        return {"choice": choice, "confidence": confidence, "has_confidence": bool(conf_str),
                "raw_choice": raw_choice}

    # Fallback: try original A/B pattern (model might ignore label change)
    match = re.search(r'\[\[(A|B|C),?\s*([\d.]*)\]\]', text)
    if match:
        choice = match.group(1)
        conf_str = match.group(2)
        confidence = float(conf_str) if conf_str else 0.8
        confidence = max(0.5, min(1.0, confidence))
        return {"choice": choice, "confidence": confidence, "has_confidence": bool(conf_str),
                "raw_choice": choice, "used_fallback_ab": True}

    return {"choice": "PARSE_FAIL", "confidence": 0.5, "has_confidence": False, "raw_choice": "PARSE_FAIL"}


def batch_generate(model, tokenizer, prompts, max_new_tokens=512, temperature=0.1, disable_thinking=False):
    texts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if disable_thinking:
            template_kwargs["enable_thinking"] = False
        try:
            text = tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(text)

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

    responses = []
    input_len = inputs['input_ids'].shape[1]
    for i in range(len(prompts)):
        new_tokens = outputs[i][input_len:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        responses.append(response)
    return responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--swap_data", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variant", required=True, choices=list(LABEL_VARIANTS.keys()))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--disable_thinking", action="store_true", help="Disable thinking for Qwen3")
    args = parser.parse_args()

    variant_config = LABEL_VARIANTS[args.variant]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    logger.info(f"Label variant: {args.variant} — {variant_config['desc']}")

    # Auto-detect swap data
    if args.swap_data is None:
        test_path = Path(args.test_data)
        swap_path = test_path.parent / test_path.name.replace("test.", "test_swap.")
        if swap_path.exists():
            args.swap_data = str(swap_path)

    # Load data
    with open(args.test_data) as f:
        test_data = json.load(f)
    with open(args.swap_data) as f:
        swap_data = json.load(f)

    assert len(test_data) == len(swap_data)
    logger.info(f"Test samples: {len(test_data)}")

    # Transform prompts with variant labels
    for item in test_data:
        item["prompt"] = transform_prompt(item["prompt"], variant_config)
    for item in swap_data:
        item["prompt"] = transform_prompt(item["prompt"], variant_config)

    # Log sample transformed prompt
    logger.info(f"Sample transformed prompt (first 300 chars):\n{test_data[0]['prompt'][:300]}")

    # Load model
    logger.info("Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()
    model.eval()
    logger.info(f"Model loaded in {time.time()-t0:.1f}s")

    # Run evaluation
    results = []
    total = len(test_data)
    start_time = time.time()
    fallback_count = 0

    for batch_start in range(0, total, args.batch_size):
        batch_end = min(batch_start + args.batch_size, total)
        batch_items = test_data[batch_start:batch_end]
        batch_swap = swap_data[batch_start:batch_end]

        orig_prompts = [item["prompt"] for item in batch_items]
        orig_responses = batch_generate(model, tokenizer, orig_prompts,
                                         max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                         disable_thinking=args.disable_thinking)

        swap_prompts = [item["prompt"] for item in batch_swap]
        swap_responses = batch_generate(model, tokenizer, swap_prompts,
                                         max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                         disable_thinking=args.disable_thinking)

        for item, swap_item, resp_orig, resp_swap in zip(batch_items, batch_swap, orig_responses, swap_responses):
            parsed_orig = parse_judge_output(resp_orig, variant_config)
            parsed_swap = parse_judge_output(resp_swap, variant_config)

            if parsed_orig.get("used_fallback_ab"):
                fallback_count += 1

            gold = item["gold_label"]  # Still A/B (mapped back)
            is_correct = parsed_orig["choice"] == gold

            flip_map = {"A": "B", "B": "A", "C": "C", "PARSE_FAIL": "PARSE_FAIL"}
            expected_swap = flip_map.get(parsed_orig["choice"], "PARSE_FAIL")
            is_consistent = parsed_swap["choice"] == expected_swap

            conf = parsed_orig["confidence"]
            brier = (conf - (1.0 if is_correct else 0.0)) ** 2

            results.append({
                "id": item.get("original_id", batch_start),
                "category": item.get("category", "unknown"),
                "gold_label": gold,
                "predicted": parsed_orig["choice"],
                "raw_predicted": parsed_orig["raw_choice"],
                "swap_predicted": parsed_swap["choice"],
                "raw_swap_predicted": parsed_swap["raw_choice"],
                "is_correct": is_correct,
                "is_consistent": is_consistent,
                "brier_score": brier,
                "response_orig": resp_orig[:500],
                "response_swap": resp_swap[:500],
            })

        # Progress
        done = len(results)
        elapsed = time.time() - start_time
        acc = sum(1 for r in results if r["is_correct"]) / done
        con = sum(1 for r in results if r["is_consistent"]) / done
        logger.info(f"[{done}/{total}] Acc={acc:.3f} Consist={con:.3f} Fallbacks={fallback_count}")

    # Save results
    with open(output_dir / "eval_results.json", 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Compute metrics
    total = len(results)
    acc = sum(1 for r in results if r["is_correct"]) / total
    con = sum(1 for r in results if r["is_consistent"]) / total
    parse_fails = sum(1 for r in results if r["predicted"] == "PARSE_FAIL")

    # A-selection rate (position 1/Left selection)
    a_rate = sum(1 for r in results if r["predicted"] == "A") / total

    metrics = {
        "variant": args.variant,
        "variant_desc": variant_config["desc"],
        "n_samples": total,
        "accuracy": acc,
        "consistency": con,
        "a_selection_rate": a_rate,
        "parse_failures": parse_fails,
        "fallback_to_ab": fallback_count,
        "model": args.model_path,
        "adapter": args.adapter_path,
    }

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"RESULTS — Label variant: {args.variant}")
    logger.info(f"Accuracy:    {acc:.4f}")
    logger.info(f"Consistency: {con:.4f}")
    logger.info(f"A-selection: {a_rate:.4f}")
    logger.info(f"Parse fails: {parse_fails}")
    logger.info(f"Fallback to A/B parsing: {fallback_count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
