"""
Download and format OOD (out-of-distribution) test datasets for Paper 5.

Datasets:
  - Math:        MATH-500 (hendrycks MATH, competition-level)
  - Science:     ARC-Challenge test set (different from ScienceQA train domain)
  - Medicine:    MMLU medical subsets (anatomy, clinical_knowledge, college_medicine,
                 medical_genetics, professional_medicine)
  - Law:         MMLU law subsets (professional_law, international_law, jurisprudence)
  - Commonsense: WinoGrande validation set

Output: data/ood/{domain}_ood.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Ensure HF cache (must set all three to avoid writing to /data_train/.cache/)
os.environ.setdefault("HF_HOME", "/path/to/workspace/cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/path/to/workspace/cache/huggingface/hub")
os.environ.setdefault("HF_DATASETS_CACHE", "/path/to/workspace/cache/huggingface/datasets")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OOD_DIR = PROJECT_ROOT / "data" / "ood"


def format_prompt_math(question: str) -> str:
    """Format math question into chat prompt."""
    return (
        f"<|im_start|>user\n{question.strip()}\n\n"
        f"Please solve this problem step by step and provide your final answer."
        f"<|im_end|>\n<|im_start|>assistant\n"
    )


def format_prompt_mcq(question: str, choices: list[str], choice_labels: list[str] | None = None) -> str:
    """Format MCQ question into chat prompt."""
    if choice_labels is None:
        choice_labels = [chr(65 + i) for i in range(len(choices))]
    choices_text = "\n".join(f"{label}. {c}" for label, c in zip(choice_labels, choices))
    return (
        f"<|im_start|>user\n{question.strip()}\n\n{choices_text}\n\n"
        f"Please think step by step and provide your answer."
        f"<|im_end|>\n<|im_start|>assistant\n"
    )


def format_prompt_winogrande(sentence: str, option1: str, option2: str) -> str:
    """Format WinoGrande question into MCQ prompt."""
    return format_prompt_mcq(
        f"Fill in the blank with the correct option:\n{sentence}",
        [option1, option2],
        ["A", "B"],
    )


# ---------------------------------------------------------------------------
# MATH-500
# ---------------------------------------------------------------------------

def _extract_boxed(text: str) -> str | None:
    """Extract answer from \\boxed{...}, handling nested braces."""
    import re
    # Find all \boxed occurrences
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    # Count braces to find matching close
    start = idx + len("\\boxed{")
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    if depth == 0:
        return text[start:pos-1]
    # Fallback: simple regex
    m = re.findall(r"\\boxed\{(.+?)\}", text)
    return m[-1] if m else None


def prepare_math_500() -> list[dict]:
    """Download MATH dataset and sample 500 problems."""
    from datasets import load_dataset

    logger.info("Loading MATH dataset (MATH-lighteval)...")
    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test")
    logger.info(f"  Loaded MATH-lighteval test: {len(ds)} problems")

    records = []
    skipped = 0
    for i, row in enumerate(ds):
        solution = row.get("solution", "")
        answer = _extract_boxed(solution)
        if not answer:
            # Fallback: last line of solution
            answer = solution.strip().split("\n")[-1] if solution.strip() else None
        if not answer:
            skipped += 1
            continue

        records.append({
            "id": f"math500_{i}",
            "prompt": format_prompt_math(row["problem"]),
            "gold_answer": answer,
            "domain": "math",
            "source": "MATH-500",
            "metadata": {
                "subdomain": row.get("type", row.get("subject", "unknown")),
                "level": row.get("level", ""),
            },
        })

    if skipped:
        logger.info(f"  Skipped {skipped} problems without extractable answer")

    # Sample 500 if more
    if len(records) > 500:
        random.seed(42)
        records = random.sample(records, 500)
        # Re-index
        for i, r in enumerate(records):
            r["id"] = f"math500_{i}"

    logger.info(f"  MATH-500: {len(records)} problems prepared")
    return records


# ---------------------------------------------------------------------------
# ARC-Challenge (Science OOD)
# ---------------------------------------------------------------------------

def prepare_arc_challenge() -> list[dict]:
    """Download ARC-Challenge test set."""
    from datasets import load_dataset

    logger.info("Loading ARC-Challenge...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    logger.info(f"  Loaded ARC-Challenge test: {len(ds)} questions")

    records = []
    for i, row in enumerate(ds):
        choices = row["choices"]
        labels = choices["label"]
        texts = choices["text"]
        gold_label = row["answerKey"]

        # Map answer key to letter index
        if gold_label in labels:
            gold_idx = labels.index(gold_label)
            gold_letter = chr(65 + gold_idx)
        else:
            gold_letter = gold_label

        records.append({
            "id": f"arc_challenge_{i}",
            "prompt": format_prompt_mcq(row["question"], texts, [chr(65 + j) for j in range(len(texts))]),
            "gold_answer": gold_letter,
            "domain": "science",
            "source": "ARC-Challenge",
            "metadata": {
                "subdomain": "arc_challenge",
                "choices": texts,
            },
        })

    logger.info(f"  ARC-Challenge: {len(records)} questions prepared")
    return records


# ---------------------------------------------------------------------------
# MMLU Medical subsets
# ---------------------------------------------------------------------------

MMLU_MED_SUBSETS = [
    "anatomy",
    "clinical_knowledge",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
]


def prepare_mmlu_med() -> list[dict]:
    """Download MMLU medical subsets."""
    from datasets import load_dataset

    logger.info("Loading MMLU medical subsets...")
    records = []

    for subset in MMLU_MED_SUBSETS:
        try:
            ds = load_dataset("cais/mmlu", subset, split="test")
            logger.info(f"  {subset}: {len(ds)} questions")
        except Exception as e:
            logger.warning(f"  Failed to load {subset}: {e}")
            continue

        for i, row in enumerate(ds):
            choices = [row["choices"][j] if isinstance(row["choices"], list) else row[f"choices"][j]
                       for j in range(4)]
            gold_idx = row["answer"]
            gold_letter = chr(65 + gold_idx)

            records.append({
                "id": f"mmlu_med_{subset}_{i}",
                "prompt": format_prompt_mcq(row["question"], choices),
                "gold_answer": gold_letter,
                "domain": "medicine",
                "source": "MMLU-Med",
                "metadata": {
                    "subdomain": subset,
                    "choices": choices,
                },
            })

    logger.info(f"  MMLU-Med total: {len(records)} questions")
    return records


# ---------------------------------------------------------------------------
# MMLU Law subsets
# ---------------------------------------------------------------------------

MMLU_LAW_SUBSETS = [
    "professional_law",
    "international_law",
    "jurisprudence",
]


def prepare_mmlu_law() -> list[dict]:
    """Download MMLU law subsets."""
    from datasets import load_dataset

    logger.info("Loading MMLU law subsets...")
    records = []

    for subset in MMLU_LAW_SUBSETS:
        try:
            ds = load_dataset("cais/mmlu", subset, split="test")
            logger.info(f"  {subset}: {len(ds)} questions")
        except Exception as e:
            logger.warning(f"  Failed to load {subset}: {e}")
            continue

        for i, row in enumerate(ds):
            choices = [row["choices"][j] for j in range(4)]
            gold_idx = row["answer"]
            gold_letter = chr(65 + gold_idx)

            records.append({
                "id": f"mmlu_law_{subset}_{i}",
                "prompt": format_prompt_mcq(row["question"], choices),
                "gold_answer": gold_letter,
                "domain": "law",
                "source": "MMLU-Law",
                "metadata": {
                    "subdomain": subset,
                    "choices": choices,
                },
            })

    logger.info(f"  MMLU-Law total: {len(records)} questions")
    return records


# ---------------------------------------------------------------------------
# WinoGrande (Commonsense OOD)
# ---------------------------------------------------------------------------

def prepare_winogrande() -> list[dict]:
    """Download WinoGrande validation set."""
    from datasets import load_dataset

    logger.info("Loading WinoGrande...")
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    logger.info(f"  Loaded WinoGrande validation: {len(ds)} questions")

    records = []
    for i, row in enumerate(ds):
        sentence = row["sentence"]
        option1 = row["option1"]
        option2 = row["option2"]
        gold = row["answer"]  # "1" or "2"
        gold_letter = "A" if gold == "1" else "B"

        records.append({
            "id": f"winogrande_{i}",
            "prompt": format_prompt_winogrande(sentence, option1, option2),
            "gold_answer": gold_letter,
            "domain": "commonsense",
            "source": "WinoGrande",
            "metadata": {
                "subdomain": "winogrande",
                "choices": [option1, option2],
            },
        })

    logger.info(f"  WinoGrande: {len(records)} questions prepared")
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PREPARERS = {
    "math": prepare_math_500,
    "science": prepare_arc_challenge,
    "medicine": prepare_mmlu_med,
    "law": prepare_mmlu_law,
    "commonsense": prepare_winogrande,
}


def main():
    parser = argparse.ArgumentParser(description="Prepare OOD evaluation datasets")
    parser.add_argument("--domains", nargs="*", default=None,
                        help="Domains to prepare (default: all)")
    parser.add_argument("--output_dir", type=str, default=str(OOD_DIR))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    domains = args.domains or list(PREPARERS.keys())

    for domain in domains:
        if domain not in PREPARERS:
            logger.warning(f"Unknown domain: {domain}, skipping")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"Preparing: {domain}")
        logger.info(f"{'='*50}")

        records = PREPARERS[domain]()

        out_path = os.path.join(args.output_dir, f"{domain}_ood.jsonl")
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        logger.info(f"  Saved {len(records)} records → {out_path}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
