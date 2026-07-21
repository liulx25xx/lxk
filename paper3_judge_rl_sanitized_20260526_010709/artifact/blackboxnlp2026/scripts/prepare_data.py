"""
Paper 3: 准备 Judge RL 训练数据

步骤:
1. 下载 RewardBench dataset
2. 构造 judge prompt (pairwise comparison format)
3. 生成 position-swapped pairs (for paired evaluation and balanced training)
4. 划分 train/val/test

Usage:
    python prepare_data.py

Output:
    ../data/train/judge_train.json     — GRPO 训练数据
    ../data/train/judge_swap.json      — position-swapped 版本
    ../data/eval/rewardbench_test.json  — 评估数据
"""

import json
import os
import sys
from pathlib import Path

from datasets import load_dataset
import random

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
EVAL_DIR = DATA_DIR / "eval"

JUDGE_PROMPT_TEMPLATE = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses.

Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.

After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" if it is a tie or you cannot determine a winner.

[User Question]
{question}

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]"""


def build_judge_instance(item, swap=False):
    """Build a judge prompt instance from a RewardBench item."""
    question = item.get("prompt", item.get("instruction", ""))
    chosen = item.get("chosen", "")
    rejected = item.get("rejected", "")
    
    if swap:
        answer_a, answer_b = rejected, chosen
        gold_label = "B"  # chosen is now B
    else:
        answer_a, answer_b = chosen, rejected
        gold_label = "A"  # chosen is A
    
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
    )
    
    return {
        "prompt": prompt,
        "gold_label": gold_label,  # A or B
        "swapped": swap,
        "original_id": item.get("id", ""),
        "category": item.get("subset", item.get("category", "unknown")),
    }


def main():
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading RewardBench dataset...")
    try:
        ds = load_dataset("allenai/reward-bench", split="filtered")
    except Exception as e:
        print(f"Failed to load 'filtered' split: {e}")
        print("Trying 'raw' split...")
        try:
            ds = load_dataset("allenai/reward-bench", split="raw")
        except Exception as e2:
            print(f"Also failed: {e2}")
            print("Please download manually or check dataset name.")
            sys.exit(1)
    
    print(f"Loaded {len(ds)} instances")
    
    # Shuffle and split
    indices = list(range(len(ds)))
    random.seed(42)
    random.shuffle(indices)
    
    # 70% train, 15% val, 15% test
    n_train = int(0.7 * len(indices))
    n_val = int(0.15 * len(indices))
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    
    print(f"Split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    
    # Build training data (original + swapped for consistency)
    train_data = []
    swap_data = []
    for idx in train_indices:
        item = ds[idx]
        train_data.append(build_judge_instance(item, swap=False))
        swap_data.append(build_judge_instance(item, swap=True))
    
    # Build eval data (original + swapped for consistency testing)
    eval_data = []
    eval_swap_data = []
    for idx in test_indices:
        item = ds[idx]
        eval_data.append(build_judge_instance(item, swap=False))
        eval_swap_data.append(build_judge_instance(item, swap=True))
    
    # Save
    train_path = TRAIN_DIR / "judge_train.json"
    swap_path = TRAIN_DIR / "judge_swap.json"
    eval_path = EVAL_DIR / "rewardbench_test.json"
    eval_swap_path = EVAL_DIR / "rewardbench_test_swap.json"
    
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open(swap_path, 'w') as f:
        json.dump(swap_data, f, indent=2, ensure_ascii=False)
    with open(eval_path, 'w') as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    with open(eval_swap_path, 'w') as f:
        json.dump(eval_swap_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved:")
    print(f"  Train:     {train_path} ({len(train_data)} instances)")
    print(f"  Swap:      {swap_path} ({len(swap_data)} instances)")
    print(f"  Eval:      {eval_path} ({len(eval_data)} instances)")
    print(f"  Eval Swap: {eval_swap_path} ({len(eval_swap_data)} instances)")
    
    # Stats
    categories = {}
    for item in train_data:
        cat = item["category"]
        categories[cat] = categories.get(cat, 0) + 1
    print(f"\nCategory distribution (train):")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
