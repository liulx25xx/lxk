"""
Create length-confounded training data for Paper 3.

Takes balanced training data and re-labels so gold_label always points to the LONGER response.
This creates a dataset where "longer response = correct" correlation ≈ 100%.

If model learns to "always choose longer", that proves RL amplifies ANY spurious 
correlation in training data — not just position bias.

Usage:
    python create_length_confounded.py
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def extract_response_lengths(prompt):
    """Extract the lengths of Assistant A's and Assistant B's responses from the prompt.
    
    The prompt format has:
    [The Start of Assistant A's Answer]
    ...
    [The End of Assistant A's Answer]
    
    [The Start of Assistant B's Answer]
    ...
    [The End of Assistant B's Answer]
    """
    # Extract Assistant A's response
    a_match = re.search(
        r"\[The Start of Assistant A's Answer\]\s*(.*?)\s*\[The End of Assistant A's Answer\]",
        prompt, re.DOTALL
    )
    # Extract Assistant B's response
    b_match = re.search(
        r"\[The Start of Assistant B's Answer\]\s*(.*?)\s*\[The End of Assistant B's Answer\]",
        prompt, re.DOTALL
    )
    
    if not a_match or not b_match:
        return None, None
    
    a_text = a_match.group(1).strip()
    b_text = b_match.group(1).strip()
    
    return len(a_text), len(b_text)


def main():
    balanced_path = PROJECT_ROOT / "data/train/judge_train_balanced.json"
    output_path = PROJECT_ROOT / "data/train/judge_train_length_confounded.json"
    
    print(f"Loading balanced data from {balanced_path}")
    with open(balanced_path) as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    confounded = []
    stats = {
        "total": 0,
        "a_longer": 0,
        "b_longer": 0,
        "equal": 0,
        "parse_fail": 0,
        "label_flipped": 0,
        "label_kept": 0,
    }
    
    for item in data:
        stats["total"] += 1
        len_a, len_b = extract_response_lengths(item["prompt"])
        
        if len_a is None or len_b is None:
            stats["parse_fail"] += 1
            continue
        
        # Determine which response is longer
        if len_a > len_b:
            new_label = "A"
            stats["a_longer"] += 1
        elif len_b > len_a:
            new_label = "B"
            stats["b_longer"] += 1
        else:
            # Equal length — skip (ambiguous for length shortcut)
            stats["equal"] += 1
            continue
        
        # Track if we changed the label
        if new_label != item["gold_label"]:
            stats["label_flipped"] += 1
        else:
            stats["label_kept"] += 1
        
        new_item = {
            "prompt": item["prompt"],
            "gold_label": new_label,
            "swapped": item.get("swapped", False),
            "original_id": item.get("original_id", stats["total"]),
            "category": item.get("category", "unknown"),
            "len_a": len_a,
            "len_b": len_b,
            "length_ratio": max(len_a, len_b) / max(min(len_a, len_b), 1),
        }
        confounded.append(new_item)
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(confounded, f, indent=2, ensure_ascii=False)
    
    # Report statistics
    print(f"\n=== Length-Confounded Data Statistics ===")
    print(f"Total input samples: {stats['total']}")
    print(f"Parse failures: {stats['parse_fail']}")
    print(f"Equal length (skipped): {stats['equal']}")
    print(f"A longer: {stats['a_longer']}")
    print(f"B longer: {stats['b_longer']}")
    print(f"Output samples: {len(confounded)}")
    print(f"Labels kept (already pointed to longer): {stats['label_kept']}")
    print(f"Labels flipped (now point to longer): {stats['label_flipped']}")
    
    if confounded:
        a_gold = sum(1 for x in confounded if x["gold_label"] == "A")
        b_gold = sum(1 for x in confounded if x["gold_label"] == "B")
        print(f"\nGold label distribution: A={a_gold} ({a_gold/len(confounded)*100:.1f}%), B={b_gold} ({b_gold/len(confounded)*100:.1f}%)")
        
        ratios = [x["length_ratio"] for x in confounded]
        print(f"Length ratio stats: min={min(ratios):.2f}, median={sorted(ratios)[len(ratios)//2]:.2f}, max={max(ratios):.2f}, mean={sum(ratios)/len(ratios):.2f}")
    
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
