"""
Analyze length preference in existing eval results (zero GPU).

For each eval prediction, check if the model chose the longer response.
Compare baseline vs RL-trained models to see if RL amplifies length preference.

This parallels the position bias analysis: just as pred_A_rate tracks position bias,
chose_longer_rate tracks length bias.

Usage:
    python analyze_length_preference.py
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def extract_response_lengths(prompt):
    """Extract A and B response lengths from the judge prompt."""
    a_match = re.search(
        r"\[The Start of Assistant A's Answer\]\s*(.*?)\s*\[The End of Assistant A's Answer\]",
        prompt, re.DOTALL
    )
    b_match = re.search(
        r"\[The Start of Assistant B's Answer\]\s*(.*?)\s*\[The End of Assistant B's Answer\]",
        prompt, re.DOTALL
    )
    
    if not a_match or not b_match:
        return None, None
    
    return len(a_match.group(1).strip()), len(b_match.group(1).strip())


def analyze_experiment(eval_results, test_data, name):
    """Analyze length preference for one experiment's eval results."""
    # Build ID -> test item lookup
    id_to_test = {}
    for item in test_data:
        oid = item.get("original_id", None)
        if oid is not None:
            id_to_test[oid] = item
    
    total = 0
    chose_longer = 0
    chose_shorter = 0
    chose_equal = 0
    parse_fail = 0
    
    # Also track: when gold=longer vs gold=shorter
    gold_is_longer = 0
    gold_is_shorter = 0
    correct_when_gold_longer = 0
    correct_when_gold_shorter = 0
    
    for result in eval_results:
        rid = result["id"]
        predicted = result["predicted"]
        gold = result["gold_label"]
        
        if predicted == "PARSE_FAIL":
            parse_fail += 1
            continue
        
        # Find corresponding test item to get the prompt
        test_item = id_to_test.get(rid)
        if test_item is None:
            continue
        
        len_a, len_b = extract_response_lengths(test_item["prompt"])
        if len_a is None:
            continue
        
        total += 1
        
        # Did model choose the longer response?
        if predicted == "A":
            model_chose_longer = len_a > len_b
            model_chose_shorter = len_b > len_a
        elif predicted == "B":
            model_chose_longer = len_b > len_a
            model_chose_shorter = len_a > len_b
        else:  # C (tie)
            chose_equal += 1
            continue
        
        if len_a == len_b:
            chose_equal += 1
            continue
        
        if model_chose_longer:
            chose_longer += 1
        else:
            chose_shorter += 1
        
        # Track gold-length interaction
        if gold == "A":
            gold_longer = len_a > len_b
        else:
            gold_longer = len_b > len_a
        
        if len_a != len_b:
            if gold_longer:
                gold_is_longer += 1
                if result["is_correct"]:
                    correct_when_gold_longer += 1
            else:
                gold_is_shorter += 1
                if result["is_correct"]:
                    correct_when_gold_shorter += 1
    
    decidable = chose_longer + chose_shorter
    chose_longer_rate = chose_longer / decidable if decidable > 0 else 0
    
    # Accuracy conditioned on gold being longer vs shorter
    acc_gold_longer = correct_when_gold_longer / gold_is_longer if gold_is_longer > 0 else 0
    acc_gold_shorter = correct_when_gold_shorter / gold_is_shorter if gold_is_shorter > 0 else 0
    
    return {
        "name": name,
        "total_decidable": decidable,
        "chose_longer": chose_longer,
        "chose_shorter": chose_shorter,
        "chose_longer_rate": chose_longer_rate,
        "chose_equal_or_tie": chose_equal,
        "parse_fail": parse_fail,
        "gold_is_longer": gold_is_longer,
        "gold_is_shorter": gold_is_shorter,
        "acc_when_gold_longer": acc_gold_longer,
        "acc_when_gold_shorter": acc_gold_shorter,
        "acc_gap": acc_gold_longer - acc_gold_shorter,
    }


def main():
    # Load test data (need prompts for length extraction)
    test_path = PROJECT_ROOT / "data/eval/rewardbench_test.json"
    print(f"Loading test data from {test_path}")
    with open(test_path) as f:
        test_data = json.load(f)
    print(f"Test samples: {len(test_data)}")
    
    # Find all experiments with eval results
    experiments = []
    
    # Baseline
    baseline_dirs = [
        ("Baseline (Qwen2.5-7B)", RESULTS_DIR / "baseline_qwen7b"),
        ("Baseline (244)", RESULTS_DIR / "baseline_qwen7b_244"),
    ]
    
    # RL experiments (unbalanced)
    rl_dirs = [
        ("EXP-006 acc-only", RESULTS_DIR / "EXP-006_accuracy_only" / "eval"),
        ("EXP-006 acc s2", RESULTS_DIR / "EXP-006_accuracy_s2" / "eval"),
        ("EXP-006 acc s3", RESULTS_DIR / "EXP-006_accuracy_s3" / "eval"),
        ("EXP-007a decisive", RESULTS_DIR / "EXP-007a_acc_decisive" / "eval"),
        ("EXP-007a decisive s2", RESULTS_DIR / "EXP-007a_acc_decisive_s2" / "eval"),
        ("EXP-008 calib", RESULTS_DIR / "EXP-008_acc_calib" / "eval"),
        ("EXP-008 calib s2", RESULTS_DIR / "EXP-008_acc_calib_s2" / "eval"),
        ("EXP-009 full", RESULTS_DIR / "EXP-009_full_composite" / "eval"),
        ("EXP-009 full s2", RESULTS_DIR / "EXP-009_full_composite_s2" / "eval"),
        ("EXP-009 full s3", RESULTS_DIR / "EXP-009_full_composite_s3" / "eval"),
        ("EXP-009 full s4", RESULTS_DIR / "EXP-009_full_s4" / "eval"),
        ("EXP-009 lr=1e-5", RESULTS_DIR / "EXP-009_full_lr1e5" / "eval"),
        ("EXP-009 lr=1e-6", RESULTS_DIR / "EXP-009_full_lr1e6" / "eval"),
    ]
    
    # Balanced experiments
    balanced_dirs = [
        ("EXP-006b balanced", RESULTS_DIR / "EXP-006b_accuracy_balanced" / "eval"),
        ("EXP-006b balanced s2", RESULTS_DIR / "EXP-006b_accuracy_balanced_s2" / "eval"),
        ("EXP-007b balanced", RESULTS_DIR / "EXP-007b_decisive_balanced" / "eval"),
        ("EXP-008b balanced", RESULTS_DIR / "EXP-008b_calib_balanced" / "eval"),
        ("EXP-009b balanced", RESULTS_DIR / "EXP-009b_full_balanced" / "eval"),
        ("EXP-009b balanced s2", RESULTS_DIR / "EXP-009b_full_balanced_s2" / "eval"),
        ("EXP-009b balanced s3", RESULTS_DIR / "EXP-009b_full_balanced_s3" / "eval"),
    ]
    
    # Length-confounded experiments
    length_dirs = [
        ("EXP-LENGTH confounded", RESULTS_DIR / "EXP-LENGTH_confounded" / "eval"),
    ]
    
    all_dirs = baseline_dirs + rl_dirs + balanced_dirs
    
    results_all = []
    
    for name, eval_dir in all_dirs:
        results_file = eval_dir / "eval_results.json"
        if not results_file.exists():
            continue
        
        with open(results_file) as f:
            eval_results = json.load(f)
        
        analysis = analyze_experiment(eval_results, test_data, name)
        results_all.append(analysis)
    
    # Print results
    print(f"\n{'='*90}")
    print(f"LENGTH PREFERENCE ANALYSIS")
    print(f"{'='*90}")
    print(f"{'Experiment':<30s} {'Chose_Longer%':>14s} {'N':>5s} {'Acc(gold=long)':>15s} {'Acc(gold=short)':>16s} {'Gap':>8s}")
    print(f"{'-'*90}")
    
    for r in results_all:
        print(f"{r['name']:<30s} {r['chose_longer_rate']*100:>13.1f}% {r['total_decidable']:>5d} "
              f"{r['acc_when_gold_longer']*100:>14.1f}% {r['acc_when_gold_shorter']*100:>15.1f}% "
              f"{r['acc_gap']*100:>7.1f}pp")
    
    print(f"{'='*90}")
    
    # Key insight: does RL amplify length preference?
    baselines = [r for r in results_all if "Baseline" in r["name"]]
    unbalanced_rl = [r for r in results_all if r["name"].startswith("EXP-00") and "balanced" not in r["name"]]
    balanced_rl = [r for r in results_all if "balanced" in r["name"]]
    
    if baselines:
        avg_bl = sum(r["chose_longer_rate"] for r in baselines) / len(baselines)
        print(f"\nBaseline avg chose_longer_rate: {avg_bl*100:.1f}%")
    
    if unbalanced_rl:
        avg_rl = sum(r["chose_longer_rate"] for r in unbalanced_rl) / len(unbalanced_rl)
        print(f"Unbalanced RL avg chose_longer_rate: {avg_rl*100:.1f}%")
    
    if balanced_rl:
        avg_bal = sum(r["chose_longer_rate"] for r in balanced_rl) / len(balanced_rl)
        print(f"Balanced RL avg chose_longer_rate: {avg_bal*100:.1f}%")
    
    # Compute correlation between accuracy and chose_longer_rate across RL models
    if len(unbalanced_rl) >= 2:
        # Load accuracy from metrics.json for correlation
        print(f"\n--- Accuracy vs Chose_Longer_Rate (Unbalanced RL models) ---")
        for r in unbalanced_rl:
            print(f"  {r['name']}: chose_longer={r['chose_longer_rate']*100:.1f}%, acc_gap={r['acc_gap']*100:.1f}pp")
    
    # Save full analysis
    output_path = RESULTS_DIR / "length_preference_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(results_all, f, indent=2)
    print(f"\nFull analysis saved to {output_path}")


if __name__ == "__main__":
    main()
