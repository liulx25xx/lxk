"""
Post-hoc Position Swap Majority Voting for Judge Evaluation.

Takes existing eval_results.json (which already has original + swap predictions),
applies position-swap majority voting as a post-hoc consistency fix, and reports
new accuracy and consistency metrics.

Logic:
- Original prediction: model sees [resp_A, resp_B] → picks X
- Swap prediction: model sees [resp_B, resp_A] → picks Y
- To compare: convert swap prediction to original-position terms (A↔B, C↔C)
- Majority vote: if both agree → use that. If disagree → use "C" (tie/abstain)
  OR: if disagree → use original (optimistic) / random (fair)

This script computes ALL three tiebreaker strategies.
"""

import json
import sys
from pathlib import Path
from collections import Counter


def swap_label(label):
    """Convert swap-position label to original-position terms."""
    if label == "A":
        return "B"
    elif label == "B":
        return "A"
    else:  # C (tie) or PARSE_FAIL
        return label


def compute_metrics_from_decisions(results, decisions, name=""):
    """Compute accuracy and consistency given final decisions."""
    total = len(results)
    correct = 0
    consistent = 0
    parse_fails = 0
    
    for r, decision in zip(results, decisions):
        if decision == "PARSE_FAIL":
            parse_fails += 1
            continue
        
        # Accuracy: does decision match gold?
        if decision == r["gold_label"]:
            correct += 1
        
        # Consistency: we define it as "decision is position-invariant"
        # Since we're combining both positions, the result IS position-invariant by construction
        # But we can check if orig and swap agreed (true consistency)
        orig = r["predicted"]
        swap_converted = swap_label(r["swap_predicted"])
        if orig == swap_converted:
            consistent += 1
    
    valid = total - parse_fails
    acc = correct / valid if valid > 0 else 0
    consist = consistent / total if total > 0 else 0
    
    # Per-category
    categories = {}
    for r, decision in zip(results, decisions):
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"correct": 0, "consistent": 0, "total": 0}
        categories[cat]["total"] += 1
        if decision == r["gold_label"]:
            categories[cat]["correct"] += 1
        orig = r["predicted"]
        swap_converted = swap_label(r["swap_predicted"])
        if orig == swap_converted:
            categories[cat]["consistent"] += 1
    
    return {
        "name": name,
        "accuracy": acc,
        "consistency": consist,
        "n_samples": total,
        "n_valid": valid,
        "parse_failures": parse_fails,
        "per_category": {
            cat: {
                "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0,
                "consistency": v["consistent"] / v["total"] if v["total"] > 0 else 0,
                "n": v["total"]
            }
            for cat, v in sorted(categories.items())
        }
    }


def main():
    # Can analyze baseline or any model's eval_results.json
    results_dir = Path("/path/to/paper3_judge_rl/results")
    
    # Analyze baseline
    baseline_path = results_dir / "baseline_qwen7b" / "eval_results.json"
    if not baseline_path.exists():
        print(f"ERROR: {baseline_path} not found")
        sys.exit(1)
    
    with open(baseline_path) as f:
        results = json.load(f)
    
    print(f"Loaded {len(results)} results from baseline")
    print(f"Original metrics: accuracy={sum(1 for r in results if r['is_correct'])/len(results):.4f}, "
          f"consistency={sum(1 for r in results if r['is_consistent'])/len(results):.4f}")
    print()
    
    # Strategy 1: Majority vote — agree → use, disagree → "C" (tie/abstain)
    decisions_abstain = []
    for r in results:
        orig = r["predicted"]
        swap_conv = swap_label(r["swap_predicted"])
        if orig == swap_conv:
            decisions_abstain.append(orig)
        else:
            decisions_abstain.append("C")  # abstain = tie
    
    # Strategy 2: Majority vote — agree → use, disagree → use original
    decisions_orig = []
    for r in results:
        orig = r["predicted"]
        swap_conv = swap_label(r["swap_predicted"])
        if orig == swap_conv:
            decisions_orig.append(orig)
        else:
            decisions_orig.append(orig)  # fallback to original
    
    # Strategy 3: Majority vote — agree → use, disagree → use swap
    decisions_swap = []
    for r in results:
        orig = r["predicted"]
        swap_conv = swap_label(r["swap_predicted"])
        if orig == swap_conv:
            decisions_swap.append(swap_conv)
        else:
            decisions_swap.append(swap_conv)  # fallback to swap
    
    # Strategy 4: Always use original (= standard single-pass, no post-hoc fix)
    decisions_standard = [r["predicted"] for r in results]
    
    # Compute all metrics
    metrics_abstain = compute_metrics_from_decisions(results, decisions_abstain, "majority_vote_abstain")
    metrics_orig = compute_metrics_from_decisions(results, decisions_orig, "majority_vote_use_orig")
    metrics_swap = compute_metrics_from_decisions(results, decisions_swap, "majority_vote_use_swap")
    metrics_standard = compute_metrics_from_decisions(results, decisions_standard, "standard_single_pass")
    
    # Count agreement/disagreement
    n_agree = sum(1 for r in results if r["predicted"] == swap_label(r["swap_predicted"]))
    n_disagree = len(results) - n_agree
    
    print(f"Position agreement: {n_agree}/{len(results)} ({n_agree/len(results)*100:.1f}%)")
    print(f"Position disagreement: {n_disagree}/{len(results)} ({n_disagree/len(results)*100:.1f}%)")
    print()
    
    print("=" * 70)
    print(f"{'Strategy':<30} {'Accuracy':>10} {'Consistency':>12}")
    print("=" * 70)
    for m in [metrics_standard, metrics_abstain, metrics_orig, metrics_swap]:
        print(f"{m['name']:<30} {m['accuracy']*100:>9.1f}% {m['consistency']*100:>11.1f}%")
    print("=" * 70)
    print()
    
    # Also analyze: when they disagree, which is right?
    orig_right_when_disagree = 0
    swap_right_when_disagree = 0
    neither_right = 0
    for r in results:
        orig = r["predicted"]
        swap_conv = swap_label(r["swap_predicted"])
        if orig != swap_conv:
            gold = r["gold_label"]
            if orig == gold:
                orig_right_when_disagree += 1
            elif swap_conv == gold:
                swap_right_when_disagree += 1
            else:
                neither_right += 1
    
    print(f"When positions DISAGREE ({n_disagree} cases):")
    print(f"  Original correct: {orig_right_when_disagree} ({orig_right_when_disagree/max(n_disagree,1)*100:.1f}%)")
    print(f"  Swap correct: {swap_right_when_disagree} ({swap_right_when_disagree/max(n_disagree,1)*100:.1f}%)")
    print(f"  Neither correct: {neither_right} ({neither_right/max(n_disagree,1)*100:.1f}%)")
    print()
    
    # Now also analyze RL-trained models if available
    rl_models = [
        "EXP-006_accuracy_only",
        "EXP-009_full_composite",
    ]
    
    for model_name in rl_models:
        eval_path = results_dir / model_name / "eval" / "eval_results.json"
        if not eval_path.exists():
            print(f"  {model_name}: eval_results.json not found, skipping")
            continue
        
        with open(eval_path) as f:
            rl_results = json.load(f)
        
        n_agree_rl = sum(1 for r in rl_results if r["predicted"] == swap_label(r["swap_predicted"]))
        n_disagree_rl = len(rl_results) - n_agree_rl
        
        # Majority vote with abstain
        decisions_rl = []
        for r in rl_results:
            orig = r["predicted"]
            swap_conv = swap_label(r["swap_predicted"])
            if orig == swap_conv:
                decisions_rl.append(orig)
            else:
                decisions_rl.append("C")
        
        metrics_rl_std = compute_metrics_from_decisions(rl_results, [r["predicted"] for r in rl_results], f"{model_name}_standard")
        metrics_rl_mv = compute_metrics_from_decisions(rl_results, decisions_rl, f"{model_name}_majority_vote")
        
        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print(f"Position agreement: {n_agree_rl}/{len(rl_results)} ({n_agree_rl/len(rl_results)*100:.1f}%)")
        print(f"{'Strategy':<30} {'Accuracy':>10} {'Consistency':>12}")
        print(f"{metrics_rl_std['name']:<30} {metrics_rl_std['accuracy']*100:>9.1f}% {metrics_rl_std['consistency']*100:>11.1f}%")
        print(f"{metrics_rl_mv['name']:<30} {metrics_rl_mv['accuracy']*100:>9.1f}% {metrics_rl_mv['consistency']*100:>11.1f}%")
    
    # Save full results
    output = {
        "baseline": {
            "n_samples": len(results),
            "n_agree": n_agree,
            "n_disagree": n_disagree,
            "strategies": {
                "standard": {"accuracy": metrics_standard["accuracy"], "consistency": metrics_standard["consistency"]},
                "majority_vote_abstain": {"accuracy": metrics_abstain["accuracy"], "consistency": metrics_abstain["consistency"]},
                "majority_vote_use_orig": {"accuracy": metrics_orig["accuracy"], "consistency": metrics_orig["consistency"]},
                "majority_vote_use_swap": {"accuracy": metrics_swap["accuracy"], "consistency": metrics_swap["consistency"]},
            },
            "disagreement_analysis": {
                "orig_correct": orig_right_when_disagree,
                "swap_correct": swap_right_when_disagree,
                "neither_correct": neither_right,
            }
        }
    }
    
    output_path = results_dir / "post_hoc_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved analysis to {output_path}")


if __name__ == "__main__":
    main()
