#!/usr/bin/env python3
"""
P0: Ablation analysis - removing "relevant" dimension from scoring.
Shows that main conclusions hold without the type-specific regex dimension.
"""
import json
import numpy as np
from collections import defaultdict

def load_json(path):
    with open(path) as f:
        return json.load(f)

def score_without_relevant(eval_data):
    """Compute 0-2 score using only file_hit + actionable."""
    return int(eval_data["file_hit"]) + int(eval_data["actionable"])

# ============================================================
# Phase 3: Main scaffold results (Table 3 ablation)
# ============================================================
print("=" * 70)
print("PHASE 3: TABLE 3 ABLATION (removing 'relevant' dimension)")
print("=" * 70)

phase3 = load_json("/data/home/xiankunlin/project/emnlp/paper1/results/phase3_full_scaffold/full_results.json")

# Group by type and strategy
type_strategy_scores = defaultdict(lambda: defaultdict(list))
type_strategy_scores_no_rel = defaultdict(lambda: defaultdict(list))

for r in phase3["results"]:
    ft = r["failure_type"]
    strat = r["strategy"]
    type_strategy_scores[ft][strat].append(r["eval"]["score"])
    type_strategy_scores_no_rel[ft][strat].append(score_without_relevant(r["eval"]))

# Control strategies - all types use same control name
control_map = {
    "LOC": "CONTROL_no_scaffold",
    "EDIT": "CONTROL_no_scaffold", 
    "LOGIC": "CONTROL_no_scaffold",
    "PLAN": "CONTROL_no_scaffold"
}

# Best strategy per type (from actual data)
best_map = {
    "EDIT": "EDIT_A_reread_file",
    "PLAN": "PLAN_A_step_back",
    "LOC": "LOC_B_reread_issue",
    "LOGIC": "LOGIC_B_minimal_fix"
}

# Worst strategy for LOC (the key mismatch claim)
worst_loc = "LOC_C_test_guided"

print("\n{:<8} {:<25} {:>8} {:>8} {:>8} {:>8}".format(
    "Type", "Strategy", "Orig", "NoRel", "Ctrl_O", "Ctrl_NR"))
print("-" * 70)

for ft in ["EDIT", "PLAN", "LOC", "LOGIC"]:
    best_strat = best_map[ft]
    ctrl_strat = control_map[ft]
    
    # Original scores
    best_orig = np.mean(type_strategy_scores[ft][best_strat])
    ctrl_orig = np.mean(type_strategy_scores[ft][ctrl_strat])
    
    # Without relevant
    best_no_rel = np.mean(type_strategy_scores_no_rel[ft][best_strat])
    ctrl_no_rel = np.mean(type_strategy_scores_no_rel[ft][ctrl_strat])
    
    delta_orig = best_orig - ctrl_orig
    delta_no_rel = best_no_rel - ctrl_no_rel
    
    print("{:<8} {:<25} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f}".format(
        ft, best_strat.split("_", 2)[-1][:20], best_orig, best_no_rel, ctrl_orig, ctrl_no_rel))

print("\n\n--- DELTA COMPARISON (Best - Control) ---")
print("{:<8} {:>12} {:>12} {:>12}".format("Type", "Δ_orig", "Δ_no_rel", "Holds?"))
print("-" * 50)

for ft in ["EDIT", "PLAN", "LOC", "LOGIC"]:
    best_strat = best_map[ft]
    ctrl_strat = control_map[ft]
    
    best_orig = np.mean(type_strategy_scores[ft][best_strat])
    ctrl_orig = np.mean(type_strategy_scores[ft][ctrl_strat])
    best_no_rel = np.mean(type_strategy_scores_no_rel[ft][best_strat])
    ctrl_no_rel = np.mean(type_strategy_scores_no_rel[ft][ctrl_strat])
    
    delta_orig = best_orig - ctrl_orig
    delta_no_rel = best_no_rel - ctrl_no_rel
    
    holds = "YES" if (delta_no_rel > 0 and delta_orig > 0) or (delta_no_rel <= 0 and delta_orig <= 0) else "CHANGED"
    
    print("{:<8} {:>12.2f} {:>12.2f} {:>12}".format(ft, delta_orig, delta_no_rel, holds))

# Key claim: LOC mismatch still harms
print("\n\n--- LOC MISMATCH CLAIM (test_guided vs control) ---")
if worst_loc in type_strategy_scores["LOC"]:
    worst_orig = np.mean(type_strategy_scores["LOC"][worst_loc])
    worst_no_rel = np.mean(type_strategy_scores_no_rel["LOC"][worst_loc])
    ctrl_orig = np.mean(type_strategy_scores["LOC"][control_map["LOC"]])
    ctrl_no_rel = np.mean(type_strategy_scores_no_rel["LOC"][control_map["LOC"]])
    
    print(f"  test_guided: orig={worst_orig:.2f}, no_rel={worst_no_rel:.2f}")
    print(f"  control:     orig={ctrl_orig:.2f}, no_rel={ctrl_no_rel:.2f}")
    print(f"  Δ (test_guided - control): orig={worst_orig-ctrl_orig:.2f}, no_rel={worst_no_rel-ctrl_no_rel:.2f}")
    print(f"  CLAIM HOLDS: {'YES' if worst_no_rel < ctrl_no_rel else 'NO'} (mismatch still < control)")

# ============================================================
# Phase 4: Strategy selection (Table 4 ablation)
# ============================================================
print("\n\n" + "=" * 70)
print("PHASE 4: TABLE 4 ABLATION (Oracle vs Fixed vs Control)")
print("=" * 70)

phase4 = load_json("/data/home/xiankunlin/project/emnlp/paper1/results/phase4_cascade_selection/results.json")

cond_scores = defaultdict(list)
cond_scores_no_rel = defaultdict(list)

for r in phase4["results"]:
    cond = r["condition"]
    cond_scores[cond].append(r["eval"]["score"])
    cond_scores_no_rel[cond].append(score_without_relevant(r["eval"]))

print("\n{:<10} {:>10} {:>10}".format("Condition", "Orig", "NoRel"))
print("-" * 35)
for cond in ["oracle", "fixed", "control"]:
    print("{:<10} {:>10.2f} {:>10.2f}".format(
        cond, np.mean(cond_scores[cond]), np.mean(cond_scores_no_rel[cond])))

oracle_orig = np.mean(cond_scores["oracle"])
fixed_orig = np.mean(cond_scores["fixed"])
control_orig = np.mean(cond_scores["control"])

oracle_nr = np.mean(cond_scores_no_rel["oracle"])
fixed_nr = np.mean(cond_scores_no_rel["fixed"])
control_nr = np.mean(cond_scores_no_rel["control"])

print(f"\n  Oracle - Fixed:   orig={oracle_orig-fixed_orig:.2f}, no_rel={oracle_nr-fixed_nr:.2f}")
print(f"  Oracle - Control: orig={oracle_orig-control_orig:.2f}, no_rel={oracle_nr-control_nr:.2f}")
print(f"  Fixed - Control:  orig={fixed_orig-control_orig:.2f}, no_rel={fixed_nr-control_nr:.2f}")
print(f"\n  ORDERING HOLDS: {'YES' if oracle_nr > fixed_nr > control_nr else 'NO'} (oracle > fixed > control)")

# ============================================================
# Per-type "relevant" hit rate analysis (checking for bias)
# ============================================================
print("\n\n" + "=" * 70)
print("RELEVANT HIT RATE BY TYPE × CONDITION (checking for lexical bias)")
print("=" * 70)

# Phase 3: relevant rate for best strategy vs control
print("\n{:<8} {:<12} {:>10} {:>10}".format("Type", "Condition", "Rel_rate", "N"))
print("-" * 45)

for ft in ["EDIT", "PLAN", "LOC", "LOGIC"]:
    for label, strat in [("best", best_map[ft]), ("control", control_map[ft])]:
        scores = type_strategy_scores[ft][strat]
        # Get relevant rates
        rel_count = 0
        total = 0
        for r in phase3["results"]:
            if r["failure_type"] == ft and r["strategy"] == strat:
                rel_count += int(r["eval"]["relevant"])
                total += 1
        if total > 0:
            print("{:<8} {:<12} {:>10.1%} {:>10}".format(ft, label, rel_count/total, total))

# ============================================================
# Summary
# ============================================================
print("\n\n" + "=" * 70)
print("SUMMARY: Does removing 'relevant' change any key conclusion?")
print("=" * 70)

conclusions = [
    "EDIT best > control",
    "PLAN best > control", 
    "LOC best > control",
    "LOGIC best ≈ control (weak)",
    "LOC mismatch < control",
    "Oracle > Fixed > Control"
]

print("\nAll conclusions to verify against no-relevant scoring:")
for c in conclusions:
    print(f"  - {c}")
print("\n(See numbers above for verification)")
