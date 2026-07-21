#!/usr/bin/env python3
"""Summarize existing judge-evaluation result JSON files.

This script is intentionally local-only: it reads files already present under
results/ and does not download datasets or load models. It recomputes the
metrics needed for the paper tables, including swap accuracy and first-position
selection rate, then writes per-run and curated group summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import mean, stdev


CHOICES = {"A", "B", "C"}
FLIP = {"A": "B", "B": "A", "C": "C"}


CURATED_GROUPS = [
    {
        "section": "primary_unbalanced",
        "name": "Baseline (Qwen2.5-7B)",
        "runs": ["baseline_qwen7b"],
    },
    {
        "section": "primary_unbalanced",
        "name": "SFT",
        "runs": ["SFT_unbalanced"],
    },
    {
        "section": "primary_unbalanced",
        "name": "DPO",
        "runs": ["DPO_unbalanced"],
    },
    {
        "section": "primary_unbalanced",
        "name": "GRPO Acc-only",
        "runs": [
            "EXP-006_accuracy_only",
            "EXP-006_accuracy_s2",
            "EXP-006_accuracy_s3",
        ],
    },
    {
        "section": "primary_unbalanced",
        "name": "GRPO + Decisive",
        "runs": [
            "EXP-007a_acc_decisive",
            "EXP-007a_acc_decisive_s2",
        ],
    },
    {
        "section": "primary_unbalanced",
        "name": "GRPO + Conf. proxy",
        "runs": [
            "EXP-008_acc_calib",
            "EXP-008_acc_calib_s2",
        ],
    },
    {
        "section": "primary_unbalanced",
        "name": "GRPO Full",
        "runs": [
            "EXP-009_full_composite",
            "EXP-009_full_composite_s2",
            "EXP-009_full_composite_s3",
            "EXP-009_full_s4",
        ],
    },
    {
        "section": "primary_lr",
        "name": "GRPO Full lr=1e-6",
        "runs": ["EXP-009_full_lr1e6"],
    },
    {
        "section": "primary_lr",
        "name": "GRPO Full lr=2e-6",
        "runs": ["EXP-009_full_lr2e6"],
    },
    {
        "section": "primary_lr",
        "name": "GRPO Full lr=3e-6",
        "runs": ["EXP-009_full_lr3e6"],
    },
    {
        "section": "primary_lr",
        "name": "GRPO Full lr=5e-6",
        "runs": ["EXP-009_full_composite"],
    },
    {
        "section": "primary_lr",
        "name": "GRPO Full lr=1e-5",
        "runs": ["EXP-009_full_lr1e5"],
    },
    {
        "section": "primary_balanced",
        "name": "SFT Balanced",
        "runs": ["SFT_balanced"],
    },
    {
        "section": "primary_balanced",
        "name": "GRPO Acc-only Balanced",
        "runs": [
            "EXP-006b_accuracy_balanced",
            "EXP-006b_accuracy_balanced_s2",
            "EXP-006b_accuracy_balanced_s3",
        ],
    },
    {
        "section": "primary_balanced",
        "name": "GRPO Decisive Balanced",
        "runs": [
            "EXP-007b_decisive_balanced",
            "balanced_decisive_s2",
            "balanced_decisive_s3",
        ],
    },
    {
        "section": "primary_balanced",
        "name": "GRPO Conf.-proxy Balanced",
        "runs": [
            "EXP-008b_calib_balanced",
            "balanced_calib_s2",
            "balanced_calib_s3",
        ],
    },
    {
        "section": "primary_balanced",
        "name": "GRPO Full Balanced",
        "runs": [
            "EXP-009b_full_balanced",
            "EXP-009b_full_balanced_s2",
            "EXP-009b_full_balanced_s3",
            "balanced_full_s4",
            "balanced_full_s5",
        ],
    },
    {
        "section": "external",
        "name": "JudgeLRM-7B",
        "runs": ["judgelrm_7b"],
    },
    {
        "section": "external",
        "name": "JudgeLRM-3B",
        "runs": ["judgelrm_3b"],
    },
]


CROSS_MODEL_SOURCES = [
    {
        "model": "Qwen2.5-7B",
        "method": "Baseline",
        "data": "---",
        "eval_paths": ["results/baseline_qwen7b/eval_results.json"],
    },
    {
        "model": "Qwen2.5-7B",
        "method": "SFT",
        "data": "unbal",
        "eval_paths": ["results/SFT_unbalanced/eval/eval_results.json"],
    },
    {
        "model": "Qwen2.5-7B",
        "method": "DPO",
        "data": "unbal",
        "eval_paths": ["results/DPO_unbalanced/eval/eval_results.json"],
    },
    {
        "model": "Qwen2.5-7B",
        "method": "GRPO",
        "data": "unbal",
        "eval_paths": [
            "results/EXP-009_full_composite/eval/eval_results.json",
            "results/EXP-009_full_composite_s2/eval/eval_results.json",
            "results/EXP-009_full_composite_s3/eval/eval_results.json",
            "results/EXP-009_full_s4/eval/eval_results.json",
        ],
    },
    {
        "model": "Qwen2.5-7B",
        "method": "SFT",
        "data": "bal",
        "eval_paths": ["results/SFT_balanced/eval/eval_results.json"],
    },
    {
        "model": "Qwen2.5-7B",
        "method": "GRPO",
        "data": "bal",
        "eval_paths": [
            "results/EXP-009b_full_balanced/eval/eval_results.json",
            "results/EXP-009b_full_balanced_s2/eval/eval_results.json",
            "results/EXP-009b_full_balanced_s3/eval/eval_results.json",
            "results/balanced_full_s4/eval/eval_results.json",
            "results/balanced_full_s5/eval/eval_results.json",
        ],
    },
    {
        "model": "Qwen3-8B",
        "method": "Baseline",
        "data": "---",
        "eval_paths": ["results/baseline_qwen3_8b/eval/eval_results.json"],
        "note": "Canonical Qwen3 baseline uses disable_thinking and low parse failure; excludes legacy top-level baseline_qwen3_8b/eval_results.json.",
    },
    {
        "model": "Qwen3-8B",
        "method": "SFT",
        "data": "unbal",
        "eval_paths": [
            "results/SFT_qwen3_8b_unbalanced/eval/eval_results.json",
            "results/SFT_qwen3_8b_unbal_s2/eval/eval_results.json",
        ],
    },
    {
        "model": "Qwen3-8B",
        "method": "DPO",
        "data": "unbal",
        "eval_paths": ["results/DPO_qwen3_8b_unbalanced/eval/eval_results.json"],
    },
    {
        "model": "Qwen3-8B",
        "method": "GRPO",
        "data": "unbal",
        "eval_paths": [
            "results/GRPO_qwen3_8b_unbalanced/eval/eval_results.json",
            "results/GRPO_qwen3_8b_unbalanced_s2/eval/eval_results.json",
            "results/GRPO_qwen3_8b_unbal_s3/eval/eval_results.json",
        ],
    },
    {
        "model": "Qwen3-8B",
        "method": "SFT",
        "data": "bal",
        "eval_paths": [
            "results/SFT_qwen3_8b_balanced/eval/eval_results.json",
            "results/SFT_qwen3_8b_balanced_s2/eval/eval_results.json",
        ],
    },
    {
        "model": "Qwen3-8B",
        "method": "DPO",
        "data": "bal",
        "eval_paths": ["results/DPO_qwen3_8b_balanced/eval/eval_results.json"],
    },
    {
        "model": "Qwen3-8B",
        "method": "GRPO",
        "data": "bal",
        "eval_paths": [
            "results/GRPO_qwen3_8b_balanced/eval/eval_results.json",
            "results/GRPO_qwen3_8b_balanced_s2/eval/eval_results.json",
        ],
    },
    {
        "model": "Mistral-7B",
        "method": "Baseline",
        "data": "---",
        "metrics_paths": ["results/baseline_mistral7b/eval/metrics.json"],
        "note": "The local eval_results artifact is a partial 80-sample file; metrics.json reports the canonical 449-sample run used in the paper.",
    },
    {
        "model": "Mistral-7B",
        "method": "SFT",
        "data": "unbal",
        "eval_paths": ["results/SFT_mistral7b_unbal/eval/eval_results.json"],
    },
    {
        "model": "Mistral-7B",
        "method": "SFT",
        "data": "bal",
        "eval_paths": ["results/SFT_mistral7b_balanced/eval/eval_results.json"],
    },
    {
        "model": "Mistral-7B",
        "method": "GRPO",
        "data": "unbal",
        "eval_paths": [
            "results/GRPO_mistral7b_unbal/eval/eval_results.json",
            "results/GRPO_mistral7b_unbal_s2/eval/eval_results.json",
            "results/GRPO_mistral7b_unbal_s3/eval/eval_results.json",
        ],
    },
    {
        "model": "Mistral-7B",
        "method": "GRPO",
        "data": "bal",
        "eval_paths": [
            "results/GRPO_mistral7b_balanced/eval/eval_results.json",
            "results/GRPO_mistral7b_balanced_s2/eval/eval_results.json",
        ],
    },
]


METRIC_ORDER = [
    "orig_acc",
    "swap_acc",
    "order_avg_acc",
    "consistency",
    "strict_consistency",
    "first_pos_rate",
    "orig_first_rate",
    "swap_first_rate",
    "position_bias",
    "tie_rate",
    "parse_fail_rate",
]


METRIC_LABELS = {
    "orig_acc": "Orig Acc",
    "swap_acc": "Swap Acc",
    "order_avg_acc": "Avg Acc",
    "consistency": "Con",
    "strict_consistency": "Strict Con",
    "first_pos_rate": "First-pos",
    "orig_first_rate": "Orig first",
    "swap_first_rate": "Swap first",
    "position_bias": "Bias",
    "tie_rate": "Tie",
    "parse_fail_rate": "Parse fail",
}


BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20260709


PAIRED_UNCERTAINTY_CHECKS = [
    {
        "comparison": "Qwen2.5 Baseline -> SFT unbalanced",
        "before": "results/baseline_qwen7b/eval_results.json",
        "after": "results/SFT_unbalanced/eval/eval_results.json",
    },
    {
        "comparison": "Qwen2.5 SFT unbalanced -> SFT balanced",
        "before": "results/SFT_unbalanced/eval/eval_results.json",
        "after": "results/SFT_balanced/eval/eval_results.json",
    },
    {
        "comparison": "Qwen2.5 Baseline -> SFT balanced",
        "before": "results/baseline_qwen7b/eval_results.json",
        "after": "results/SFT_balanced/eval/eval_results.json",
    },
]


SEED_UNCERTAINTY_CHECKS = [
    {
        "comparison": "Qwen2.5 GRPO full unbalanced -> balanced",
        "before": [
            "results/EXP-009_full_composite/eval/eval_results.json",
            "results/EXP-009_full_composite_s2/eval/eval_results.json",
            "results/EXP-009_full_composite_s3/eval/eval_results.json",
            "results/EXP-009_full_s4/eval/eval_results.json",
        ],
        "after": [
            "results/EXP-009b_full_balanced/eval/eval_results.json",
            "results/EXP-009b_full_balanced_s2/eval/eval_results.json",
            "results/EXP-009b_full_balanced_s3/eval/eval_results.json",
            "results/balanced_full_s4/eval/eval_results.json",
            "results/balanced_full_s5/eval/eval_results.json",
        ],
    },
    {
        "comparison": "Qwen3 GRPO full unbalanced -> balanced",
        "before": [
            "results/GRPO_qwen3_8b_unbalanced/eval/eval_results.json",
            "results/GRPO_qwen3_8b_unbalanced_s2/eval/eval_results.json",
            "results/GRPO_qwen3_8b_unbal_s3/eval/eval_results.json",
        ],
        "after": [
            "results/GRPO_qwen3_8b_balanced/eval/eval_results.json",
            "results/GRPO_qwen3_8b_balanced_s2/eval/eval_results.json",
        ],
    },
    {
        "comparison": "Mistral GRPO full unbalanced -> balanced",
        "before": [
            "results/GRPO_mistral7b_unbal/eval/eval_results.json",
            "results/GRPO_mistral7b_unbal_s2/eval/eval_results.json",
            "results/GRPO_mistral7b_unbal_s3/eval/eval_results.json",
        ],
        "after": [
            "results/GRPO_mistral7b_balanced/eval/eval_results.json",
            "results/GRPO_mistral7b_balanced_s2/eval/eval_results.json",
        ],
    },
]


DOMAIN_GROUPS = [
    {
        "name": "Chat",
        "short": "Chat",
        "categories": [
            "alpacaeval-easy",
            "alpacaeval-hard",
            "alpacaeval-length",
            "mt-bench-easy",
            "mt-bench-med",
            "mt-bench-hard",
            "llmbar-natural",
        ],
    },
    {
        "name": "Reasoning",
        "short": "Reas.",
        "categories": ["math-prm"],
    },
    {
        "name": "Safety",
        "short": "Safe",
        "categories": [
            "xstest-should-respond",
            "xstest-should-refuse",
            "refusals-offensive",
            "refusals-dangerous",
            "donotanswer",
        ],
    },
    {
        "name": "Code",
        "short": "Code",
        "categories": [
            "hep-cpp",
            "hep-go",
            "hep-java",
            "hep-js",
            "hep-python",
            "hep-rust",
        ],
    },
    {
        "name": "Adversarial",
        "short": "Adv.",
        "categories": [
            "llmbar-adver-GPTInst",
            "llmbar-adver-GPTOut",
            "llmbar-adver-manual",
            "llmbar-adver-neighbor",
        ],
    },
]


DOMAIN_SLICE_SOURCES = [
    {
        "setting": "Baseline",
        "paths": ["results/baseline_qwen7b/eval_results.json"],
    },
    {
        "setting": "GRPO unbalanced",
        "paths": [
            "results/EXP-009_full_composite/eval/eval_results.json",
            "results/EXP-009_full_composite_s2/eval/eval_results.json",
            "results/EXP-009_full_composite_s3/eval/eval_results.json",
            "results/EXP-009_full_s4/eval/eval_results.json",
        ],
    },
    {
        "setting": "GRPO balanced",
        "paths": [
            "results/EXP-009b_full_balanced/eval/eval_results.json",
            "results/EXP-009b_full_balanced_s2/eval/eval_results.json",
            "results/EXP-009b_full_balanced_s3/eval/eval_results.json",
            "results/balanced_full_s4/eval/eval_results.json",
            "results/balanced_full_s5/eval/eval_results.json",
        ],
    },
]


DOSE_RESPONSE_SOURCES = [
    {
        "condition": "50 mirrored balanced reference",
        "position_a_ratio": 50,
        "training_samples": 4178,
        "duplicated": "yes",
        "eval_paths": [
            "results/EXP-009b_full_balanced/eval/eval_results.json",
            "results/EXP-009b_full_balanced_s2/eval/eval_results.json",
            "results/EXP-009b_full_balanced_s3/eval/eval_results.json",
            "results/balanced_full_s4/eval/eval_results.json",
            "results/balanced_full_s5/eval/eval_results.json",
        ],
        "note": "Multi-seed GRPO full balanced reference used for consistency-loss normalization.",
    },
    {
        "condition": "50 non-duplicated balanced",
        "position_a_ratio": 50,
        "training_samples": 2089,
        "duplicated": "no",
        "eval_paths": ["results/GRPO_balanced_nodupe/eval/eval_results.json"],
        "note": "Single-seed control with one balanced orientation per training instance.",
    },
    {
        "condition": "60 ratio",
        "position_a_ratio": 60,
        "training_samples": 2089,
        "duplicated": "no",
        "eval_paths": ["results/GRPO_ratio60/eval/eval_results.json"],
        "note": "Single-seed intermediate confound-ratio diagnostic.",
    },
    {
        "condition": "75 ratio",
        "position_a_ratio": 75,
        "training_samples": 2089,
        "duplicated": "no",
        "eval_paths": ["results/GRPO_ratio75/eval/eval_results.json"],
        "note": "Single-seed intermediate confound-ratio diagnostic.",
    },
    {
        "condition": "80 ratio",
        "position_a_ratio": 80,
        "training_samples": 2089,
        "duplicated": "no",
        "eval_paths": ["results/GRPO_ratio80/eval/eval_results.json"],
        "note": "Single-seed intermediate confound-ratio diagnostic.",
    },
    {
        "condition": "90 ratio",
        "position_a_ratio": 90,
        "training_samples": 2089,
        "duplicated": "no",
        "eval_paths": ["results/GRPO_ratio90/eval/eval_results.json"],
        "note": "Single-seed intermediate confound-ratio diagnostic.",
    },
    {
        "condition": "95 ratio",
        "position_a_ratio": 95,
        "training_samples": 2089,
        "duplicated": "no",
        "eval_paths": ["results/GRPO_ratio95/eval/eval_results.json"],
        "note": "Single-seed intermediate confound-ratio diagnostic.",
    },
    {
        "condition": "100 fully unbalanced",
        "position_a_ratio": 100,
        "training_samples": 2089,
        "duplicated": "no",
        "eval_paths": [
            "results/EXP-009_full_composite/eval/eval_results.json",
            "results/EXP-009_full_composite_s2/eval/eval_results.json",
            "results/EXP-009_full_composite_s3/eval/eval_results.json",
            "results/EXP-009_full_s4/eval/eval_results.json",
        ],
        "note": "Multi-seed GRPO full unbalanced endpoint.",
    },
]


LEARNING_RATE_SWEEP_SOURCES = [
    {
        "model": "Qwen2.5-7B",
        "learning_rate": "baseline",
        "learning_rate_value": None,
        "reward": "none",
        "eval_paths": ["results/baseline_qwen7b/eval_results.json"],
        "note": "Untrained baseline.",
    },
    {
        "model": "Qwen2.5-7B",
        "learning_rate": "1e-6",
        "learning_rate_value": 1e-6,
        "reward": "full",
        "eval_paths": ["results/EXP-009_full_lr1e6/eval/eval_results.json"],
        "note": "Single-seed full-reward LR control.",
    },
    {
        "model": "Qwen2.5-7B",
        "learning_rate": "2e-6",
        "learning_rate_value": 2e-6,
        "reward": "full",
        "eval_paths": ["results/EXP-009_full_lr2e6/eval/eval_results.json"],
        "note": "Single-seed full-reward LR control.",
    },
    {
        "model": "Qwen2.5-7B",
        "learning_rate": "3e-6",
        "learning_rate_value": 3e-6,
        "reward": "full",
        "eval_paths": ["results/EXP-009_full_lr3e6/eval/eval_results.json"],
        "note": "Single-seed full-reward LR control.",
    },
    {
        "model": "Qwen2.5-7B",
        "learning_rate": "5e-6",
        "learning_rate_value": 5e-6,
        "reward": "full",
        "eval_paths": ["results/EXP-009_full_composite/eval/eval_results.json"],
        "note": "Standard full-reward seed used in the single-seed LR sweep.",
    },
    {
        "model": "Qwen2.5-7B",
        "learning_rate": "1e-5",
        "learning_rate_value": 1e-5,
        "reward": "full",
        "eval_paths": ["results/EXP-009_full_lr1e5/eval/eval_results.json"],
        "note": "Single-seed full-reward LR control.",
    },
    {
        "model": "Qwen3-8B",
        "learning_rate": "baseline",
        "learning_rate_value": None,
        "reward": "none",
        "eval_paths": ["results/baseline_qwen3_8b/eval/eval_results.json"],
        "note": "Untrained disable-thinking baseline.",
    },
    {
        "model": "Qwen3-8B",
        "learning_rate": "1e-6",
        "learning_rate_value": 1e-6,
        "reward": "accuracy",
        "eval_paths": ["results/GRPO_qwen3_8b_unbal_lr1e6/eval/eval_results.json"],
        "note": "Single-seed accuracy-reward LR control.",
    },
    {
        "model": "Qwen3-8B",
        "learning_rate": "2e-6",
        "learning_rate_value": 2e-6,
        "reward": "accuracy",
        "eval_paths": ["results/GRPO_qwen3_8b_unbal_lr2e6/eval/eval_results.json"],
        "note": "Single-seed accuracy-reward LR control.",
    },
    {
        "model": "Qwen3-8B",
        "learning_rate": "3e-6",
        "learning_rate_value": 3e-6,
        "reward": "accuracy",
        "eval_paths": ["results/GRPO_qwen3_8b_unbal_lr3e6/eval/eval_results.json"],
        "note": "Single-seed accuracy-reward LR control.",
    },
    {
        "model": "Qwen3-8B",
        "learning_rate": "5e-6",
        "learning_rate_value": 5e-6,
        "reward": "accuracy",
        "eval_paths": ["results/GRPO_qwen3_8b_unbalanced/eval/eval_results.json"],
        "note": "Seed-42 standard accuracy-reward run; multi-seed mean is reported in the cross-model table.",
    },
    {
        "model": "Qwen3-8B",
        "learning_rate": "1e-5",
        "learning_rate_value": 1e-5,
        "reward": "accuracy",
        "eval_paths": ["results/GRPO_qwen3_8b_unbal_lr1e5/eval/eval_results.json"],
        "note": "Single-seed accuracy-reward LR control.",
    },
]


TRAINING_DYNAMICS_SOURCES = [
    {
        "variant": "Baseline",
        "reward": "none",
        "steps": 0,
        "eval_paths": ["results/baseline_qwen7b/eval_results.json"],
        "include_in_paper": True,
        "note": "Untrained Qwen2.5-7B baseline shared by both trajectories.",
    },
    {
        "variant": "Acc-only",
        "reward": "accuracy",
        "steps": 100,
        "eval_paths": ["results/EXP-006_accuracy_only/eval_ckpt100/eval_results.json"],
        "include_in_paper": True,
        "note": "Accuracy-reward checkpoint used for the paired dynamics table.",
    },
    {
        "variant": "Acc-only",
        "reward": "accuracy",
        "steps": 200,
        "eval_paths": ["results/EXP-006_accuracy_only/eval_ckpt200/eval_results.json"],
        "include_in_paper": True,
        "note": "Accuracy-reward checkpoint used for the paired dynamics table.",
    },
    {
        "variant": "Acc-only",
        "reward": "accuracy",
        "steps": 300,
        "eval_paths": ["results/EXP-006_accuracy_only/eval_ckpt300/eval_results.json"],
        "include_in_paper": True,
        "note": "Accuracy-reward checkpoint used for the paired dynamics table.",
    },
    {
        "variant": "Acc-only",
        "reward": "accuracy",
        "steps": 500,
        "eval_paths": ["results/EXP-006_accuracy_only/eval/eval_results.json"],
        "include_in_paper": True,
        "note": "Final accuracy-reward run used as the 500-step endpoint.",
    },
    {
        "variant": "Full",
        "reward": "full",
        "steps": 100,
        "eval_paths": ["results/EXP-009_full_composite/eval_ckpt100/eval_results.json"],
        "include_in_paper": True,
        "note": "Full-reward checkpoint used for the paired dynamics table.",
    },
    {
        "variant": "Full",
        "reward": "full",
        "steps": 200,
        "eval_paths": ["results/EXP-009_full_composite/eval_ckpt200/eval_results.json"],
        "include_in_paper": True,
        "note": "Full-reward checkpoint used for the paired dynamics table.",
    },
    {
        "variant": "Full",
        "reward": "full",
        "steps": 300,
        "eval_paths": ["results/EXP-009_full_composite/eval_ckpt300/eval_results.json"],
        "include_in_paper": True,
        "note": "Full-reward checkpoint used for the paired dynamics table.",
    },
    {
        "variant": "Full",
        "reward": "full",
        "steps": 400,
        "eval_paths": ["results/EXP-009_full_composite/eval_step400/eval_results.json"],
        "include_in_paper": False,
        "note": "Additional full-reward-only checkpoint; omitted from the paired paper table because acc-only lacks a matching 400-step artifact.",
    },
    {
        "variant": "Full",
        "reward": "full",
        "steps": 500,
        "eval_paths": ["results/EXP-009_full_composite/eval/eval_results.json"],
        "include_in_paper": True,
        "note": "Final full-reward run used as the 500-step endpoint.",
    },
]


REWARD_ABLATION_SOURCES = [
    {
        "model": "Qwen2.5-7B",
        "reward": "Accuracy only",
        "eval_paths": [
            "results/EXP-006_accuracy_only/eval/eval_results.json",
            "results/EXP-006_accuracy_s2/eval/eval_results.json",
            "results/EXP-006_accuracy_s3/eval/eval_results.json",
        ],
        "note": "Multi-seed accuracy-reward GRPO on unbalanced data.",
    },
    {
        "model": "Qwen2.5-7B",
        "reward": "+ Decisive",
        "eval_paths": [
            "results/EXP-007a_acc_decisive/eval/eval_results.json",
            "results/EXP-007a_acc_decisive_s2/eval/eval_results.json",
        ],
        "note": "Multi-seed accuracy plus decisiveness reward on unbalanced data.",
    },
    {
        "model": "Qwen2.5-7B",
        "reward": "+ Confidence proxy",
        "eval_paths": [
            "results/EXP-008_acc_calib/eval/eval_results.json",
            "results/EXP-008_acc_calib_s2/eval/eval_results.json",
        ],
        "note": "Multi-seed accuracy plus fixed-confidence proxy on unbalanced data.",
    },
    {
        "model": "Qwen2.5-7B",
        "reward": "Full composite",
        "eval_paths": [
            "results/EXP-009_full_composite/eval/eval_results.json",
            "results/EXP-009_full_composite_s2/eval/eval_results.json",
            "results/EXP-009_full_composite_s3/eval/eval_results.json",
            "results/EXP-009_full_s4/eval/eval_results.json",
        ],
        "note": "Multi-seed full composite reward on unbalanced data.",
    },
    {
        "model": "Qwen3-8B",
        "reward": "Accuracy only",
        "eval_paths": ["results/GRPO_qwen3_8b_unbalanced/eval/eval_results.json"],
        "note": "Single-seed accuracy-reward GRPO on unbalanced data.",
    },
    {
        "model": "Qwen3-8B",
        "reward": "+ Decisive",
        "eval_paths": ["results/GRPO_qwen3_8b_decisive_unbal/eval/eval_results.json"],
        "note": "Single-seed accuracy plus decisiveness reward on unbalanced data.",
    },
    {
        "model": "Qwen3-8B",
        "reward": "+ Confidence proxy",
        "eval_paths": ["results/GRPO_qwen3_8b_calib_unbal/eval/eval_results.json"],
        "note": "Single-seed accuracy plus fixed-confidence proxy on unbalanced data.",
    },
    {
        "model": "Qwen3-8B",
        "reward": "Full composite",
        "eval_paths": ["results/GRPO_qwen3_8b_full_composite_unbal/eval/eval_results.json"],
        "note": "Single-seed full composite reward on unbalanced data.",
    },
]


PROMPT_LABEL_CONTROL_SOURCES = [
    {
        "model": "Qwen2.5-7B",
        "control": "A / B",
        "eval_mode": "standard prompt",
        "eval_paths": ["results/EXP-009_full_composite/eval/eval_results.json"],
        "include_in_paper": True,
        "note": "Standard unbalanced-GRPO full-composite run used as the Qwen2.5 label-control anchor.",
    },
    {
        "model": "Qwen2.5-7B",
        "control": "Anti-prompt",
        "eval_mode": "ordering-randomized instruction",
        "eval_paths": ["results/GRPO_antiprompt_eval/eval_results.json"],
        "include_in_paper": True,
        "note": "Prompt-level mitigation: instructs the judge to ignore order and notes that ordering is random.",
    },
    {
        "model": "Qwen2.5-7B",
        "control": "1 / 2",
        "eval_mode": "numeric labels",
        "eval_paths": ["results/label_variant_numeric/eval_results.json"],
        "include_in_paper": True,
        "note": "Alternative label vocabulary for the same unbalanced-GRPO checkpoint.",
    },
    {
        "model": "Qwen2.5-7B",
        "control": "Left / Right",
        "eval_mode": "spatial labels",
        "eval_paths": ["results/label_variant_leftright/eval_results.json"],
        "include_in_paper": True,
        "note": "Alternative label vocabulary for the same unbalanced-GRPO checkpoint.",
    },
    {
        "model": "Qwen3-8B",
        "control": "A / B",
        "eval_mode": "standard prompt",
        "eval_paths": ["results/GRPO_qwen3_8b_unbalanced/eval/eval_results.json"],
        "include_in_paper": True,
        "note": "Standard Qwen3 unbalanced accuracy-reward run used as the label-control anchor.",
    },
    {
        "model": "Qwen3-8B",
        "control": "Anti-prompt",
        "eval_mode": "ordering-randomized instruction, thinking disabled",
        "eval_paths": ["results/antiprompt_qwen3_nothink/eval_results.json"],
        "include_in_paper": True,
        "note": "Reported anti-prompt control with thinking disabled to keep parse failures low.",
    },
    {
        "model": "Qwen3-8B",
        "control": "1 / 2",
        "eval_mode": "numeric labels, thinking disabled",
        "eval_paths": ["results/label_variant_numeric_qwen3_nothink/eval_results.json"],
        "include_in_paper": True,
        "note": "Reported numeric-label control with thinking disabled to keep parse failures low.",
    },
    {
        "model": "Qwen3-8B",
        "control": "Left / Right",
        "eval_mode": "spatial labels, thinking disabled",
        "eval_paths": ["results/label_variant_leftright_qwen3_nothink/eval_results.json"],
        "include_in_paper": True,
        "note": "Reported spatial-label control with thinking disabled to keep parse failures low.",
    },
    {
        "model": "Qwen3-8B",
        "control": "Anti-prompt (thinking on)",
        "eval_mode": "ordering-randomized instruction, thinking enabled",
        "eval_paths": ["results/antiprompt_qwen3/eval_results.json"],
        "include_in_paper": False,
        "note": "Diagnostic only: high parse-failure rate makes the control hard to interpret.",
    },
    {
        "model": "Qwen3-8B",
        "control": "1 / 2 (thinking on)",
        "eval_mode": "numeric labels, thinking enabled",
        "eval_paths": ["results/label_variant_numeric_qwen3/eval_results.json"],
        "include_in_paper": False,
        "note": "Diagnostic only: high parse-failure rate makes the control hard to interpret.",
    },
    {
        "model": "Qwen3-8B",
        "control": "Left / Right (thinking on)",
        "eval_mode": "spatial labels, thinking enabled",
        "eval_paths": ["results/label_variant_leftright_qwen3/eval_results.json"],
        "include_in_paper": False,
        "note": "Diagnostic only: high parse-failure rate makes the control hard to interpret.",
    },
    {
        "model": "Mistral-7B",
        "control": "A / B",
        "eval_mode": "standard prompt",
        "eval_paths": ["results/GRPO_mistral7b_unbal/eval/eval_results.json"],
        "include_in_paper": True,
        "note": "Standard Mistral unbalanced-GRPO run used as the label-control anchor.",
    },
    {
        "model": "Mistral-7B",
        "control": "Anti-prompt",
        "eval_mode": "ordering-randomized instruction",
        "eval_paths": ["results/antiprompt_mistral/eval_results.json"],
        "include_in_paper": True,
        "note": "Prompt-level mitigation: instructs the judge to ignore order and notes that ordering is random.",
    },
    {
        "model": "Mistral-7B",
        "control": "1 / 2",
        "eval_mode": "numeric labels",
        "eval_paths": ["results/label_variant_numeric_mistral/eval_results.json"],
        "include_in_paper": True,
        "note": "Alternative label vocabulary for the same unbalanced-GRPO checkpoint.",
    },
    {
        "model": "Mistral-7B",
        "control": "Left / Right",
        "eval_mode": "spatial labels",
        "eval_paths": ["results/label_variant_leftright_mistral/eval_results.json"],
        "include_in_paper": True,
        "note": "Alternative label vocabulary for the same unbalanced-GRPO checkpoint.",
    },
]


POSTHOC_FILTERING_SOURCES = [
    {
        "setting": "Baseline",
        "eval_paths": ["results/baseline_qwen7b/eval_results.json"],
        "note": "Untrained Qwen2.5 baseline.",
    },
    {
        "setting": "SFT unbalanced",
        "eval_paths": ["results/SFT_unbalanced/eval/eval_results.json"],
        "note": "Unbalanced SFT endpoint.",
    },
    {
        "setting": "SFT balanced",
        "eval_paths": ["results/SFT_balanced/eval/eval_results.json"],
        "note": "Balanced SFT endpoint.",
    },
    {
        "setting": "GRPO acc-only unbalanced",
        "eval_paths": [
            "results/EXP-006_accuracy_only/eval/eval_results.json",
            "results/EXP-006_accuracy_s2/eval/eval_results.json",
            "results/EXP-006_accuracy_s3/eval/eval_results.json",
        ],
        "note": "Multi-seed unbalanced accuracy-reward GRPO.",
    },
    {
        "setting": "GRPO full unbalanced",
        "eval_paths": [
            "results/EXP-009_full_composite/eval/eval_results.json",
            "results/EXP-009_full_composite_s2/eval/eval_results.json",
            "results/EXP-009_full_composite_s3/eval/eval_results.json",
            "results/EXP-009_full_s4/eval/eval_results.json",
        ],
        "note": "Multi-seed unbalanced full-reward GRPO.",
    },
    {
        "setting": "GRPO full balanced",
        "eval_paths": [
            "results/EXP-009b_full_balanced/eval/eval_results.json",
            "results/EXP-009b_full_balanced_s2/eval/eval_results.json",
            "results/EXP-009b_full_balanced_s3/eval/eval_results.json",
            "results/balanced_full_s4/eval/eval_results.json",
            "results/balanced_full_s5/eval/eval_results.json",
        ],
        "note": "Multi-seed balanced full-reward GRPO.",
    },
]


LENGTH_CONFOUND_SOURCES = [
    {
        "setting": "Baseline",
        "confound": "none",
        "eval_paths": ["results/baseline_qwen7b/eval_results.json"],
        "note": "Untrained Qwen2.5 baseline.",
    },
    {
        "setting": "Position-confounded GRPO-A",
        "confound": "preferred response always first",
        "eval_paths": [
            "results/EXP-006_accuracy_only/eval/eval_results.json",
            "results/EXP-006_accuracy_s2/eval/eval_results.json",
            "results/EXP-006_accuracy_s3/eval/eval_results.json",
        ],
        "note": "Primary unbalanced accuracy-reward GRPO comparison.",
    },
    {
        "setting": "Length-confounded GRPO",
        "confound": "longer response preferred in training",
        "eval_paths": [
            "results/EXP-LENGTH_confounded/eval/eval_results.json",
            "results/length_confounded_s2/eval/eval_results.json",
            "results/length_confounded_s3/eval/eval_results.json",
        ],
        "note": "Control trained on a length-confounded objective for 300 steps.",
    },
]


EXTERNAL_JUDGE_SOURCES = [
    {
        "model": "JudgeLRM-7B",
        "checkpoint": "nuojohnchen/JudgeLRM-7B",
        "eval_paths": ["results/judgelrm_7b/eval/eval_results.json"],
        "note": "Public JudgeLRM 7B checkpoint evaluated with the same two-order diagnostic.",
    },
    {
        "model": "JudgeLRM-3B",
        "checkpoint": "nuojohnchen/JudgeLRM-3B",
        "eval_paths": ["results/judgelrm_3b/eval/eval_results.json"],
        "note": "Public JudgeLRM 3B checkpoint evaluated with the same two-order diagnostic.",
    },
]


def discover_eval_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("**/eval_results.json"))


def infer_run_name(path: Path, results_dir: Path) -> str:
    rel = path.relative_to(results_dir)
    if rel.parts[-2] == "eval":
        return rel.parts[-3]
    return rel.parts[-2]


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    radius = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return (max(0.0, center - radius), min(1.0, center + radius))


def is_choice(value: object) -> bool:
    return isinstance(value, str) and value in CHOICES


def summarize_run(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        raise ValueError("empty eval_results.json")

    orig_correct = 0
    swap_correct = 0
    consistent = 0
    strict_consistent = 0
    orig_first = 0
    swap_first = 0
    both_first = 0
    both_second = 0
    tie_count = 0
    parse_fail_count = 0

    for row in results:
        gold = row.get("gold_label")
        pred = row.get("predicted")
        swap_pred = row.get("swap_predicted")
        swap_gold = FLIP.get(gold)

        if pred == gold:
            orig_correct += 1
        if swap_pred == swap_gold:
            swap_correct += 1

        if "is_consistent" in row:
            is_consistent = bool(row.get("is_consistent"))
        else:
            is_consistent = swap_pred == FLIP.get(pred)
        if is_consistent:
            consistent += 1
        if is_consistent and not (pred == "C" and swap_pred == "C"):
            strict_consistent += 1

        if pred == "A":
            orig_first += 1
        if swap_pred == "A":
            swap_first += 1
        if pred == "A" and swap_pred == "A":
            both_first += 1
        if pred == "B" and swap_pred == "B":
            both_second += 1
        if pred == "C":
            tie_count += 1
        if swap_pred == "C":
            tie_count += 1
        if not is_choice(pred):
            parse_fail_count += 1
        if not is_choice(swap_pred):
            parse_fail_count += 1

    counts = {
        "orig_acc": (orig_correct, n),
        "swap_acc": (swap_correct, n),
        "consistency": (consistent, n),
        "strict_consistency": (strict_consistent, n),
        "first_pos_rate": (orig_first + swap_first, 2 * n),
        "orig_first_rate": (orig_first, n),
        "swap_first_rate": (swap_first, n),
        "tie_rate": (tie_count, 2 * n),
        "parse_fail_rate": (parse_fail_count, 2 * n),
    }

    metrics = {}
    for metric, (num, den) in counts.items():
        lo, hi = wilson_interval(num, den)
        metrics[metric] = {
            "value": num / den if den else 0.0,
            "count": num,
            "denominator": den,
            "ci95_low": lo,
            "ci95_high": hi,
        }

    # Equal-weight accuracy over the original and swapped orientations.  We
    # retain the paired structure for bootstrap checks below instead of using
    # a binomial interval that would incorrectly treat orientations as
    # independent.
    metrics["order_avg_acc"] = {
        "value": (orig_correct + swap_correct) / (2 * n),
        "count": orig_correct + swap_correct,
        "denominator": 2 * n,
        "ci95_low": None,
        "ci95_high": None,
    }

    # This matches the paper's "always-first minus always-second" position-bias
    # convention and can be negative if a run favors the second position.
    metrics["position_bias"] = {
        "value": (both_first - both_second) / n,
        "count": both_first - both_second,
        "denominator": n,
        "ci95_low": None,
        "ci95_high": None,
        "always_first_rate": both_first / n,
        "always_second_rate": both_second / n,
    }
    return {"n": n, "metrics": metrics}


def pct(value: float | None) -> float | None:
    if value is None:
        return None
    return 100.0 * value


def fmt_pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{100.0 * value:.1f}"


def fmt_group_metric(metric_values: list[dict], metric_name: str) -> str:
    values = [m["value"] for m in metric_values]
    if not values:
        return ""
    value = mean(values)
    if len(values) > 1:
        return f"{100.0 * value:.1f} +/- {100.0 * stdev(values):.1f}"
    metric = metric_values[0]
    lo = metric.get("ci95_low")
    hi = metric.get("ci95_high")
    if lo is None or hi is None:
        return f"{100.0 * value:.1f}"
    return f"{100.0 * value:.1f} [{100.0 * lo:.1f}, {100.0 * hi:.1f}]"


def fmt_mean_std(values: list[float]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return f"{100.0 * values[0]:.1f}"
    return f"{100.0 * mean(values):.1f} +/- {100.0 * stdev(values):.1f}"


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def bootstrap_mean_interval(values: list[float], rng: random.Random, reps: int = BOOTSTRAP_REPS) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    n = len(values)
    draws = []
    for _ in range(reps):
        draws.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    draws.sort()
    return (percentile(draws, 0.025), percentile(draws, 0.975))


def bootstrap_seed_diff_interval(
    before_values: list[float],
    after_values: list[float],
    rng: random.Random,
    reps: int = BOOTSTRAP_REPS,
) -> tuple[float, float]:
    if not before_values or not after_values:
        return (0.0, 0.0)
    nb = len(before_values)
    na = len(after_values)
    draws = []
    for _ in range(reps):
        before_mean = sum(before_values[rng.randrange(nb)] for _ in range(nb)) / nb
        after_mean = sum(after_values[rng.randrange(na)] for _ in range(na)) / na
        draws.append(after_mean - before_mean)
    draws.sort()
    return (percentile(draws, 0.025), percentile(draws, 0.975))


def row_metric(row: dict, metric: str) -> float:
    gold = row.get("gold_label")
    pred = row.get("predicted")
    swap_pred = row.get("swap_predicted")
    if metric == "orig_acc":
        return 1.0 if pred == gold else 0.0
    if metric == "swap_acc":
        return 1.0 if swap_pred == FLIP.get(gold) else 0.0
    if metric == "order_avg_acc":
        orig = 1.0 if pred == gold else 0.0
        swap = 1.0 if swap_pred == FLIP.get(gold) else 0.0
        return (orig + swap) / 2.0
    if metric == "consistency":
        if "is_consistent" in row:
            return 1.0 if row.get("is_consistent") else 0.0
        return 1.0 if swap_pred == FLIP.get(pred) else 0.0
    if metric == "first_pos_rate":
        return ((1.0 if pred == "A" else 0.0) + (1.0 if swap_pred == "A" else 0.0)) / 2.0
    raise ValueError(f"unsupported paired metric: {metric}")


def paired_metric_values(results: list[dict], metric: str) -> dict[str, float]:
    values = {}
    for idx, row in enumerate(results):
        key = str(row.get("id", idx))
        values[key] = row_metric(row, metric)
    return values


def make_paired_uncertainty_rows(results_dir: Path) -> list[dict]:
    rng = random.Random(BOOTSTRAP_SEED)
    rows = []
    for check in PAIRED_UNCERTAINTY_CHECKS:
        before_path = Path(check["before"])
        after_path = Path(check["after"])
        if not before_path.exists() or not after_path.exists():
            continue
        before_results = load_json(before_path)
        after_results = load_json(after_path)
        for metric in ["orig_acc", "swap_acc", "order_avg_acc", "consistency", "first_pos_rate"]:
            before_values = paired_metric_values(before_results, metric)
            after_values = paired_metric_values(after_results, metric)
            common_ids = sorted(set(before_values) & set(after_values))
            if not common_ids:
                continue
            diffs = [after_values[item_id] - before_values[item_id] for item_id in common_ids]
            before_mean = mean(before_values[item_id] for item_id in common_ids)
            after_mean = mean(after_values[item_id] for item_id in common_ids)
            lo, hi = bootstrap_mean_interval(diffs, rng)
            rows.append({
                "comparison": check["comparison"],
                "kind": "paired_instance_bootstrap",
                "metric": metric,
                "before": before_mean,
                "after": after_mean,
                "diff": after_mean - before_mean,
                "ci95_low": lo,
                "ci95_high": hi,
                "n_pairs": len(common_ids),
                "before_runs": "",
                "after_runs": "",
            })
    return rows


def make_seed_uncertainty_rows(run_by_path: dict[str, dict]) -> list[dict]:
    rng = random.Random(BOOTSTRAP_SEED + 1)
    rows = []
    for check in SEED_UNCERTAINTY_CHECKS:
        before_runs = [run_by_path[path] for path in check["before"] if path in run_by_path]
        after_runs = [run_by_path[path] for path in check["after"] if path in run_by_path]
        if not before_runs or not after_runs:
            continue
        for metric in ["orig_acc", "order_avg_acc", "consistency", "first_pos_rate"]:
            before_values = [run["metrics"][metric]["value"] for run in before_runs]
            after_values = [run["metrics"][metric]["value"] for run in after_runs]
            lo, hi = bootstrap_seed_diff_interval(before_values, after_values, rng)
            rows.append({
                "comparison": check["comparison"],
                "kind": "seed_mean_bootstrap",
                "metric": metric,
                "before": mean(before_values),
                "after": mean(after_values),
                "diff": mean(after_values) - mean(before_values),
                "ci95_low": lo,
                "ci95_high": hi,
                "n_pairs": "",
                "before_runs": len(before_runs),
                "after_runs": len(after_runs),
            })
    return rows


def write_uncertainty_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "comparison",
        "kind",
        "metric",
        "before_pct",
        "after_pct",
        "diff_pp",
        "ci95_low_pp",
        "ci95_high_pp",
        "n_pairs",
        "before_runs",
        "after_runs",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "comparison": row["comparison"],
                "kind": row["kind"],
                "metric": row["metric"],
                "before_pct": pct(row["before"]),
                "after_pct": pct(row["after"]),
                "diff_pp": pct(row["diff"]),
                "ci95_low_pp": pct(row["ci95_low"]),
                "ci95_high_pp": pct(row["ci95_high"]),
                "n_pairs": row["n_pairs"],
                "before_runs": row["before_runs"],
                "after_runs": row["after_runs"],
            })


def fmt_ci(row: dict) -> str:
    return f"{pct(row['diff']):.1f} [{pct(row['ci95_low']):.1f}, {pct(row['ci95_high']):.1f}]"


def write_uncertainty_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Uncertainty Checks",
        "",
        f"Generated from local artifacts only. Bootstrap intervals use {BOOTSTRAP_REPS:,} resamples with seed {BOOTSTRAP_SEED}.",
        "Differences are `after - before` in percentage points.",
        "Paired-instance checks resample matched test examples; seed-mean checks resample available training seeds.",
        "",
        "## Paired Instance Bootstrap",
        "",
        "| Comparison | Metric | Before | After | Diff [95% CI] | Pairs |",
        "|---|---|---:|---:|---:|---:|",
    ]
    paired_rows = [row for row in rows if row["kind"] == "paired_instance_bootstrap"]
    for row in paired_rows:
        lines.append(
            f"| {row['comparison']} | {METRIC_LABELS[row['metric']]} | "
            f"{pct(row['before']):.1f} | {pct(row['after']):.1f} | {fmt_ci(row)} | {row['n_pairs']} |"
        )
    lines.extend([
        "",
        "## Seed-Mean Bootstrap",
        "",
        "| Comparison | Metric | Before | After | Diff [95% CI] | Runs |",
        "|---|---|---:|---:|---:|---:|",
    ])
    seed_rows = [row for row in rows if row["kind"] == "seed_mean_bootstrap"]
    for row in seed_rows:
        runs = f"{row['before_runs']} -> {row['after_runs']}"
        lines.append(
            f"| {row['comparison']} | {METRIC_LABELS[row['metric']]} | "
            f"{pct(row['before']):.1f} | {pct(row['after']):.1f} | {fmt_ci(row)} | {runs} |"
        )
    path.write_text("\n".join(lines) + "\n")


def domain_name_for_category(category: str) -> str | None:
    for domain in DOMAIN_GROUPS:
        if category in domain["categories"]:
            return domain["name"]
    return None


def make_domain_slice_rows() -> list[dict]:
    rows = []
    for source in DOMAIN_SLICE_SOURCES:
        run_domain_summaries: list[dict[str, dict]] = []
        run_paths = []
        for path_str in source["paths"]:
            path = Path(path_str)
            if not path.exists():
                continue
            results = load_json(path)
            if not isinstance(results, list):
                continue
            by_domain = {domain["name"]: [] for domain in DOMAIN_GROUPS}
            for row in results:
                domain_name = domain_name_for_category(row.get("category", ""))
                if domain_name is not None:
                    by_domain[domain_name].append(row)
            run_domain_summaries.append({
                domain_name: summarize_run(domain_rows)
                for domain_name, domain_rows in by_domain.items()
                if domain_rows
            })
            run_paths.append(path_str)

        if not run_domain_summaries:
            continue

        for domain in DOMAIN_GROUPS:
            domain_name = domain["name"]
            present = [
                run_summary[domain_name]
                for run_summary in run_domain_summaries
                if domain_name in run_summary
            ]
            if not present:
                continue
            n_values = [summary["n"] for summary in present]
            for metric in ["orig_acc", "swap_acc", "consistency", "first_pos_rate"]:
                values = [summary["metrics"][metric]["value"] for summary in present]
                rows.append({
                    "setting": source["setting"],
                    "domain": domain_name,
                    "domain_short": domain["short"],
                    "categories": domain["categories"],
                    "metric": metric,
                    "n_runs": len(values),
                    "n_per_run": n_values[0] if len(set(n_values)) == 1 else ";".join(str(n) for n in n_values),
                    "mean": mean(values),
                    "seed_std": stdev(values) if len(values) > 1 else None,
                    "values": values,
                    "paths": run_paths,
                })
    return rows


def write_domain_slice_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "setting",
        "domain",
        "domain_short",
        "n_runs",
        "n_per_run",
        "metric",
        "mean_pct",
        "seed_std_pct",
        "values_pct",
        "categories",
        "paths",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "setting": row["setting"],
                "domain": row["domain"],
                "domain_short": row["domain_short"],
                "n_runs": row["n_runs"],
                "n_per_run": row["n_per_run"],
                "metric": row["metric"],
                "mean_pct": pct(row["mean"]),
                "seed_std_pct": pct(row["seed_std"]),
                "values_pct": ";".join(f"{pct(value):.6f}" for value in row["values"]),
                "categories": ";".join(row["categories"]),
                "paths": ";".join(row["paths"]),
            })


def domain_cell(rows_by_key: dict[tuple[str, str, str], dict], setting: str, domain: str, metric: str) -> str:
    row = rows_by_key[(setting, domain, metric)]
    if row["seed_std"] is None:
        return f"{pct(row['mean']):.1f}"
    return f"{pct(row['mean']):.1f} +/- {pct(row['seed_std']):.1f}"


def write_domain_slice_markdown(path: Path, rows: list[dict]) -> None:
    rows_by_key = {
        (row["setting"], row["domain"], row["metric"]): row
        for row in rows
    }
    domain_names = [domain["name"] for domain in DOMAIN_GROUPS]
    domain_headers = []
    for domain in DOMAIN_GROUPS:
        sample_row = next(
            row for row in rows
            if row["domain"] == domain["name"] and row["metric"] == "orig_acc"
        )
        domain_headers.append(f"{domain['short']} (n={sample_row['n_per_run']})")

    lines = [
        "# Domain Slice Summary",
        "",
        "Generated from local artifacts only. Values are percentages.",
        "Multi-run settings are formatted as mean +/- sample standard deviation across seeds.",
        "",
        "Domain mapping:",
    ]
    for domain in DOMAIN_GROUPS:
        lines.append(f"- {domain['name']}: {', '.join(domain['categories'])}")
    lines.extend([
        "",
        "| Setting | Metric | " + " | ".join(domain_headers) + " |",
        "|---|---|" + "---:|" * len(domain_headers),
    ])
    for source in DOMAIN_SLICE_SOURCES:
        for metric in ["orig_acc", "consistency", "first_pos_rate"]:
            label = {
                "orig_acc": "Orig Acc",
                "consistency": "Con",
                "first_pos_rate": "First-pos",
            }[metric]
            cells = [
                domain_cell(rows_by_key, source["setting"], domain_name, metric)
                for domain_name in domain_names
            ]
            lines.append(f"| {source['setting']} | {label} | " + " | ".join(cells) + " |")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def make_dose_response_rows(run_by_path: dict[str, dict]) -> list[dict]:
    rows = []
    reference = None
    for source in DOSE_RESPONSE_SOURCES:
        runs = [
            run_by_path[path]
            for path in source["eval_paths"]
            if path in run_by_path
        ]
        if not runs:
            continue
        metric_values = {}
        for metric in ["orig_acc", "swap_acc", "consistency", "first_pos_rate"]:
            values = [run["metrics"][metric]["value"] for run in runs]
            metric_values[metric] = {
                "mean": mean(values),
                "seed_std": stdev(values) if len(values) > 1 else None,
                "values": values,
            }
        if source["condition"] == "50 mirrored balanced reference":
            reference = metric_values["consistency"]["mean"]
        rows.append({
            "condition": source["condition"],
            "position_a_ratio": source["position_a_ratio"],
            "training_samples": source["training_samples"],
            "duplicated": source["duplicated"],
            "n_runs": len(runs),
            "n_samples": ";".join(str(run["n"]) for run in runs),
            "source_paths": source["eval_paths"],
            "note": source["note"],
            "metrics": metric_values,
        })

    if reference is None:
        return rows
    for row in rows:
        row["consistency_loss"] = reference - row["metrics"]["consistency"]["mean"]
        row["reference_consistency"] = reference
    return rows


def write_dose_response_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "condition",
        "position_a_ratio",
        "training_samples",
        "duplicated",
        "n_runs",
        "n_samples",
        "orig_acc_mean_pct",
        "orig_acc_seed_std_pct",
        "swap_acc_mean_pct",
        "swap_acc_seed_std_pct",
        "consistency_mean_pct",
        "consistency_seed_std_pct",
        "first_pos_rate_mean_pct",
        "first_pos_rate_seed_std_pct",
        "reference_consistency_pct",
        "consistency_loss_pp",
        "orig_acc_values_pct",
        "consistency_values_pct",
        "first_pos_rate_values_pct",
        "source_paths",
        "note",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metrics = row["metrics"]
            writer.writerow({
                "condition": row["condition"],
                "position_a_ratio": row["position_a_ratio"],
                "training_samples": row["training_samples"],
                "duplicated": row["duplicated"],
                "n_runs": row["n_runs"],
                "n_samples": row["n_samples"],
                "orig_acc_mean_pct": pct(metrics["orig_acc"]["mean"]),
                "orig_acc_seed_std_pct": pct(metrics["orig_acc"]["seed_std"]),
                "swap_acc_mean_pct": pct(metrics["swap_acc"]["mean"]),
                "swap_acc_seed_std_pct": pct(metrics["swap_acc"]["seed_std"]),
                "consistency_mean_pct": pct(metrics["consistency"]["mean"]),
                "consistency_seed_std_pct": pct(metrics["consistency"]["seed_std"]),
                "first_pos_rate_mean_pct": pct(metrics["first_pos_rate"]["mean"]),
                "first_pos_rate_seed_std_pct": pct(metrics["first_pos_rate"]["seed_std"]),
                "reference_consistency_pct": pct(row.get("reference_consistency")),
                "consistency_loss_pp": pct(row.get("consistency_loss")),
                "orig_acc_values_pct": ";".join(f"{pct(value):.6f}" for value in metrics["orig_acc"]["values"]),
                "consistency_values_pct": ";".join(f"{pct(value):.6f}" for value in metrics["consistency"]["values"]),
                "first_pos_rate_values_pct": ";".join(f"{pct(value):.6f}" for value in metrics["first_pos_rate"]["values"]),
                "source_paths": ";".join(row["source_paths"]),
                "note": row["note"],
            })


def fmt_dose_metric(row: dict, metric: str) -> str:
    metric_data = row["metrics"][metric]
    if metric_data["seed_std"] is None:
        return f"{pct(metric_data['mean']):.1f}"
    return f"{pct(metric_data['mean']):.1f} +/- {pct(metric_data['seed_std']):.1f}"


def write_dose_response_markdown(path: Path, rows: list[dict]) -> None:
    reference = rows[0].get("reference_consistency") if rows else None
    lines = [
        "# Confound-Ratio and Duplication Summary",
        "",
        "Generated from local artifacts only. Values are percentages unless marked as pp.",
    ]
    if reference is not None:
        lines.append(f"Consistency loss is computed as {pct(reference):.1f} minus each row's consistency.")
    lines.extend([
        "Multi-run rows are formatted as mean +/- sample standard deviation; intermediate ratio rows are single-seed diagnostics.",
        "",
        "| Condition | Pos-A | Train n | Dup. | Runs | Acc | Con | First-pos | Loss pp |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            "| {condition} | {ratio} | {train_n} | {dup} | {runs} | {acc} | {con} | {first} | {loss:.1f} |".format(
                condition=row["condition"],
                ratio=row["position_a_ratio"],
                train_n=row["training_samples"],
                dup=row["duplicated"],
                runs=row["n_runs"],
                acc=fmt_dose_metric(row, "orig_acc"),
                con=fmt_dose_metric(row, "consistency"),
                first=fmt_dose_metric(row, "first_pos_rate"),
                loss=pct(row.get("consistency_loss")),
            )
        )
    lines.extend([
        "",
        "## Sources",
        "",
    ])
    for row in rows:
        lines.append(f"- {row['condition']}: {row['note']} Sources: {', '.join(row['source_paths'])}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def make_learning_rate_rows(run_by_path: dict[str, dict]) -> list[dict]:
    rows = []
    for source in LEARNING_RATE_SWEEP_SOURCES:
        runs = [
            run_by_path[path]
            for path in source["eval_paths"]
            if path in run_by_path
        ]
        if not runs:
            continue
        metrics = {}
        for metric in ["orig_acc", "swap_acc", "consistency", "first_pos_rate"]:
            values = [run["metrics"][metric]["value"] for run in runs]
            metrics[metric] = {
                "mean": mean(values),
                "seed_std": stdev(values) if len(values) > 1 else None,
                "values": values,
            }
        rows.append({
            "model": source["model"],
            "learning_rate": source["learning_rate"],
            "learning_rate_value": source["learning_rate_value"],
            "reward": source["reward"],
            "n_runs": len(runs),
            "n_samples": ";".join(str(run["n"]) for run in runs),
            "source_paths": source["eval_paths"],
            "note": source["note"],
            "metrics": metrics,
        })
    return rows


def write_learning_rate_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "model",
        "learning_rate",
        "learning_rate_value",
        "reward",
        "n_runs",
        "n_samples",
        "orig_acc_mean_pct",
        "orig_acc_seed_std_pct",
        "swap_acc_mean_pct",
        "swap_acc_seed_std_pct",
        "consistency_mean_pct",
        "consistency_seed_std_pct",
        "first_pos_rate_mean_pct",
        "first_pos_rate_seed_std_pct",
        "orig_acc_values_pct",
        "consistency_values_pct",
        "first_pos_rate_values_pct",
        "source_paths",
        "note",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metrics = row["metrics"]
            writer.writerow({
                "model": row["model"],
                "learning_rate": row["learning_rate"],
                "learning_rate_value": row["learning_rate_value"],
                "reward": row["reward"],
                "n_runs": row["n_runs"],
                "n_samples": row["n_samples"],
                "orig_acc_mean_pct": pct(metrics["orig_acc"]["mean"]),
                "orig_acc_seed_std_pct": pct(metrics["orig_acc"]["seed_std"]),
                "swap_acc_mean_pct": pct(metrics["swap_acc"]["mean"]),
                "swap_acc_seed_std_pct": pct(metrics["swap_acc"]["seed_std"]),
                "consistency_mean_pct": pct(metrics["consistency"]["mean"]),
                "consistency_seed_std_pct": pct(metrics["consistency"]["seed_std"]),
                "first_pos_rate_mean_pct": pct(metrics["first_pos_rate"]["mean"]),
                "first_pos_rate_seed_std_pct": pct(metrics["first_pos_rate"]["seed_std"]),
                "orig_acc_values_pct": ";".join(f"{pct(value):.6f}" for value in metrics["orig_acc"]["values"]),
                "consistency_values_pct": ";".join(f"{pct(value):.6f}" for value in metrics["consistency"]["values"]),
                "first_pos_rate_values_pct": ";".join(f"{pct(value):.6f}" for value in metrics["first_pos_rate"]["values"]),
                "source_paths": ";".join(row["source_paths"]),
                "note": row["note"],
            })


def fmt_lr_metric(row: dict, metric: str) -> str:
    metric_data = row["metrics"][metric]
    if metric_data["seed_std"] is None:
        return f"{pct(metric_data['mean']):.1f}"
    return f"{pct(metric_data['mean']):.1f} +/- {pct(metric_data['seed_std']):.1f}"


def write_learning_rate_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Learning Rate Sweep Summary",
        "",
        "Generated from local artifacts only. Values are percentages.",
        "Rows are single-seed unless a row lists more than one run.",
        "Qwen2.5 uses the full composite reward for the sweep; Qwen3 uses the accuracy reward for the sweep.",
        "",
    ]
    for model in ["Qwen2.5-7B", "Qwen3-8B"]:
        model_rows = [row for row in rows if row["model"] == model]
        if not model_rows:
            continue
        lines.append(f"## {model}")
        lines.append("")
        lines.append("| LR | Reward | Runs | Acc | Swap Acc | Con | First-pos | Source note |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---|")
        for row in model_rows:
            lines.append(
                "| {lr} | {reward} | {runs} | {acc} | {swap} | {con} | {first} | {note} |".format(
                    lr=row["learning_rate"],
                    reward=row["reward"],
                    runs=row["n_runs"],
                    acc=fmt_lr_metric(row, "orig_acc"),
                    swap=fmt_lr_metric(row, "swap_acc"),
                    con=fmt_lr_metric(row, "consistency"),
                    first=fmt_lr_metric(row, "first_pos_rate"),
                    note=row["note"],
                )
            )
        lines.append("")
    lines.append("## Sources")
    lines.append("")
    for row in rows:
        lines.append(f"- {row['model']} {row['learning_rate']}: {', '.join(row['source_paths'])}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def make_training_dynamics_rows(run_by_path: dict[str, dict]) -> list[dict]:
    rows = []
    for source in TRAINING_DYNAMICS_SOURCES:
        runs = [
            run_by_path[path]
            for path in source["eval_paths"]
            if path in run_by_path
        ]
        if not runs:
            continue
        metrics = {}
        for metric in ["orig_acc", "swap_acc", "consistency", "first_pos_rate"]:
            values = [run["metrics"][metric]["value"] for run in runs]
            metrics[metric] = {
                "mean": mean(values),
                "seed_std": stdev(values) if len(values) > 1 else None,
                "values": values,
            }
        rows.append({
            "variant": source["variant"],
            "reward": source["reward"],
            "steps": source["steps"],
            "n_runs": len(runs),
            "n_samples": ";".join(str(run["n"]) for run in runs),
            "source_paths": source["eval_paths"],
            "include_in_paper": source["include_in_paper"],
            "note": source["note"],
            "metrics": metrics,
        })

    rows.sort(key=lambda row: (row["variant"] != "Baseline", row["variant"], row["steps"]))
    previous_by_variant: dict[str, dict] = {}
    baseline = next((row for row in rows if row["variant"] == "Baseline"), None)
    for row in rows:
        if row["variant"] == "Baseline":
            row["delta_from_previous"] = {"orig_acc": None, "consistency": None, "first_pos_rate": None}
            continue
        previous = previous_by_variant.get(row["variant"], baseline)
        if previous is None:
            row["delta_from_previous"] = {"orig_acc": None, "consistency": None, "first_pos_rate": None}
        else:
            row["delta_from_previous"] = {
                metric: row["metrics"][metric]["mean"] - previous["metrics"][metric]["mean"]
                for metric in ["orig_acc", "consistency", "first_pos_rate"]
            }
        previous_by_variant[row["variant"]] = row
    return rows


def write_training_dynamics_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "variant",
        "reward",
        "steps",
        "include_in_paper",
        "n_runs",
        "n_samples",
        "orig_acc_mean_pct",
        "swap_acc_mean_pct",
        "consistency_mean_pct",
        "first_pos_rate_mean_pct",
        "delta_orig_acc_from_previous_pp",
        "delta_consistency_from_previous_pp",
        "delta_first_pos_from_previous_pp",
        "source_paths",
        "note",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metrics = row["metrics"]
            deltas = row["delta_from_previous"]
            writer.writerow({
                "variant": row["variant"],
                "reward": row["reward"],
                "steps": row["steps"],
                "include_in_paper": row["include_in_paper"],
                "n_runs": row["n_runs"],
                "n_samples": row["n_samples"],
                "orig_acc_mean_pct": pct(metrics["orig_acc"]["mean"]),
                "swap_acc_mean_pct": pct(metrics["swap_acc"]["mean"]),
                "consistency_mean_pct": pct(metrics["consistency"]["mean"]),
                "first_pos_rate_mean_pct": pct(metrics["first_pos_rate"]["mean"]),
                "delta_orig_acc_from_previous_pp": pct(deltas["orig_acc"]),
                "delta_consistency_from_previous_pp": pct(deltas["consistency"]),
                "delta_first_pos_from_previous_pp": pct(deltas["first_pos_rate"]),
                "source_paths": ";".join(row["source_paths"]),
                "note": row["note"],
            })


def fmt_training_metric(row: dict, metric: str) -> str:
    metric_data = row["metrics"][metric]
    if metric_data["seed_std"] is None:
        return f"{pct(metric_data['mean']):.1f}"
    return f"{pct(metric_data['mean']):.1f} +/- {pct(metric_data['seed_std']):.1f}"


def fmt_training_delta(row: dict, metric: str) -> str:
    value = row["delta_from_previous"][metric]
    if value is None:
        return "---"
    return f"{pct(value):+.1f}"


def write_training_dynamics_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Training Dynamics Summary",
        "",
        "Generated from local artifacts only. Values are percentages; deltas are percentage points relative to the previous available checkpoint for the same variant, using the baseline as the first reference.",
        "The paper table uses only checkpoints shared by the acc-only and full-reward trajectories; extra one-sided artifacts are retained here for auditing.",
        "",
        "| Variant | Steps | Paper | Acc | Swap Acc | Con | First-pos | Delta Acc | Delta Con | Delta First-pos | Source note |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {steps} | {paper} | {acc} | {swap} | {con} | {first} | {dacc} | {dcon} | {dfirst} | {note} |".format(
                variant=row["variant"],
                steps=row["steps"],
                paper="yes" if row["include_in_paper"] else "no",
                acc=fmt_training_metric(row, "orig_acc"),
                swap=fmt_training_metric(row, "swap_acc"),
                con=fmt_training_metric(row, "consistency"),
                first=fmt_training_metric(row, "first_pos_rate"),
                dacc=fmt_training_delta(row, "orig_acc"),
                dcon=fmt_training_delta(row, "consistency"),
                dfirst=fmt_training_delta(row, "first_pos_rate"),
                note=row["note"],
            )
        )
    lines.extend([
        "",
        "## Sources",
        "",
    ])
    for row in rows:
        lines.append(f"- {row['variant']} step {row['steps']}: {', '.join(row['source_paths'])}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def make_reward_ablation_rows(run_by_path: dict[str, dict]) -> list[dict]:
    rows = []
    for source in REWARD_ABLATION_SOURCES:
        runs = [
            run_by_path[path]
            for path in source["eval_paths"]
            if path in run_by_path
        ]
        if not runs:
            continue
        metrics = {}
        for metric in ["orig_acc", "swap_acc", "consistency", "first_pos_rate"]:
            values = [run["metrics"][metric]["value"] for run in runs]
            metrics[metric] = {
                "mean": mean(values),
                "seed_std": stdev(values) if len(values) > 1 else None,
                "values": values,
            }
        rows.append({
            "model": source["model"],
            "reward": source["reward"],
            "n_runs": len(runs),
            "n_samples": ";".join(str(run["n"]) for run in runs),
            "source_paths": source["eval_paths"],
            "note": source["note"],
            "metrics": metrics,
        })
    return rows


def write_reward_ablation_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "model",
        "reward",
        "n_runs",
        "n_samples",
        "orig_acc_mean_pct",
        "orig_acc_seed_std_pct",
        "swap_acc_mean_pct",
        "swap_acc_seed_std_pct",
        "consistency_mean_pct",
        "consistency_seed_std_pct",
        "first_pos_rate_mean_pct",
        "first_pos_rate_seed_std_pct",
        "orig_acc_values_pct",
        "consistency_values_pct",
        "first_pos_rate_values_pct",
        "source_paths",
        "note",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metrics = row["metrics"]
            writer.writerow({
                "model": row["model"],
                "reward": row["reward"],
                "n_runs": row["n_runs"],
                "n_samples": row["n_samples"],
                "orig_acc_mean_pct": pct(metrics["orig_acc"]["mean"]),
                "orig_acc_seed_std_pct": pct(metrics["orig_acc"]["seed_std"]),
                "swap_acc_mean_pct": pct(metrics["swap_acc"]["mean"]),
                "swap_acc_seed_std_pct": pct(metrics["swap_acc"]["seed_std"]),
                "consistency_mean_pct": pct(metrics["consistency"]["mean"]),
                "consistency_seed_std_pct": pct(metrics["consistency"]["seed_std"]),
                "first_pos_rate_mean_pct": pct(metrics["first_pos_rate"]["mean"]),
                "first_pos_rate_seed_std_pct": pct(metrics["first_pos_rate"]["seed_std"]),
                "orig_acc_values_pct": ";".join(f"{pct(value):.6f}" for value in metrics["orig_acc"]["values"]),
                "consistency_values_pct": ";".join(f"{pct(value):.6f}" for value in metrics["consistency"]["values"]),
                "first_pos_rate_values_pct": ";".join(f"{pct(value):.6f}" for value in metrics["first_pos_rate"]["values"]),
                "source_paths": ";".join(row["source_paths"]),
                "note": row["note"],
            })


def fmt_reward_metric(row: dict, metric: str) -> str:
    metric_data = row["metrics"][metric]
    if metric_data["seed_std"] is None:
        return f"{pct(metric_data['mean']):.1f}"
    return f"{pct(metric_data['mean']):.1f} +/- {pct(metric_data['seed_std']):.1f}"


def write_reward_ablation_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Reward Ablation Summary",
        "",
        "Generated from local artifacts only. Values are percentages.",
        "Qwen2.5 rows are multi-seed where available; Qwen3 rows are single-seed reward controls.",
        "",
    ]
    for model in ["Qwen2.5-7B", "Qwen3-8B"]:
        model_rows = [row for row in rows if row["model"] == model]
        if not model_rows:
            continue
        lines.append(f"## {model}")
        lines.append("")
        lines.append("| Reward | Runs | Acc | Swap Acc | Con | First-pos | Source note |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for row in model_rows:
            lines.append(
                "| {reward} | {runs} | {acc} | {swap} | {con} | {first} | {note} |".format(
                    reward=row["reward"],
                    runs=row["n_runs"],
                    acc=fmt_reward_metric(row, "orig_acc"),
                    swap=fmt_reward_metric(row, "swap_acc"),
                    con=fmt_reward_metric(row, "consistency"),
                    first=fmt_reward_metric(row, "first_pos_rate"),
                    note=row["note"],
                )
            )
        lines.append("")
    lines.append("## Sources")
    lines.append("")
    for row in rows:
        lines.append(f"- {row['model']} {row['reward']}: {', '.join(row['source_paths'])}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def make_prompt_label_control_rows(run_by_path: dict[str, dict]) -> list[dict]:
    rows = []
    for source in PROMPT_LABEL_CONTROL_SOURCES:
        runs = [
            run_by_path[path]
            for path in source["eval_paths"]
            if path in run_by_path
        ]
        if not runs:
            continue
        metrics = {}
        for metric in [
            "orig_acc",
            "swap_acc",
            "consistency",
            "first_pos_rate",
            "orig_first_rate",
            "parse_fail_rate",
        ]:
            values = [run["metrics"][metric]["value"] for run in runs]
            metrics[metric] = {
                "mean": mean(values),
                "seed_std": stdev(values) if len(values) > 1 else None,
                "values": values,
            }
        rows.append({
            "model": source["model"],
            "control": source["control"],
            "eval_mode": source["eval_mode"],
            "include_in_paper": source["include_in_paper"],
            "n_runs": len(runs),
            "n_samples": ";".join(str(run["n"]) for run in runs),
            "source_paths": source["eval_paths"],
            "note": source["note"],
            "metrics": metrics,
        })
    return rows


def write_prompt_label_control_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "model",
        "control",
        "eval_mode",
        "include_in_paper",
        "n_runs",
        "n_samples",
        "orig_acc_mean_pct",
        "swap_acc_mean_pct",
        "consistency_mean_pct",
        "first_pos_rate_mean_pct",
        "orig_first_rate_mean_pct",
        "parse_fail_rate_mean_pct",
        "orig_acc_values_pct",
        "consistency_values_pct",
        "orig_first_rate_values_pct",
        "parse_fail_rate_values_pct",
        "source_paths",
        "note",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metrics = row["metrics"]
            writer.writerow({
                "model": row["model"],
                "control": row["control"],
                "eval_mode": row["eval_mode"],
                "include_in_paper": row["include_in_paper"],
                "n_runs": row["n_runs"],
                "n_samples": row["n_samples"],
                "orig_acc_mean_pct": pct(metrics["orig_acc"]["mean"]),
                "swap_acc_mean_pct": pct(metrics["swap_acc"]["mean"]),
                "consistency_mean_pct": pct(metrics["consistency"]["mean"]),
                "first_pos_rate_mean_pct": pct(metrics["first_pos_rate"]["mean"]),
                "orig_first_rate_mean_pct": pct(metrics["orig_first_rate"]["mean"]),
                "parse_fail_rate_mean_pct": pct(metrics["parse_fail_rate"]["mean"]),
                "orig_acc_values_pct": ";".join(f"{pct(value):.6f}" for value in metrics["orig_acc"]["values"]),
                "consistency_values_pct": ";".join(f"{pct(value):.6f}" for value in metrics["consistency"]["values"]),
                "orig_first_rate_values_pct": ";".join(f"{pct(value):.6f}" for value in metrics["orig_first_rate"]["values"]),
                "parse_fail_rate_values_pct": ";".join(f"{pct(value):.6f}" for value in metrics["parse_fail_rate"]["values"]),
                "source_paths": ";".join(row["source_paths"]),
                "note": row["note"],
            })


def fmt_control_metric(row: dict, metric: str) -> str:
    metric_data = row["metrics"][metric]
    if metric_data["seed_std"] is None:
        return f"{pct(metric_data['mean']):.1f}"
    return f"{pct(metric_data['mean']):.1f} +/- {pct(metric_data['seed_std']):.1f}"


def write_prompt_label_control_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Prompt and Label Control Summary",
        "",
        "Generated from local artifacts only. Values are percentages.",
        "Reported rows are the interpretable prompt/label controls used in the appendix table.",
        "Qwen3 thinking-enabled diagnostics are retained here because their high parse-failure rates explain why the paper reports disable-thinking controls.",
        "",
    ]
    for include_in_paper, heading in [(True, "Reported Controls"), (False, "Diagnostic Qwen3 Thinking-On Controls")]:
        section_rows = [row for row in rows if row["include_in_paper"] is include_in_paper]
        if not section_rows:
            continue
        lines.append(f"## {heading}")
        lines.append("")
        lines.append("| Model | Control | Eval mode | Acc | Swap Acc | Con | First-pos | Orig-first | Parse fail | Source note |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|")
        for row in section_rows:
            lines.append(
                "| {model} | {control} | {eval_mode} | {acc} | {swap} | {con} | {first} | {orig_first} | {parse} | {note} |".format(
                    model=row["model"],
                    control=row["control"],
                    eval_mode=row["eval_mode"],
                    acc=fmt_control_metric(row, "orig_acc"),
                    swap=fmt_control_metric(row, "swap_acc"),
                    con=fmt_control_metric(row, "consistency"),
                    first=fmt_control_metric(row, "first_pos_rate"),
                    orig_first=fmt_control_metric(row, "orig_first_rate"),
                    parse=fmt_control_metric(row, "parse_fail_rate"),
                    note=row["note"],
                )
            )
        lines.append("")

    lines.append("## Sources")
    lines.append("")
    for row in rows:
        lines.append(f"- {row['model']} {row['control']}: {', '.join(row['source_paths'])}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def posthoc_metrics_for_results(results: list[dict]) -> dict[str, float | int]:
    n = len(results)
    orig_correct = 0
    consistent_count = 0
    consistent_correct = 0
    inconsistent_count = 0
    inconsistent_orig_correct = 0

    for row in results:
        gold = row.get("gold_label")
        pred = row.get("predicted")
        swap_pred = row.get("swap_predicted")
        is_orig_correct = pred == gold
        if is_orig_correct:
            orig_correct += 1

        if "is_consistent" in row:
            is_consistent = bool(row.get("is_consistent"))
        else:
            is_consistent = swap_pred == FLIP.get(pred)

        if is_consistent:
            consistent_count += 1
            if is_orig_correct:
                consistent_correct += 1
        else:
            inconsistent_count += 1
            if is_orig_correct:
                inconsistent_orig_correct += 1

    return {
        "n": n,
        "standard_accuracy": orig_correct / n if n else 0.0,
        "coverage": consistent_count / n if n else 0.0,
        "covered_accuracy": consistent_correct / consistent_count if consistent_count else 0.0,
        "random_fallback_accuracy": (consistent_correct + 0.5 * inconsistent_count) / n if n else 0.0,
        "consistent_pairs": consistent_count,
        "inconsistent_pairs": inconsistent_count,
        "inconsistent_orig_correct_rate": inconsistent_orig_correct / inconsistent_count if inconsistent_count else None,
    }


def make_posthoc_filtering_rows() -> list[dict]:
    rows = []
    for source in POSTHOC_FILTERING_SOURCES:
        run_metrics = []
        present_paths = []
        for path in source["eval_paths"]:
            eval_path = Path(path)
            if not eval_path.exists():
                continue
            run_metrics.append(posthoc_metrics_for_results(load_json(eval_path)))
            present_paths.append(path)
        if not run_metrics:
            continue

        metrics = {}
        for metric in [
            "standard_accuracy",
            "coverage",
            "covered_accuracy",
            "random_fallback_accuracy",
            "inconsistent_orig_correct_rate",
        ]:
            values = [run[metric] for run in run_metrics if run[metric] is not None]
            metrics[metric] = {
                "mean": mean(values) if values else None,
                "seed_std": stdev(values) if len(values) > 1 else None,
                "values": values,
            }

        rows.append({
            "setting": source["setting"],
            "n_runs": len(run_metrics),
            "n_samples": ";".join(str(run["n"]) for run in run_metrics),
            "consistent_pairs": ";".join(str(run["consistent_pairs"]) for run in run_metrics),
            "inconsistent_pairs": ";".join(str(run["inconsistent_pairs"]) for run in run_metrics),
            "source_paths": present_paths,
            "note": source["note"],
            "metrics": metrics,
        })
    return rows


def write_posthoc_filtering_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "setting",
        "n_runs",
        "n_samples",
        "standard_accuracy_mean_pct",
        "standard_accuracy_seed_std_pct",
        "coverage_mean_pct",
        "coverage_seed_std_pct",
        "covered_accuracy_mean_pct",
        "covered_accuracy_seed_std_pct",
        "random_fallback_accuracy_mean_pct",
        "random_fallback_accuracy_seed_std_pct",
        "inconsistent_orig_correct_rate_mean_pct",
        "inconsistent_orig_correct_rate_seed_std_pct",
        "consistent_pairs",
        "inconsistent_pairs",
        "source_paths",
        "note",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metrics = row["metrics"]
            writer.writerow({
                "setting": row["setting"],
                "n_runs": row["n_runs"],
                "n_samples": row["n_samples"],
                "standard_accuracy_mean_pct": pct(metrics["standard_accuracy"]["mean"]),
                "standard_accuracy_seed_std_pct": pct(metrics["standard_accuracy"]["seed_std"]),
                "coverage_mean_pct": pct(metrics["coverage"]["mean"]),
                "coverage_seed_std_pct": pct(metrics["coverage"]["seed_std"]),
                "covered_accuracy_mean_pct": pct(metrics["covered_accuracy"]["mean"]),
                "covered_accuracy_seed_std_pct": pct(metrics["covered_accuracy"]["seed_std"]),
                "random_fallback_accuracy_mean_pct": pct(metrics["random_fallback_accuracy"]["mean"]),
                "random_fallback_accuracy_seed_std_pct": pct(metrics["random_fallback_accuracy"]["seed_std"]),
                "inconsistent_orig_correct_rate_mean_pct": pct(metrics["inconsistent_orig_correct_rate"]["mean"]),
                "inconsistent_orig_correct_rate_seed_std_pct": pct(metrics["inconsistent_orig_correct_rate"]["seed_std"]),
                "consistent_pairs": row["consistent_pairs"],
                "inconsistent_pairs": row["inconsistent_pairs"],
                "source_paths": ";".join(row["source_paths"]),
                "note": row["note"],
            })


def fmt_posthoc_metric(row: dict, metric: str) -> str:
    metric_data = row["metrics"][metric]
    if metric_data["mean"] is None:
        return ""
    if metric_data["seed_std"] is None:
        return f"{pct(metric_data['mean']):.1f}"
    return f"{pct(metric_data['mean']):.1f} +/- {pct(metric_data['seed_std']):.1f}"


def write_posthoc_filtering_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Post-hoc Swap Filtering Summary",
        "",
        "Generated from local eval artifacts only. Values are percentages.",
        "Coverage is the fraction of examples whose original and swapped predictions agree after label flipping.",
        "Covered accuracy is accuracy on that retained subset; random fallback accuracy assumes a random binary choice on inconsistent examples.",
        "",
        "| Setting | Runs | Standard Acc | Coverage | Covered Acc | Random Fallback Acc | Inconsistent Orig-Correct | Source note |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {setting} | {runs} | {standard} | {coverage} | {covered} | {random} | {inconsistent} | {note} |".format(
                setting=row["setting"],
                runs=row["n_runs"],
                standard=fmt_posthoc_metric(row, "standard_accuracy"),
                coverage=fmt_posthoc_metric(row, "coverage"),
                covered=fmt_posthoc_metric(row, "covered_accuracy"),
                random=fmt_posthoc_metric(row, "random_fallback_accuracy"),
                inconsistent=fmt_posthoc_metric(row, "inconsistent_orig_correct_rate"),
                note=row["note"],
            )
        )
    lines.extend([
        "",
        "## Sources",
        "",
    ])
    for row in rows:
        lines.append(f"- {row['setting']}: {', '.join(row['source_paths'])}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def make_length_confound_rows(run_by_path: dict[str, dict]) -> list[dict]:
    rows = []
    for source in LENGTH_CONFOUND_SOURCES:
        runs = [
            run_by_path[path]
            for path in source["eval_paths"]
            if path in run_by_path
        ]
        if not runs:
            continue
        metrics = {}
        for metric in ["orig_acc", "swap_acc", "consistency", "first_pos_rate", "tie_rate"]:
            values = [run["metrics"][metric]["value"] for run in runs]
            metrics[metric] = {
                "mean": mean(values),
                "seed_std": stdev(values) if len(values) > 1 else None,
                "values": values,
            }
        rows.append({
            "setting": source["setting"],
            "confound": source["confound"],
            "n_runs": len(runs),
            "n_samples": ";".join(str(run["n"]) for run in runs),
            "source_paths": source["eval_paths"],
            "note": source["note"],
            "metrics": metrics,
        })
    return rows


def write_length_confound_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "setting",
        "confound",
        "n_runs",
        "n_samples",
        "orig_acc_mean_pct",
        "orig_acc_seed_std_pct",
        "swap_acc_mean_pct",
        "swap_acc_seed_std_pct",
        "consistency_mean_pct",
        "consistency_seed_std_pct",
        "first_pos_rate_mean_pct",
        "first_pos_rate_seed_std_pct",
        "tie_rate_mean_pct",
        "tie_rate_seed_std_pct",
        "orig_acc_values_pct",
        "consistency_values_pct",
        "first_pos_rate_values_pct",
        "source_paths",
        "note",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metrics = row["metrics"]
            writer.writerow({
                "setting": row["setting"],
                "confound": row["confound"],
                "n_runs": row["n_runs"],
                "n_samples": row["n_samples"],
                "orig_acc_mean_pct": pct(metrics["orig_acc"]["mean"]),
                "orig_acc_seed_std_pct": pct(metrics["orig_acc"]["seed_std"]),
                "swap_acc_mean_pct": pct(metrics["swap_acc"]["mean"]),
                "swap_acc_seed_std_pct": pct(metrics["swap_acc"]["seed_std"]),
                "consistency_mean_pct": pct(metrics["consistency"]["mean"]),
                "consistency_seed_std_pct": pct(metrics["consistency"]["seed_std"]),
                "first_pos_rate_mean_pct": pct(metrics["first_pos_rate"]["mean"]),
                "first_pos_rate_seed_std_pct": pct(metrics["first_pos_rate"]["seed_std"]),
                "tie_rate_mean_pct": pct(metrics["tie_rate"]["mean"]),
                "tie_rate_seed_std_pct": pct(metrics["tie_rate"]["seed_std"]),
                "orig_acc_values_pct": ";".join(f"{pct(value):.6f}" for value in metrics["orig_acc"]["values"]),
                "consistency_values_pct": ";".join(f"{pct(value):.6f}" for value in metrics["consistency"]["values"]),
                "first_pos_rate_values_pct": ";".join(f"{pct(value):.6f}" for value in metrics["first_pos_rate"]["values"]),
                "source_paths": ";".join(row["source_paths"]),
                "note": row["note"],
            })


def fmt_length_metric(row: dict, metric: str) -> str:
    metric_data = row["metrics"][metric]
    if metric_data["seed_std"] is None:
        return f"{pct(metric_data['mean']):.1f}"
    return f"{pct(metric_data['mean']):.1f} +/- {pct(metric_data['seed_std']):.1f}"


def write_length_confound_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Length Confound Control Summary",
        "",
        "Generated from local eval artifacts only. Values are percentages.",
        "The length-confounded control is included to test whether a non-positional artifact produces the same position-shortcut collapse.",
        "",
        "| Setting | Runs | Confound | Acc | Swap Acc | Con | First-pos | Tie | Source note |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {setting} | {runs} | {confound} | {acc} | {swap} | {con} | {first} | {tie} | {note} |".format(
                setting=row["setting"],
                runs=row["n_runs"],
                confound=row["confound"],
                acc=fmt_length_metric(row, "orig_acc"),
                swap=fmt_length_metric(row, "swap_acc"),
                con=fmt_length_metric(row, "consistency"),
                first=fmt_length_metric(row, "first_pos_rate"),
                tie=fmt_length_metric(row, "tie_rate"),
                note=row["note"],
            )
        )
    lines.extend([
        "",
        "## Sources",
        "",
    ])
    for row in rows:
        lines.append(f"- {row['setting']}: {', '.join(row['source_paths'])}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def make_external_judge_rows(run_by_path: dict[str, dict]) -> list[dict]:
    rows = []
    for source in EXTERNAL_JUDGE_SOURCES:
        runs = [
            run_by_path[path]
            for path in source["eval_paths"]
            if path in run_by_path
        ]
        if not runs:
            continue
        metrics = {}
        for metric in ["orig_acc", "swap_acc", "consistency", "first_pos_rate", "position_bias"]:
            values = [run["metrics"][metric]["value"] for run in runs]
            metrics[metric] = {
                "mean": mean(values),
                "seed_std": stdev(values) if len(values) > 1 else None,
                "values": values,
            }
            if len(runs) == 1:
                run_metric = runs[0]["metrics"][metric]
                metrics[metric]["ci95_low"] = run_metric.get("ci95_low")
                metrics[metric]["ci95_high"] = run_metric.get("ci95_high")
        rows.append({
            "model": source["model"],
            "checkpoint": source["checkpoint"],
            "n_runs": len(runs),
            "n_samples": ";".join(str(run["n"]) for run in runs),
            "source_paths": source["eval_paths"],
            "note": source["note"],
            "metrics": metrics,
            "orig_swap_gap": metrics["orig_acc"]["mean"] - metrics["swap_acc"]["mean"],
        })
    return rows


def fmt_external_judge_metric(row: dict, metric: str, include_ci: bool = False) -> str:
    metric_data = row["metrics"][metric]
    value = metric_data["mean"]
    if include_ci and metric_data.get("ci95_low") is not None:
        return (
            f"{pct(value):.1f} "
            f"[{pct(metric_data['ci95_low']):.1f}, {pct(metric_data['ci95_high']):.1f}]"
        )
    if metric_data["seed_std"] is None:
        return f"{pct(value):.1f}"
    return f"{pct(value):.1f} +/- {pct(metric_data['seed_std']):.1f}"


def write_external_judge_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "model",
        "checkpoint",
        "n_runs",
        "n_samples",
        "orig_acc_mean_pct",
        "orig_acc_ci95_low_pct",
        "orig_acc_ci95_high_pct",
        "swap_acc_mean_pct",
        "swap_acc_ci95_low_pct",
        "swap_acc_ci95_high_pct",
        "orig_swap_gap_pp",
        "consistency_mean_pct",
        "consistency_ci95_low_pct",
        "consistency_ci95_high_pct",
        "first_pos_rate_mean_pct",
        "first_pos_rate_ci95_low_pct",
        "first_pos_rate_ci95_high_pct",
        "position_bias_pct",
        "source_paths",
        "note",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metrics = row["metrics"]
            writer.writerow({
                "model": row["model"],
                "checkpoint": row["checkpoint"],
                "n_runs": row["n_runs"],
                "n_samples": row["n_samples"],
                "orig_acc_mean_pct": pct(metrics["orig_acc"]["mean"]),
                "orig_acc_ci95_low_pct": pct(metrics["orig_acc"].get("ci95_low")),
                "orig_acc_ci95_high_pct": pct(metrics["orig_acc"].get("ci95_high")),
                "swap_acc_mean_pct": pct(metrics["swap_acc"]["mean"]),
                "swap_acc_ci95_low_pct": pct(metrics["swap_acc"].get("ci95_low")),
                "swap_acc_ci95_high_pct": pct(metrics["swap_acc"].get("ci95_high")),
                "orig_swap_gap_pp": pct(row["orig_swap_gap"]),
                "consistency_mean_pct": pct(metrics["consistency"]["mean"]),
                "consistency_ci95_low_pct": pct(metrics["consistency"].get("ci95_low")),
                "consistency_ci95_high_pct": pct(metrics["consistency"].get("ci95_high")),
                "first_pos_rate_mean_pct": pct(metrics["first_pos_rate"]["mean"]),
                "first_pos_rate_ci95_low_pct": pct(metrics["first_pos_rate"].get("ci95_low")),
                "first_pos_rate_ci95_high_pct": pct(metrics["first_pos_rate"].get("ci95_high")),
                "position_bias_pct": pct(metrics["position_bias"]["mean"]),
                "source_paths": ";".join(row["source_paths"]),
                "note": row["note"],
            })


def write_external_judge_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# External Judge Summary",
        "",
        "Generated from local eval artifacts only. Values are percentages.",
        "Gap is original-order accuracy minus swapped-order accuracy; Bias is always-first minus always-second inconsistent rate.",
        "",
        "| Model | Runs | Orig Acc | Swap Acc | Gap | Con | First-pos | Bias | Source note |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {runs} | {orig} | {swap} | {gap:.1f} | {con} | {first} | {bias} | {note} |".format(
                model=row["model"],
                runs=row["n_runs"],
                orig=fmt_external_judge_metric(row, "orig_acc", include_ci=True),
                swap=fmt_external_judge_metric(row, "swap_acc", include_ci=True),
                gap=pct(row["orig_swap_gap"]),
                con=fmt_external_judge_metric(row, "consistency", include_ci=True),
                first=fmt_external_judge_metric(row, "first_pos_rate", include_ci=True),
                bias=fmt_external_judge_metric(row, "position_bias"),
                note=row["note"],
            )
        )
    lines.extend([
        "",
        "## Sources",
        "",
    ])
    for row in rows:
        lines.append(f"- {row['model']}: {', '.join(row['source_paths'])}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def make_group_summary(group: dict, run_by_name: dict[str, dict]) -> dict:
    present = [run_by_name[name] for name in group["runs"] if name in run_by_name]
    missing = [name for name in group["runs"] if name not in run_by_name]
    summary = {
        "section": group["section"],
        "name": group["name"],
        "requested_runs": group["runs"],
        "missing_runs": missing,
        "n_runs": len(present),
        "runs": [run["run_name"] for run in present],
        "metrics": {},
    }
    for metric_name in METRIC_ORDER:
        values = [run["metrics"][metric_name]["value"] for run in present if metric_name in run["metrics"]]
        if not values:
            continue
        summary["metrics"][metric_name] = {
            "mean": mean(values),
            "seed_std": stdev(values) if len(values) > 1 else None,
            "values": values,
        }
        if len(values) == 1:
            metric = present[0]["metrics"][metric_name]
            summary["metrics"][metric_name]["ci95_low"] = metric.get("ci95_low")
            summary["metrics"][metric_name]["ci95_high"] = metric.get("ci95_high")
    return summary


def write_per_run_csv(path: Path, runs: list[dict]) -> None:
    fieldnames = ["run_name", "path", "n"]
    for metric in METRIC_ORDER:
        fieldnames.extend([
            f"{metric}_pct",
            f"{metric}_ci95_low_pct",
            f"{metric}_ci95_high_pct",
        ])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            row = {
                "run_name": run["run_name"],
                "path": run["path"],
                "n": run["n"],
            }
            for metric in METRIC_ORDER:
                metric_data = run["metrics"].get(metric, {})
                row[f"{metric}_pct"] = pct(metric_data.get("value"))
                row[f"{metric}_ci95_low_pct"] = pct(metric_data.get("ci95_low"))
                row[f"{metric}_ci95_high_pct"] = pct(metric_data.get("ci95_high"))
            writer.writerow(row)


def write_group_csv(path: Path, groups: list[dict]) -> None:
    fieldnames = ["section", "name", "n_runs", "runs", "missing_runs"]
    for metric in METRIC_ORDER:
        fieldnames.extend([
            f"{metric}_mean_pct",
            f"{metric}_seed_std_pct",
            f"{metric}_ci95_low_pct",
            f"{metric}_ci95_high_pct",
        ])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for group in groups:
            row = {
                "section": group["section"],
                "name": group["name"],
                "n_runs": group["n_runs"],
                "runs": ";".join(group["runs"]),
                "missing_runs": ";".join(group["missing_runs"]),
            }
            for metric in METRIC_ORDER:
                metric_data = group["metrics"].get(metric, {})
                row[f"{metric}_mean_pct"] = pct(metric_data.get("mean"))
                row[f"{metric}_seed_std_pct"] = pct(metric_data.get("seed_std"))
                row[f"{metric}_ci95_low_pct"] = pct(metric_data.get("ci95_low"))
                row[f"{metric}_ci95_high_pct"] = pct(metric_data.get("ci95_high"))
            writer.writerow(row)


def write_main_table(path: Path, groups: list[dict], run_by_name: dict[str, dict]) -> None:
    lines = []
    lines.append("# Local Result Summary for Main Table Draft")
    lines.append("")
    lines.append("Values are percentages. Multi-run groups are formatted as mean +/- seed std.")
    lines.append("Single-run groups are formatted as value [Wilson 95% CI].")
    lines.append("")
    for section in ["primary_unbalanced", "primary_balanced", "primary_lr", "external"]:
        section_groups = [g for g in groups if g["section"] == section]
        if not section_groups:
            continue
        lines.append(f"## {section}")
        lines.append("")
        lines.append("| Method | Runs | Orig Acc | Swap Acc | Avg Acc | Con | First-pos | Bias |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for group in section_groups:
            metric_values = []
            for run_name in group["runs"]:
                if run_name in run_by_name:
                    metric_values.append(run_by_name[run_name]["metrics"])
            def cells(metric_name: str) -> str:
                return fmt_group_metric([m[metric_name] for m in metric_values], metric_name)
            lines.append(
                "| {name} | {n_runs} | {orig_acc} | {swap_acc} | {order_avg} | {consistency} | {first_pos} | {bias} |".format(
                    name=group["name"],
                    n_runs=group["n_runs"],
                    orig_acc=cells("orig_acc"),
                    swap_acc=cells("swap_acc"),
                    order_avg=cells("order_avg_acc"),
                    consistency=cells("consistency"),
                    first_pos=cells("first_pos_rate"),
                    bias=cells("position_bias"),
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def write_readme(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Local Statistics",
                "",
                "Generated by `scripts/summarize_existing_results.py` from local `results/**/*.json` files.",
                "No models are loaded and no external datasets are downloaded.",
                "",
                "Files:",
                "- `per_run_metrics.csv`: all discovered runs with recomputed metrics and Wilson 95% intervals.",
                "- `selected_group_summary.csv`: curated paper-facing groups with seed means and seed standard deviations.",
                "- `selected_group_summary.json`: machine-readable version of the curated groups.",
                "- `main_table_candidate.md`: compact table draft for the main paper.",
                "- `metrics_integrity_check.csv`: differences between recomputed eval_results metrics and stored metrics.json files.",
                "- `duplicate_run_names.csv`: eval result paths that collapse to the same run name.",
                "- `cross_model_source_of_truth.csv`: canonical sources for the cross-model table.",
                "- `cross_model_source_of_truth.md`: human-readable cross-model source audit.",
                "- `accuracy_decomposition_summary.csv`: cross-model apparent/genuine/shortcut gain decomposition.",
                "- `accuracy_decomposition_summary.md`: human-readable accuracy decomposition summary for paper writing.",
                "- `uncertainty_checks.csv`: paired-instance and seed-mean bootstrap checks.",
                "- `uncertainty_checks.md`: human-readable uncertainty summary for paper writing.",
                "- `domain_slice_summary.csv`: per-domain Qwen2.5 baseline and GRPO diagnostics.",
                "- `domain_slice_summary.md`: human-readable domain-slice summary for paper writing.",
                "- `dose_response_summary.csv`: confound-ratio and non-duplicated balanced diagnostics.",
                "- `dose_response_summary.md`: human-readable dose-response and duplication-control summary.",
                "- `learning_rate_sweep_summary.csv`: Qwen2.5/Qwen3 learning-rate sweep diagnostics.",
                "- `learning_rate_sweep_summary.md`: human-readable learning-rate summary for paper writing.",
                "- `training_dynamics_summary.csv`: checkpoint-level Qwen2.5 training dynamics diagnostics.",
                "- `training_dynamics_summary.md`: human-readable training dynamics summary for paper writing.",
                "- `reward_ablation_summary.csv`: cross-model reward-component ablation diagnostics.",
                "- `reward_ablation_summary.md`: human-readable reward ablation summary for paper writing.",
                "- `prompt_label_control_summary.csv`: prompt and label-control diagnostics.",
                "- `prompt_label_control_summary.md`: human-readable prompt and label-control summary for paper writing.",
                "- `posthoc_filtering_summary.csv`: post-hoc swap-filtering diagnostics.",
                "- `posthoc_filtering_summary.md`: human-readable post-hoc filtering summary for paper writing.",
                "- `length_confound_summary.csv`: length-confound control diagnostics.",
                "- `length_confound_summary.md`: human-readable length-confound summary for paper writing.",
                "- `external_judge_summary.csv`: public JudgeLRM checkpoint diagnostics.",
                "- `external_judge_summary.md`: human-readable public-judge summary for paper writing.",
                "",
                "Metric notes:",
                "- `Orig Acc`: accuracy on the original ordering.",
                "- `Swap Acc`: accuracy after flipping the gold label for the swapped ordering.",
                "- `Con`: position-swap consistency.",
                "- `Strict Con`: position-swap consistency after treating tie/tie pairs as inconsistent.",
                "- `First-pos`: fraction of predictions selecting the first-listed response across both orderings.",
                "- `Bias`: always-first inconsistent rate minus always-second inconsistent rate.",
            ]
        )
        + "\n"
    )


def find_stored_metrics_path(eval_results_path: Path) -> Path | None:
    candidates = [
        eval_results_path.with_name("metrics.json"),
        eval_results_path.parent.parent / "metrics.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def write_integrity_check(path: Path, runs: list[dict]) -> None:
    fieldnames = [
        "run_name",
        "eval_results_path",
        "metrics_path",
        "stored_n",
        "recomputed_n",
        "stored_accuracy_pct",
        "recomputed_orig_acc_pct",
        "accuracy_diff_pp",
        "stored_consistency_pct",
        "recomputed_consistency_pct",
        "consistency_diff_pp",
        "status",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            eval_path = Path(run["path"])
            metrics_path = find_stored_metrics_path(eval_path)
            if metrics_path is None:
                writer.writerow({
                    "run_name": run["run_name"],
                    "eval_results_path": str(eval_path),
                    "metrics_path": "",
                    "recomputed_n": run["n"],
                    "recomputed_orig_acc_pct": pct(run["metrics"]["orig_acc"]["value"]),
                    "recomputed_consistency_pct": pct(run["metrics"]["consistency"]["value"]),
                    "status": "missing_metrics_json",
                })
                continue
            try:
                stored = load_json(metrics_path)
            except Exception as exc:
                writer.writerow({
                    "run_name": run["run_name"],
                    "eval_results_path": str(eval_path),
                    "metrics_path": str(metrics_path),
                    "recomputed_n": run["n"],
                    "status": f"could_not_read_metrics_json:{exc}",
                })
                continue
            stored_acc = stored.get("accuracy")
            stored_con = stored.get("consistency")
            stored_n = stored.get("n_samples")
            recomputed_acc = run["metrics"]["orig_acc"]["value"]
            recomputed_con = run["metrics"]["consistency"]["value"]
            acc_diff = None if stored_acc is None else 100.0 * (recomputed_acc - stored_acc)
            con_diff = None if stored_con is None else 100.0 * (recomputed_con - stored_con)
            n_diff = stored_n is not None and stored_n != run["n"]
            mismatch = (
                not n_diff
                and (
                    (acc_diff is not None and abs(acc_diff) > 0.05)
                    or (con_diff is not None and abs(con_diff) > 0.05)
                )
            )
            writer.writerow({
                "run_name": run["run_name"],
                "eval_results_path": str(eval_path),
                "metrics_path": str(metrics_path),
                "stored_n": stored_n,
                "recomputed_n": run["n"],
                "stored_accuracy_pct": pct(stored_acc),
                "recomputed_orig_acc_pct": pct(recomputed_acc),
                "accuracy_diff_pp": acc_diff,
                "stored_consistency_pct": pct(stored_con),
                "recomputed_consistency_pct": pct(recomputed_con),
                "consistency_diff_pp": con_diff,
                "status": "stored_n_diff" if n_diff else "mismatch" if mismatch else "ok",
            })


def write_duplicate_run_names(path: Path, runs: list[dict]) -> None:
    by_name: dict[str, list[dict]] = {}
    for run in runs:
        by_name.setdefault(run["run_name"], []).append(run)

    fieldnames = [
        "run_name",
        "n_variants",
        "path",
        "n",
        "orig_acc_pct",
        "consistency_pct",
        "parse_fail_rate_pct",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run_name, variants in sorted(by_name.items()):
            if len(variants) <= 1:
                continue
            for run in variants:
                writer.writerow({
                    "run_name": run_name,
                    "n_variants": len(variants),
                    "path": run["path"],
                    "n": run["n"],
                    "orig_acc_pct": pct(run["metrics"]["orig_acc"]["value"]),
                    "consistency_pct": pct(run["metrics"]["consistency"]["value"]),
                    "parse_fail_rate_pct": pct(run["metrics"]["parse_fail_rate"]["value"]),
                })


def load_metrics_json_summary(path: Path) -> dict:
    stored = load_json(path)
    n = stored.get("n_samples", 0)
    parse_failures = (stored.get("parse_failures_orig") or 0) + (stored.get("parse_failures_swap") or 0)
    parse_den = 2 * n if n else 0
    return {
        "path": str(path),
        "n": n,
        "metrics": {
            "orig_acc": {
                "value": stored.get("accuracy"),
                "ci95_low": None,
                "ci95_high": None,
            },
            "consistency": {
                "value": stored.get("consistency"),
                "ci95_low": None,
                "ci95_high": None,
            },
            "parse_fail_rate": {
                "value": parse_failures / parse_den if parse_den else None,
                "ci95_low": None,
                "ci95_high": None,
            },
        },
    }


def summarize_cross_model_source(source: dict, run_by_path: dict[str, dict]) -> dict:
    summaries = []
    missing = []
    source_paths = []
    source_kind = "eval_results"

    for path_str in source.get("eval_paths", []):
        path = Path(path_str)
        source_paths.append(path_str)
        run = run_by_path.get(str(path))
        if run is None:
            missing.append(path_str)
        else:
            summaries.append(run)

    for path_str in source.get("metrics_paths", []):
        path = Path(path_str)
        source_paths.append(path_str)
        source_kind = "metrics_json"
        if not path.exists():
            missing.append(path_str)
        else:
            summaries.append(load_metrics_json_summary(path))

    row = {
        "model": source["model"],
        "method": source["method"],
        "data": source["data"],
        "source_kind": source_kind,
        "source_paths": source_paths,
        "missing_sources": missing,
        "n_runs": len(summaries),
        "n_samples": ";".join(str(s["n"]) for s in summaries),
        "note": source.get("note", ""),
        "metrics": {},
    }
    for metric in ["orig_acc", "consistency", "parse_fail_rate"]:
        values = [
            summary["metrics"][metric]["value"]
            for summary in summaries
            if metric in summary["metrics"] and summary["metrics"][metric]["value"] is not None
        ]
        row["metrics"][metric] = {
            "mean": mean(values) if values else None,
            "seed_std": stdev(values) if len(values) > 1 else None,
            "values": values,
        }
    return row


def write_cross_model_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "model",
        "method",
        "data",
        "source_kind",
        "n_runs",
        "n_samples",
        "orig_acc_mean_pct",
        "orig_acc_seed_std_pct",
        "consistency_mean_pct",
        "consistency_seed_std_pct",
        "parse_fail_rate_mean_pct",
        "missing_sources",
        "source_paths",
        "note",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metrics = row["metrics"]
            writer.writerow({
                "model": row["model"],
                "method": row["method"],
                "data": row["data"],
                "source_kind": row["source_kind"],
                "n_runs": row["n_runs"],
                "n_samples": row["n_samples"],
                "orig_acc_mean_pct": pct(metrics["orig_acc"]["mean"]),
                "orig_acc_seed_std_pct": pct(metrics["orig_acc"]["seed_std"]),
                "consistency_mean_pct": pct(metrics["consistency"]["mean"]),
                "consistency_seed_std_pct": pct(metrics["consistency"]["seed_std"]),
                "parse_fail_rate_mean_pct": pct(metrics["parse_fail_rate"]["mean"]),
                "missing_sources": ";".join(row["missing_sources"]),
                "source_paths": ";".join(row["source_paths"]),
                "note": row["note"],
            })


def write_cross_model_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Cross-Model Source of Truth",
        "",
        "Generated from local artifacts only. Multi-run values are mean +/- sample standard deviation.",
        "Rows marked `metrics_json` are sourced from stored metrics because the full eval_results artifact is unavailable or partial.",
        "",
    ]

    for model in ["Qwen2.5-7B", "Qwen3-8B", "Mistral-7B"]:
        model_rows = [row for row in rows if row["model"] == model]
        if not model_rows:
            continue
        lines.append(f"## {model}")
        lines.append("")
        lines.append("| Method | Data | Runs | Acc | Con | Parse fail | Source |")
        lines.append("|---|---|---:|---:|---:|---:|---|")
        for row in model_rows:
            metrics = row["metrics"]
            acc = fmt_mean_std(metrics["orig_acc"]["values"])
            con = fmt_mean_std(metrics["consistency"]["values"])
            parse = fmt_mean_std(metrics["parse_fail_rate"]["values"])
            sources = "<br>".join(row["source_paths"])
            lines.append(
                f"| {row['method']} | {row['data']} | {row['n_runs']} | {acc} | {con} | {parse} | {sources} |"
            )
        lines.append("")

    notes = [row for row in rows if row["note"] or row["missing_sources"]]
    if notes:
        lines.append("## Notes")
        lines.append("")
        for row in notes:
            note = row["note"]
            if row["missing_sources"]:
                note = (note + " " if note else "") + "Missing: " + "; ".join(row["missing_sources"])
            lines.append(f"- {row['model']} {row['method']} {row['data']}: {note}")
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


def make_accuracy_decomposition_rows(cross_model_rows: list[dict]) -> list[dict]:
    rows = []
    by_key = {
        (row["model"], row["method"], row["data"]): row
        for row in cross_model_rows
    }
    for model in ["Qwen2.5-7B", "Qwen3-8B", "Mistral-7B"]:
        baseline = by_key.get((model, "Baseline", "---"))
        unbalanced = by_key.get((model, "GRPO", "unbal"))
        balanced = by_key.get((model, "GRPO", "bal"))
        if not baseline or not unbalanced or not balanced:
            continue
        baseline_acc = baseline["metrics"]["orig_acc"]["mean"]
        unbalanced_acc = unbalanced["metrics"]["orig_acc"]["mean"]
        balanced_acc = balanced["metrics"]["orig_acc"]["mean"]
        if baseline_acc is None or unbalanced_acc is None or balanced_acc is None:
            continue
        rows.append({
            "model": model,
            "baseline_acc": baseline_acc,
            "unbalanced_acc": unbalanced_acc,
            "balanced_acc": balanced_acc,
            "apparent_gain": unbalanced_acc - baseline_acc,
            "genuine_gain": balanced_acc - baseline_acc,
            "shortcut_gain": unbalanced_acc - balanced_acc,
            "source_paths": (
                baseline["source_paths"]
                + unbalanced["source_paths"]
                + balanced["source_paths"]
            ),
            "note": "Derived from canonical cross-model original-order accuracy rows.",
        })
    return rows


def fmt_signed_pp(value: float) -> str:
    return f"{pct(value):+.1f}"


def write_accuracy_decomposition_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "model",
        "baseline_acc_pct",
        "unbalanced_grpo_acc_pct",
        "balanced_grpo_acc_pct",
        "apparent_gain_pp",
        "genuine_gain_pp",
        "shortcut_gain_pp",
        "source_paths",
        "note",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "model": row["model"],
                "baseline_acc_pct": pct(row["baseline_acc"]),
                "unbalanced_grpo_acc_pct": pct(row["unbalanced_acc"]),
                "balanced_grpo_acc_pct": pct(row["balanced_acc"]),
                "apparent_gain_pp": pct(row["apparent_gain"]),
                "genuine_gain_pp": pct(row["genuine_gain"]),
                "shortcut_gain_pp": pct(row["shortcut_gain"]),
                "source_paths": ";".join(row["source_paths"]),
                "note": row["note"],
            })


def write_accuracy_decomposition_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Accuracy Decomposition Summary",
        "",
        "Generated from canonical cross-model rows. Values are original-order accuracy percentages.",
        "Genuine = balanced GRPO - baseline; Shortcut = unbalanced GRPO - balanced GRPO.",
        "",
        "| Model | Baseline | Unbal GRPO | Bal GRPO | Apparent gain | Genuine | Shortcut |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {baseline:.1f} | {unbal:.1f} | {bal:.1f} | {app} | {genuine} | {shortcut} |".format(
                model=row["model"],
                baseline=pct(row["baseline_acc"]),
                unbal=pct(row["unbalanced_acc"]),
                bal=pct(row["balanced_acc"]),
                app=fmt_signed_pp(row["apparent_gain"]),
                genuine=fmt_signed_pp(row["genuine_gain"]),
                shortcut=fmt_signed_pp(row["shortcut_gain"]),
            )
        )
    lines.extend([
        "",
        "## Sources",
        "",
    ])
    for row in rows:
        lines.append(f"- {row['model']}: {', '.join(row['source_paths'])}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="results/local_stats")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for path in discover_eval_files(results_dir):
        try:
            data = load_json(path)
            if not isinstance(data, list):
                continue
            run_summary = summarize_run(data)
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
            continue
        run_name = infer_run_name(path, results_dir)
        run_summary.update({
            "run_name": run_name,
            "path": str(path),
        })
        runs.append(run_summary)

    run_by_name = {run["run_name"]: run for run in runs}
    run_by_path = {run["path"]: run for run in runs}
    groups = [make_group_summary(group, run_by_name) for group in CURATED_GROUPS]
    cross_model_rows = [
        summarize_cross_model_source(source, run_by_path)
        for source in CROSS_MODEL_SOURCES
    ]
    accuracy_decomposition_rows = make_accuracy_decomposition_rows(cross_model_rows)
    uncertainty_rows = (
        make_paired_uncertainty_rows(results_dir)
        + make_seed_uncertainty_rows(run_by_path)
    )
    domain_slice_rows = make_domain_slice_rows()
    dose_response_rows = make_dose_response_rows(run_by_path)
    learning_rate_rows = make_learning_rate_rows(run_by_path)
    training_dynamics_rows = make_training_dynamics_rows(run_by_path)
    reward_ablation_rows = make_reward_ablation_rows(run_by_path)
    prompt_label_control_rows = make_prompt_label_control_rows(run_by_path)
    posthoc_filtering_rows = make_posthoc_filtering_rows()
    length_confound_rows = make_length_confound_rows(run_by_path)
    external_judge_rows = make_external_judge_rows(run_by_path)

    write_per_run_csv(out_dir / "per_run_metrics.csv", runs)
    write_group_csv(out_dir / "selected_group_summary.csv", groups)
    (out_dir / "selected_group_summary.json").write_text(json.dumps(groups, indent=2))
    write_main_table(out_dir / "main_table_candidate.md", groups, run_by_name)
    write_integrity_check(out_dir / "metrics_integrity_check.csv", runs)
    write_duplicate_run_names(out_dir / "duplicate_run_names.csv", runs)
    write_cross_model_csv(out_dir / "cross_model_source_of_truth.csv", cross_model_rows)
    write_cross_model_markdown(out_dir / "cross_model_source_of_truth.md", cross_model_rows)
    write_accuracy_decomposition_csv(out_dir / "accuracy_decomposition_summary.csv", accuracy_decomposition_rows)
    write_accuracy_decomposition_markdown(out_dir / "accuracy_decomposition_summary.md", accuracy_decomposition_rows)
    write_uncertainty_csv(out_dir / "uncertainty_checks.csv", uncertainty_rows)
    write_uncertainty_markdown(out_dir / "uncertainty_checks.md", uncertainty_rows)
    write_domain_slice_csv(out_dir / "domain_slice_summary.csv", domain_slice_rows)
    write_domain_slice_markdown(out_dir / "domain_slice_summary.md", domain_slice_rows)
    write_dose_response_csv(out_dir / "dose_response_summary.csv", dose_response_rows)
    write_dose_response_markdown(out_dir / "dose_response_summary.md", dose_response_rows)
    write_learning_rate_csv(out_dir / "learning_rate_sweep_summary.csv", learning_rate_rows)
    write_learning_rate_markdown(out_dir / "learning_rate_sweep_summary.md", learning_rate_rows)
    write_training_dynamics_csv(out_dir / "training_dynamics_summary.csv", training_dynamics_rows)
    write_training_dynamics_markdown(out_dir / "training_dynamics_summary.md", training_dynamics_rows)
    write_reward_ablation_csv(out_dir / "reward_ablation_summary.csv", reward_ablation_rows)
    write_reward_ablation_markdown(out_dir / "reward_ablation_summary.md", reward_ablation_rows)
    write_prompt_label_control_csv(out_dir / "prompt_label_control_summary.csv", prompt_label_control_rows)
    write_prompt_label_control_markdown(out_dir / "prompt_label_control_summary.md", prompt_label_control_rows)
    write_posthoc_filtering_csv(out_dir / "posthoc_filtering_summary.csv", posthoc_filtering_rows)
    write_posthoc_filtering_markdown(out_dir / "posthoc_filtering_summary.md", posthoc_filtering_rows)
    write_length_confound_csv(out_dir / "length_confound_summary.csv", length_confound_rows)
    write_length_confound_markdown(out_dir / "length_confound_summary.md", length_confound_rows)
    write_external_judge_csv(out_dir / "external_judge_summary.csv", external_judge_rows)
    write_external_judge_markdown(out_dir / "external_judge_summary.md", external_judge_rows)
    write_readme(out_dir / "README.md")

    print(f"Discovered runs: {len(runs)}")
    print(f"Curated groups: {len(groups)}")
    missing_groups = [g for g in groups if g["missing_runs"]]
    if missing_groups:
        print("Groups with missing runs:")
        for group in missing_groups:
            print(f"  {group['name']}: {', '.join(group['missing_runs'])}")
    missing_cross_model = [row for row in cross_model_rows if row["missing_sources"]]
    if missing_cross_model:
        print("Cross-model rows with missing sources:")
        for row in missing_cross_model:
            print(f"  {row['model']} {row['method']} {row['data']}: {', '.join(row['missing_sources'])}")
    print(f"Wrote: {out_dir}")


if __name__ == "__main__":
    main()
