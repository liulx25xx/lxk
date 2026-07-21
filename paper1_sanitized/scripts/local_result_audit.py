#!/usr/bin/env python3
"""Recompute manuscript-facing summaries from the stored local JSON results.

This script intentionally makes no API calls and requires only the Python
standard library.  It does not attempt to reproduce the online classifier,
because the repository does not contain its per-instance predictions or the
training script used to create ``online_classifier_results.json``.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(path: Path):
    with path.open() as handle:
        return json.load(handle)


def group_means(rows, keys, value):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(value(row))
    return {
        " / ".join(map(str, key)): {
            "n": len(values),
            "mean": mean(values),
        }
        for key, values in sorted(groups.items())
    }


def paired_percentile_bootstrap(rows, failure_type, strategy, *, seed=20260714):
    """Recompute the manuscript's paired mean difference and 95% interval."""
    selected = {
        (row["instance_id"], row["strategy"]): row["eval"]["score"]
        for row in rows
        if row["failure_type"] == failure_type
    }
    control = "CONTROL_no_scaffold"
    instance_ids = sorted(
        {instance_id for instance_id, name in selected if name == strategy}
        & {instance_id for instance_id, name in selected if name == control}
    )
    differences = [
        selected[(instance_id, strategy)] - selected[(instance_id, control)]
        for instance_id in instance_ids
    ]
    rng = random.Random(seed)
    bootstrap_means = sorted(
        mean(rng.choice(differences) for _ in differences)
        for _ in range(10_000)
    )
    return {
        "n": len(differences),
        "mean_difference": mean(differences),
        "ci_95_percentile": [bootstrap_means[249], bootstrap_means[9749]],
        "resamples": 10_000,
        "seed": seed,
    }


def main() -> None:
    annotations = load(
        RESULTS / "phase0_annotations" / "phase0_v2_annotations.json"
    )["annotations"]
    phase3 = load(RESULTS / "phase3_full_scaffold" / "full_results.json")[
        "results"
    ]
    phase4 = load(RESULTS / "phase4_cascade_selection" / "results.json")[
        "results"
    ]
    cross_type = load(RESULTS / "cross_type_matrix.json")["results"]
    classifier = load(RESULTS / "online_classifier_results.json")
    policy = load(RESULTS / "opd_results.json")

    tail_ratios = [row["cascade"]["waste_ratio"] for row in annotations]
    by_type_tail = defaultdict(list)
    for row in annotations:
        by_type_tail[row["failure_type"]].append(row["cascade"]["waste_ratio"])

    classifier_confusion_from_tex = {
        "EDIT": [25, 3, 0, 0],
        "LOC": [10, 11, 16, 0],
        "LOGIC": [0, 3, 66, 1],
        "PLAN": [0, 2, 6, 0],
    }
    confusion_correct = sum(
        row[index]
        for index, row in enumerate(classifier_confusion_from_tex.values())
    )
    confusion_total = sum(sum(row) for row in classifier_confusion_from_tex.values())

    oracle = policy["scores"]["oracle"]
    predicted_policy = policy["scores"]["policy"]
    universal = policy["scores"]["universal"]
    control = policy["scores"]["control"]

    cross_type_by_pair = group_means(
        cross_type,
        ["source_type", "true_type"],
        lambda row: row["eval"]["score"],
    )
    selected_strategies = {
        "EDIT": "EDIT_A_reread_file",
        "PLAN": "PLAN_A_step_back",
        "LOC": "LOC_B_reread_issue",
        "LOGIC": "LOGIC_B_minimal_fix",
    }

    summary = {
        "taxonomy": {
            "n": len(annotations),
            "counts": dict(sorted(Counter(row["failure_type"] for row in annotations).items())),
        },
        "post_error_tail_ratio": {
            "definition": (
                "fraction of assistant actions at or after the first observable error; "
                "the local implementation does not label step-level progress"
            ),
            "mean": mean(tail_ratios),
            "median": median(tail_ratios),
            "by_type_mean": {
                key: mean(values) for key, values in sorted(by_type_tail.items())
            },
        },
        "phase3_next_action_score": group_means(
            phase3,
            ["failure_type", "strategy"],
            lambda row: row["eval"]["score"],
        ),
        "phase3_selected_paired_bootstrap": {
            failure_type: paired_percentile_bootstrap(
                phase3, failure_type, strategy
            )
            for failure_type, strategy in selected_strategies.items()
        },
        "phase4_routing_score": group_means(
            phase4,
            ["condition"],
            lambda row: row["eval"]["score"],
        ),
        "cross_type_score": cross_type_by_pair,
        "classifier_audit": {
            "summary_file_accuracy": classifier["loo_accuracy"],
            "summary_file_feature_count": classifier["n_features"],
            "manuscript_feature_count": 11,
            "manuscript_confusion_accuracy": confusion_correct / confusion_total,
            "confusion_correct": confusion_correct,
            "confusion_total": confusion_total,
            "reproducible_from_repository": False,
            "reason": (
                "per-instance predictions and the training script for the reported "
                "74.8% model are absent"
            ),
        },
        "policy_advantage_audit": {
            "oracle": oracle,
            "policy": predicted_policy,
            "universal": universal,
            "control": control,
            "fraction_of_oracle_over_control": (
                (predicted_policy - control) / (oracle - control)
            ),
            "fraction_of_oracle_over_universal": (
                (predicted_policy - universal) / (oracle - universal)
            ),
            "stored_fraction": policy["oracle_advantage_captured"],
        },
        "protocol_flags": [
            "scaffolding scripts use the first four recorded turns rather than the annotated first-error prefix",
            "scaffolding prompts include the gold failure type",
            "scaffolding prompts include gold target files",
            "the 0--3 score is a proxy next-action metric, not SWE-bench resolution",
        ],
    }

    output_json = RESULTS / "local_audit_summary.json"
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    md = [
        "# Local Result Audit",
        "",
        "Generated without API calls from the JSON artifacts currently in this repository.",
        "",
        "## Stable local facts",
        "",
        f"- Failure trajectories: {len(annotations)}.",
        f"- Type counts: {summary['taxonomy']['counts']}.",
        (
            "- Mean/median post-error tail ratio: "
            f"{mean(tail_ratios):.3f}/{median(tail_ratios):.3f}."
        ),
        "- Phase-4 routing scores:",
    ]
    for key, item in summary["phase4_routing_score"].items():
        md.append(f"  - {key}: {item['mean']:.4f} (n={item['n']})")
    md.append("- Paired percentile-bootstrap intervals (10,000 resamples):")
    for key, item in summary["phase3_selected_paired_bootstrap"].items():
        low, high = item["ci_95_percentile"]
        md.append(
            f"  - {key}: delta={item['mean_difference']:.4f}, "
            f"95% CI=[{low:.4f}, {high:.4f}] (n={item['n']})"
        )
    md.extend(
        [
            "",
            "## Inconsistencies that must not be presented as settled results",
            "",
            (
                "- The saved classifier summary reports 74.8% with 15 features, "
                "whereas the manuscript says 11 features."
            ),
            (
                "- The manuscript confusion matrix yields "
                f"{confusion_correct}/{confusion_total} = "
                f"{confusion_correct / confusion_total:.1%}, not 74.8%."
            ),
            (
                "- Policy advantage captured is "
                f"{summary['policy_advantage_audit']['fraction_of_oracle_over_control']:.1%} "
                "relative to control and "
                f"{summary['policy_advantage_audit']['fraction_of_oracle_over_universal']:.1%} "
                "relative to the universal strategy."
            ),
            "- The repository cannot reproduce the reported classifier because its training script and predictions are missing.",
            "",
            "## Protocol flags",
            "",
        ]
    )
    md.extend(f"- {flag}." for flag in summary["protocol_flags"])
    md.append("")
    (ROOT / "LOCAL_AUDIT.md").write_text("\n".join(md))

    print(f"wrote {output_json}")
    print(f"wrote {ROOT / 'LOCAL_AUDIT.md'}")


if __name__ == "__main__":
    main()
