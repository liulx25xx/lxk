#!/usr/bin/env python3
"""Run CPU-only experiments over the stored per-instance result artifacts.

The script makes no API calls and does not require the original repositories.
It addresses three reviewer risks in the current draft:

1. candidate scaffolds were selected on the evaluation set;
2. the 0--3 proxy contains disclosed-file and type-lexical components;
3. cross-type means were previously compared with unmatched control samples.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEED = 20260715
N_BOOT = 10_000
CONTROL = "CONTROL_no_scaffold"

SELECTED = {
    "EDIT": "EDIT_A_reread_file",
    "PLAN": "PLAN_A_step_back",
    "LOC": "LOC_B_reread_issue",
    "LOGIC": "LOGIC_B_minimal_fix",
}


def load(path: Path):
    with path.open() as handle:
        return json.load(handle)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(q * len(ordered))))
    return ordered[index]


def instance_bootstrap(values: list[float], seed: int) -> list[float]:
    rng = random.Random(seed)
    return [mean(rng.choice(values) for _ in values) for _ in range(N_BOOT)]


def cluster_bootstrap(records: list[dict], seed: int) -> list[float]:
    """Resample projects, retaining all paired instance effects in a project."""
    by_project: dict[str, list[float]] = defaultdict(list)
    for row in records:
        by_project[row["project"]].append(row["delta"])
    projects = sorted(by_project)
    rng = random.Random(seed)
    boot = []
    for _ in range(N_BOOT):
        sampled = [rng.choice(projects) for _ in projects]
        values = [value for project in sampled for value in by_project[project]]
        boot.append(mean(values))
    return boot


def summarize_effects(records: list[dict], seed: int, clustered: bool = True) -> dict:
    values = [row["delta"] for row in records]
    boot = (
        cluster_bootstrap(records, seed)
        if clustered and len({row["project"] for row in records}) > 1
        else instance_bootstrap(values, seed)
    )
    return {
        "n": len(values),
        "n_projects": len({row["project"] for row in records}),
        "mean_delta": mean(values),
        "ci_95": [percentile(boot, 0.025), percentile(boot, 0.975)],
        "bootstrap_unit": "project" if clustered else "instance",
        "resamples": N_BOOT,
    }


def score(eval_result: dict, metric: str) -> int:
    file_hit = int(eval_result["file_hit"])
    actionable = int(eval_result["actionable"])
    relevant = int(eval_result["relevant"])
    return {
        "full": file_hit + actionable + relevant,
        "no_relevance": file_hit + actionable,
        "no_file_hit": actionable + relevant,
        "actionable_only": actionable,
    }[metric]


def project_map(annotations: list[dict]) -> dict[str, str]:
    return {row["instance_id"]: row["project"] for row in annotations}


def phase3_index(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["instance_id"], row["strategy"]): row for row in rows}


def heldout_strategy_selection(phase3: list[dict], projects: dict[str, str]) -> dict:
    """Select a scaffold on other projects and evaluate it on the held-out project."""
    candidates: dict[str, set[str]] = defaultdict(set)
    for row in phase3:
        if row["strategy"] != CONTROL:
            candidates[row["failure_type"]].add(row["strategy"])
    index = phase3_index(phase3)
    instances = sorted({row["instance_id"] for row in phase3})
    instance_type = {
        row["instance_id"]: row["failure_type"] for row in phase3
    }
    folds = sorted({projects[instance_id] for instance_id in instances})
    selected_by_fold: dict[str, dict[str, str]] = {}
    records = []

    for fold_index, heldout_project in enumerate(folds):
        train_ids = [
            instance_id
            for instance_id in instances
            if projects[instance_id] != heldout_project
        ]
        chosen: dict[str, str] = {}
        for failure_type, strategies in sorted(candidates.items()):
            train_type_ids = [
                instance_id
                for instance_id in train_ids
                if instance_type[instance_id] == failure_type
            ]
            means = {
                strategy: mean(
                    index[(instance_id, strategy)]["eval"]["score"]
                    for instance_id in train_type_ids
                )
                for strategy in sorted(strategies)
            }
            chosen[failure_type] = max(
                sorted(means), key=lambda strategy: means[strategy]
            )
        selected_by_fold[heldout_project] = chosen

        for instance_id in instances:
            if projects[instance_id] != heldout_project:
                continue
            failure_type = instance_type[instance_id]
            strategy = chosen[failure_type]
            scaffold_score = index[(instance_id, strategy)]["eval"]["score"]
            control_score = index[(instance_id, CONTROL)]["eval"]["score"]
            records.append(
                {
                    "instance_id": instance_id,
                    "project": heldout_project,
                    "failure_type": failure_type,
                    "strategy": strategy,
                    "scaffold_score": scaffold_score,
                    "control_score": control_score,
                    "delta": scaffold_score - control_score,
                    "fold_index": fold_index,
                }
            )

    selection_counts = {
        failure_type: dict(
            Counter(
                selection[failure_type]
                for selection in selected_by_fold.values()
            )
        )
        for failure_type in sorted(candidates)
    }
    by_type = {}
    for offset, failure_type in enumerate(sorted(candidates)):
        subset = [row for row in records if row["failure_type"] == failure_type]
        by_type[failure_type] = summarize_effects(
            subset, SEED + 10 + offset, clustered=True
        )
    summary = summarize_effects(records, SEED + 1, clustered=True)
    summary.update(
        {
            "mean_scaffold_score": mean(row["scaffold_score"] for row in records),
            "mean_control_score": mean(row["control_score"] for row in records),
            "n_folds": len(folds),
            "selection_counts": selection_counts,
            "by_type": by_type,
        }
    )
    return summary


def proxy_sensitivity(phase3: list[dict], projects: dict[str, str]) -> dict:
    index = phase3_index(phase3)
    output = {}
    aggregate: dict[str, list[dict]] = defaultdict(list)
    for type_offset, (failure_type, strategy) in enumerate(SELECTED.items()):
        instance_ids = sorted(
            row["instance_id"]
            for row in phase3
            if row["failure_type"] == failure_type
            and row["strategy"] == strategy
        )
        output[failure_type] = {}
        for metric_offset, metric in enumerate(
            ["full", "no_relevance", "no_file_hit", "actionable_only"]
        ):
            records = []
            for instance_id in instance_ids:
                scaffold_eval = index[(instance_id, strategy)]["eval"]
                control_eval = index[(instance_id, CONTROL)]["eval"]
                records.append(
                    {
                        "instance_id": instance_id,
                        "project": projects[instance_id],
                        "delta": score(scaffold_eval, metric)
                        - score(control_eval, metric),
                    }
                )
            output[failure_type][metric] = summarize_effects(
                records,
                SEED + 100 + type_offset * 10 + metric_offset,
                clustered=True,
            )
            aggregate[metric].extend(records)

    output["aggregate"] = {
        metric: summarize_effects(
            records, SEED + 200 + offset, clustered=True
        )
        for offset, (metric, records) in enumerate(sorted(aggregate.items()))
    }
    return output


def llm_judge_sensitivity(judged: list[dict], projects: dict[str, str]) -> dict:
    index = {(row["instance_id"], row["strategy"]): row for row in judged}
    records_regex = []
    records_judge = []
    by_type = {}
    for offset, (failure_type, strategy) in enumerate(SELECTED.items()):
        instance_ids = sorted(
            {
                instance_id
                for instance_id, name in index
                if name == strategy
            }
            & {
                instance_id
                for instance_id, name in index
                if name == CONTROL
            }
        )
        type_regex = []
        type_judge = []
        for instance_id in instance_ids:
            scaffold = index[(instance_id, strategy)]
            control = index[(instance_id, CONTROL)]
            base = {
                "instance_id": instance_id,
                "project": projects[instance_id],
            }
            type_regex.append(
                {**base, "delta": scaffold["regex_score"] - control["regex_score"]}
            )
            type_judge.append(
                {**base, "delta": scaffold["llm_score"] - control["llm_score"]}
            )
        records_regex.extend(type_regex)
        records_judge.extend(type_judge)
        by_type[failure_type] = {
            "regex": summarize_effects(
                type_regex, SEED + 300 + offset, clustered=True
            ),
            "llm_judge": summarize_effects(
                type_judge, SEED + 310 + offset, clustered=True
            ),
        }
    return {
        "regex": summarize_effects(records_regex, SEED + 320, clustered=True),
        "llm_judge": summarize_effects(records_judge, SEED + 321, clustered=True),
        "by_type": by_type,
    }


def paired_cross_type(
    phase3: list[dict], cross_type: list[dict], projects: dict[str, str]
) -> tuple[dict, list[dict]]:
    control_eval = {
        (row["instance_id"], row["failure_type"]): row["eval"]
        for row in phase3
        if row["strategy"] == CONTROL
    }
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in cross_type:
        control = control_eval[(row["instance_id"], row["true_type"])]
        for metric in ["full", "no_relevance"]:
            grouped[(row["source_type"], row["true_type"], metric)].append(
                {
                    "instance_id": row["instance_id"],
                    "project": projects[row["instance_id"]],
                    "delta": score(row["eval"], metric) - score(control, metric),
                }
            )

    output = {}
    csv_rows = []
    pairs = sorted({(source, target) for source, target, _ in grouped})
    for pair_index, (source, target) in enumerate(pairs):
        full = summarize_effects(
            grouped[(source, target, "full")],
            SEED + 400 + pair_index,
            clustered=True,
        )
        no_relevance = summarize_effects(
            grouped[(source, target, "no_relevance")],
            SEED + 500 + pair_index,
            clustered=True,
        )
        key = f"{source}->{target}"
        output[key] = {"full": full, "no_relevance": no_relevance}
        csv_rows.append(
            {
                "source_type": source,
                "target_type": target,
                "n": full["n"],
                "n_projects": full["n_projects"],
                "mean_delta_full": full["mean_delta"],
                "ci_low_full": full["ci_95"][0],
                "ci_high_full": full["ci_95"][1],
                "mean_delta_no_relevance": no_relevance["mean_delta"],
                "ci_low_no_relevance": no_relevance["ci_95"][0],
                "ci_high_no_relevance": no_relevance["ci_95"][1],
            }
        )

    output["summary"] = {
        "negative_mean_full": sum(
            row["mean_delta_full"] < 0 for row in csv_rows
        ),
        "negative_ci_full": sum(row["ci_high_full"] < 0 for row in csv_rows),
        "negative_mean_no_relevance": sum(
            row["mean_delta_no_relevance"] < 0 for row in csv_rows
        ),
        "negative_ci_no_relevance": sum(
            row["ci_high_no_relevance"] < 0 for row in csv_rows
        ),
        "n_pairs": len(csv_rows),
    }
    return output, csv_rows


def fmt_effect(item: dict) -> str:
    low, high = item["ci_95"]
    return (
        f"{item['mean_delta']:+.3f} "
        f"[{low:+.3f}, {high:+.3f}] "
        f"(n={item['n']}, projects={item['n_projects']})"
    )


def write_report(summary: dict) -> None:
    heldout = summary["heldout_strategy_selection"]
    proxy = summary["proxy_sensitivity"]
    judge = summary["llm_judge_sensitivity"]
    transfer = summary["paired_cross_type"]
    lines = [
        "# Local Offline Experiments",
        "",
        "Generated entirely from stored per-instance outputs. No API, GPU, or",
        "repository container was used.",
        "",
        "## 1. Leave-one-project-out scaffold selection",
        "",
        (
            f"Across {heldout['n_folds']} project-held-out folds, the selected "
            f"scaffold scores {heldout['mean_scaffold_score']:.3f} versus "
            f"{heldout['mean_control_score']:.3f} for control. The paired delta is "
            f"{fmt_effect(heldout)} using a project-clustered bootstrap."
        ),
        "",
        "Per-category held-out deltas:",
        "",
    ]
    for failure_type, item in sorted(heldout["by_type"].items()):
        lines.append(f"- {failure_type}: {fmt_effect(item)}")
    lines.extend(
        [
            "",
            "Selection counts across project folds:",
            "",
        ]
    )
    for failure_type, counts in sorted(heldout["selection_counts"].items()):
        lines.append(f"- {failure_type}: {counts}")

    lines.extend(
        [
            "",
            "## 2. Proxy-component sensitivity",
            "",
            "Aggregate paired deltas for the manuscript-selected scaffolds:",
            "",
        ]
    )
    metric_names = {
        "full": "Full 0--3 proxy",
        "no_relevance": "Without type-lexical relevance",
        "no_file_hit": "Without disclosed-file hit",
        "actionable_only": "Actionable only",
    }
    for metric in ["full", "no_relevance", "no_file_hit", "actionable_only"]:
        lines.append(f"- {metric_names[metric]}: {fmt_effect(proxy['aggregate'][metric])}")
    lines.extend(
        [
            "",
            "The full-proxy gain disappears when the type-specific relevance term",
            "is removed. This indicates that the large reported matched gains are",
            "primarily lexical/context-following effects, not evidence of repair.",
            "",
            "## 3. Stored LLM-judge sensitivity",
            "",
            f"- Regex proxy on the paired judged subset: {fmt_effect(judge['regex'])}",
            f"- Blinded LLM judge on the same subset: {fmt_effect(judge['llm_judge'])}",
            "",
            "The LLM-judge estimate is substantially smaller and its interval",
            "includes zero; the judged subset is also incomplete and should not be",
            "treated as a replacement primary metric.",
            "",
            "## 4. Paired cross-category transfer",
            "",
            (
                f"Using each instance's own control, {transfer['summary']['negative_mean_full']}"
                f"/{transfer['summary']['n_pairs']} full-proxy means are negative and "
                f"{transfer['summary']['negative_mean_no_relevance']}/"
                f"{transfer['summary']['n_pairs']} remain negative without the relevance term."
            ),
            (
                f"Project-clustered intervals lie fully below zero for "
                f"{transfer['summary']['negative_ci_full']}/"
                f"{transfer['summary']['n_pairs']} full-proxy pairs and "
                f"{transfer['summary']['negative_ci_no_relevance']}/"
                f"{transfer['summary']['n_pairs']} no-relevance pairs."
            ),
            "",
            "This paired analysis supersedes the earlier 9/12 count, which mixed",
            "cross-type subsets with category-wide control means.",
            "",
            "## Interpretation for the paper",
            "",
            "- Supported locally: scaffold effects vary by operational category;",
            "  mismatched scaffolds often reduce the immediate proxy.",
            "- Not supported locally: large matched gains under evaluator-independent",
            "  next-action quality, online routing, or end-to-end repair.",
            "- Still requires new runs: prompts without gold category/files and",
            "  repository-level continuation to test actual recovery.",
            "",
        ]
    )
    (ROOT / "LOCAL_EXPERIMENTS.md").write_text("\n".join(lines))


def main() -> None:
    annotations = load(
        RESULTS / "phase0_annotations" / "phase0_v2_annotations.json"
    )["annotations"]
    phase3 = load(RESULTS / "phase3_full_scaffold" / "full_results.json")[
        "results"
    ]
    cross_type = load(RESULTS / "cross_type_matrix.json")["results"]
    judged = load(RESULTS / "llm_judge_expanded.json")["results"]
    projects = project_map(annotations)

    heldout = heldout_strategy_selection(phase3, projects)
    proxy = proxy_sensitivity(phase3, projects)
    judge = llm_judge_sensitivity(judged, projects)
    paired_transfer, csv_rows = paired_cross_type(phase3, cross_type, projects)

    summary = {
        "metadata": {
            "seed": SEED,
            "bootstrap_resamples": N_BOOT,
            "api_calls": 0,
            "phase3_rows": len(phase3),
            "phase3_instances": len({row["instance_id"] for row in phase3}),
            "cross_type_rows": len(cross_type),
            "llm_judged_rows": len(judged),
        },
        "heldout_strategy_selection": heldout,
        "proxy_sensitivity": proxy,
        "llm_judge_sensitivity": judge,
        "paired_cross_type": paired_transfer,
    }

    json_path = RESULTS / "local_offline_experiments.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    csv_path = RESULTS / "paired_transfer_effects.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    write_report(summary)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {ROOT / 'LOCAL_EXPERIMENTS.md'}")


if __name__ == "__main__":
    main()
