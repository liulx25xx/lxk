"""Build an evidence-only audit table and figure from saved evaluation JSON.

This script intentionally does not reproduce the paper's hand-entered numbers.
Every plotted point is loaded from an explicit JSON result file.  The resulting
figure is an audit artifact, not yet the camera-ready main figure: learning-rate
and SFT-data-size configurations in the current manuscript were selected using
test performance and must be re-selected on a validation split before submission.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "eval_results"
OUT_DIR = Path(__file__).resolve().parent


# Explicit inventory.  Keeping this list visible prevents silent inclusion of
# exploratory or known-broken runs.
RUNS = {
    "Math": {
        "SFT": [
            "math_sft_seed42_n100.json",
            "math_sft_seed123_n100.json",
            "math_sft_seed456_n100.json",
            "math_sft_seed789_n100.json",
        ],
        "GRPO-5e-7": [
            "math_grpo_FIXED_seed42_gsm8k_n2000.json",
            "math_grpo_FIXED_seed123_gsm8k_n2000.json",
        ],
        "GRPO-high": [
            "math_grpo_FIXED_seed42_gsm8k_lr2e5.json",
            "math_grpo_FIXED_seed123_gsm8k_n2000_lr2e5.json",
            "math_grpo_FIXED_seed456_gsm8k_n2000_lr2e5.json",
            "math_grpo_FIXED_seed789_gsm8k_n2000_lr2e5.json",
        ],
    },
    "Science": {
        "SFT": [
            "science_sft_seed42_n5000.json",
            "science_sft_seed123_n5000.json",
            "science_sft_seed456_n5000.json",
            "science_sft_seed789_n5000.json",
        ],
        "GRPO-5e-7": [
            "science_grpo_FIXED_seed42_n2000.json",
            "science_grpo_FIXED_seed123_n2000.json",
        ],
        "GRPO-high": [
            "science_grpo_FIXED_seed42_n2000_lr2e5.json",
            "science_grpo_FIXED_seed123_n2000_lr2e5.json",
            "science_grpo_FIXED_seed456_n2000_lr2e5.json",
            "science_grpo_FIXED_seed789_n2000_lr2e5.json",
        ],
    },
    "Medicine": {
        "SFT": [
            "medicine_sft_seed42_n100.json",
            "medicine_sft_seed123_n100.json",
            "medicine_sft_seed456_n100.json",
            "medicine_sft_seed789_n100.json",
        ],
        "GRPO-5e-7": [
            "medicine_grpo_FIXED_seed42_n2000.json",
            "medicine_grpo_FIXED_seed123_n2000.json",
        ],
        "GRPO-high": [
            "medicine_grpo_FIXED_seed42_n2000_lr2e5.json",
            "medicine_grpo_FIXED_seed123_n2000_lr2e5.json",
            "medicine_grpo_FIXED_seed456_n2000_lr2e5.json",
            "medicine_grpo_FIXED_seed789_n2000_lr2e5.json",
        ],
    },
    "Law": {
        "SFT": [
            "law_sft_seed42.json",
            "law_sft_seed123.json",
            "law_sft_seed456.json",
        ],
        "GRPO-5e-7": ["law_grpo_FIXED_seed42.json"],
        "GRPO-high": [
            "law_grpo_FIXED_seed42_lr5e6.json",
            "law_grpo_FIXED_seed123_lr5e6.json",
            "law_grpo_FIXED_seed456_lr5e6.json",
            "law_grpo_FIXED_seed789_lr5e6.json",
        ],
    },
    "Commonsense": {
        "SFT": [
            "commonsense_sft_seed42.json",
            "commonsense_sft_seed123.json",
            "commonsense_sft_seed456.json",
            "commonsense_sft_seed789.json",
        ],
        "GRPO-5e-7": ["commonsense_grpo_FIXED_seed42_n2000.json"],
        "GRPO-high": [
            "commonsense_grpo_FIXED_seed42_n2000_lr2e5.json",
            "commonsense_grpo_FIXED_seed123_n2000_lr2e5.json",
            "commonsense_grpo_FIXED_seed456_n2000_lr2e5.json",
        ],
    },
}


BASE = {
    "Math": ("math", "gsm8k"),
    "Science": ("science", None),
    "Medicine": ("medicine", None),
    "Law": ("law", None),
    "Commonsense": ("commonsense", None),
}


def seed_from_name(name: str) -> int | None:
    match = re.search(r"seed(\d+)", name)
    return int(match.group(1)) if match else None


def load_base() -> dict[str, float]:
    with (RESULTS / "base_summary.json").open() as handle:
        summary = json.load(handle)
    values = {}
    for display, (domain, subdomain) in BASE.items():
        record = summary[domain]
        if subdomain is not None:
            record = record["subdomains"][subdomain]
        values[display] = 100.0 * float(record["accuracy"])
    return values


def load_rows() -> list[dict]:
    rows = []
    for domain, methods in RUNS.items():
        for method, filenames in methods.items():
            for filename in filenames:
                path = RESULTS / "trained" / filename
                if not path.exists():
                    raise FileNotFoundError(path)
                with path.open() as handle:
                    record = json.load(handle)
                seed_tag = seed_from_name(filename)
                seed_metadata = record.get("seed")
                rows.append(
                    {
                        "domain": domain,
                        "method": method,
                        "run_name": record.get("run_name", path.stem),
                        "seed_tag": seed_tag,
                        "seed_metadata": seed_metadata,
                        "seed_metadata_matches_tag": seed_tag == seed_metadata,
                        "n_train": record.get("n_train"),
                        "n_test": record.get("n_test"),
                        "accuracy_pct": 100.0 * float(record["accuracy"]),
                        "source_json": str(path.relative_to(ROOT)),
                    }
                )
    return rows


def write_csv(rows: list[dict], base: dict[str, float]) -> None:
    fields = list(rows[0]) + ["base_accuracy_pct", "delta_from_base_pp"]
    with (OUT_DIR / "canonical_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            enriched = dict(row)
            enriched["base_accuracy_pct"] = base[row["domain"]]
            enriched["delta_from_base_pp"] = row["accuracy_pct"] - base[row["domain"]]
            writer.writerow(enriched)

    summary_fields = [
        "domain",
        "method",
        "n_runs",
        "mean_accuracy_pct",
        "sd_accuracy_pct",
        "base_accuracy_pct",
        "mean_delta_from_base_pp",
        "seed_metadata_mismatches",
    ]
    with (OUT_DIR / "canonical_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for domain in RUNS:
            for method in ["SFT", "GRPO-5e-7", "GRPO-high"]:
                group = [
                    row for row in rows
                    if row["domain"] == domain and row["method"] == method
                ]
                values = np.array([row["accuracy_pct"] for row in group])
                writer.writerow(
                    {
                        "domain": domain,
                        "method": method,
                        "n_runs": len(group),
                        "mean_accuracy_pct": float(values.mean()),
                        "sd_accuracy_pct": (
                            float(values.std(ddof=1)) if len(values) > 1 else ""
                        ),
                        "base_accuracy_pct": base[domain],
                        "mean_delta_from_base_pp": float(values.mean() - base[domain]),
                        "seed_metadata_mismatches": sum(
                            not row["seed_metadata_matches_tag"] for row in group
                        ),
                    }
                )


def plot(rows: list[dict], base: dict[str, float]) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {
        "SFT": "#9A91CF",
        "GRPO-5e-7": "#8CCDD9",
        "GRPO-high": "#5EADC2",
    }
    markers = {"SFT": "o", "GRPO-5e-7": "s", "GRPO-high": "D"}
    order = ["SFT", "GRPO-5e-7", "GRPO-high"]
    labels = ["SFT\n(reported cfg)", "GRPO\n5e-7", "GRPO\nhigh LR"]

    fig, axes = plt.subplots(2, 3, figsize=(6.75, 3.65), sharey=True)
    axes = axes.ravel()
    rng = np.random.default_rng(17)

    for ax, domain in zip(axes, RUNS):
        domain_rows = [row for row in rows if row["domain"] == domain]
        for x, method in enumerate(order):
            values = np.array(
                [
                    row["accuracy_pct"] - base[domain]
                    for row in domain_rows
                    if row["method"] == method
                ]
            )
            jitter = rng.uniform(-0.09, 0.09, size=len(values))
            ax.scatter(
                x + jitter,
                values,
                s=20,
                marker=markers[method],
                color=colors[method],
                edgecolor="white",
                linewidth=0.45,
                alpha=0.9,
                zorder=3,
            )
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            ax.errorbar(
                x,
                mean,
                yerr=std if len(values) > 1 else None,
                fmt="_",
                markersize=13,
                markeredgewidth=1.5,
                color="#202A33",
                capsize=2.5,
                linewidth=0.9,
                zorder=4,
            )
            ax.text(x, mean + 1.0, f"{mean:+.1f}", ha="center", va="bottom", fontsize=6.5)

        ax.axhline(0, color="#8C96A0", linewidth=0.8, linestyle="--", zorder=1)
        ax.set_title(domain, pad=3, fontweight="semibold")
        ax.set_xticks(range(len(order)), labels)
        ax.set_xlim(-0.5, 2.5)
        ax.grid(axis="y", color="#E8EDF1", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=2.2, width=0.55)

    axes[-1].axis("off")
    axes[0].set_ylabel("Accuracy change from base (pp)")
    axes[3].set_ylabel("Accuracy change from base (pp)")
    axes[0].set_ylim(-5, 27)
    fig.text(
        0.995,
        0.012,
        "Points: saved runs; bar: mean ± SD. High LR = 2e-5 except Law (5e-6).",
        ha="right",
        fontsize=6.6,
        color="#5A6670",
    )
    fig.tight_layout(w_pad=1.1, h_pad=1.0, rect=(0, 0.035, 1, 1))
    fig.savefig(OUT_DIR / "fig_main_result_audit.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_main_result_audit.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base()
    rows = load_rows()
    write_csv(rows, base)
    plot(rows, base)
    mismatches = sum(not row["seed_metadata_matches_tag"] for row in rows)
    print(f"Wrote {len(rows)} runs; {mismatches} seed-tag/metadata mismatches found.")


if __name__ == "__main__":
    main()
