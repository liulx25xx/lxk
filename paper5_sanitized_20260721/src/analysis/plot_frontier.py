"""
Generate the "RLVR Benefit Frontier" visualization and supporting plots.

Key figures for the paper:
  1. RLVR Benefit Frontier — heatmap: domain x data_size, color = GRPO - SFT
  2. Domain comparison bar chart — 6 domains x 3 methods
  3. Data scaling curves — accuracy vs data size per method
  4. Difficulty analysis — performance by easy/medium/hard
  5. OOD generalization gap — in-domain vs out-of-domain
  6. Radar chart — multi-domain performance profile
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

DOMAIN_LABELS = {
    "math": "Math",
    "science": "Science",
    "law": "Law",
    "medicine": "Medicine",
    "code": "Code",
    "commonsense": "Commonsense",
}
METHOD_COLORS = {
    "sft": "#2196F3",
    "grpo": "#F44336",
    "dpo": "#4CAF50",
    "base": "#9E9E9E",
}
METHOD_LABELS = {
    "sft": "SFT",
    "grpo": "GRPO (RLVR)",
    "dpo": "DPO",
    "base": "Base Model",
}
DATA_SIZES = [500, 2000, 5000, 20000]
DOMAINS = ["math", "science", "law", "medicine", "code", "commonsense"]


def load_results(results_dir: str) -> dict:
    """
    Load all experiment results into a structured dict.

    Expected directory structure:
      results/{method}/{domain}/{size}/summary.json
      results/{method}/{domain}/difficulty/{level}/summary.json
    """
    results = {}
    results_path = Path(results_dir)

    for method in ("sft", "grpo", "dpo", "base"):
        results[method] = {}
        for domain in DOMAINS:
            results[method][domain] = {}
            for size in DATA_SIZES:
                summary_file = results_path / method / domain / str(size) / "summary.json"
                if summary_file.exists():
                    with open(summary_file) as f:
                        data = json.load(f)
                    results[method][domain][size] = data
                else:
                    results[method][domain][size] = None

            # Difficulty splits
            for diff in ("easy", "medium", "hard"):
                diff_file = results_path / method / domain / "difficulty" / diff / "summary.json"
                if diff_file.exists():
                    with open(diff_file) as f:
                        results[method][domain][f"diff_{diff}"] = json.load(f)

    return results


def _get_accuracy(results: dict, method: str, domain: str, size: int) -> float | None:
    """Extract accuracy from results structure."""
    data = results.get(method, {}).get(domain, {}).get(size)
    if data is None:
        return None
    return data.get("metrics", {}).get("overall", {}).get("accuracy")


# ---------------------------------------------------------------------------
# Figure 1: RLVR Benefit Frontier (heatmap)
# ---------------------------------------------------------------------------

def plot_frontier(results: dict, output_dir: str):
    """
    The main figure: heatmap of GRPO-SFT accuracy difference.
    Rows = domains, Columns = data sizes.
    Green = RLVR wins, Red = SFT wins.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    matrix = np.zeros((len(DOMAINS), len(DATA_SIZES)))
    annotations = np.empty_like(matrix, dtype=object)

    for i, domain in enumerate(DOMAINS):
        for j, size in enumerate(DATA_SIZES):
            sft_acc = _get_accuracy(results, "sft", domain, size)
            grpo_acc = _get_accuracy(results, "grpo", domain, size)
            if sft_acc is not None and grpo_acc is not None:
                diff = (grpo_acc - sft_acc) * 100  # percentage points
                matrix[i, j] = diff
                annotations[i, j] = f"{diff:+.1f}"
            else:
                matrix[i, j] = 0.0
                annotations[i, j] = "—"

    # Diverging colormap centered at 0
    vmax = max(abs(matrix.min()), abs(matrix.max()), 5)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=[f"{s//1000}K" if s >= 1000 else str(s) for s in DATA_SIZES],
        yticklabels=[DOMAIN_LABELS[d] for d in DOMAINS],
        cmap="RdYlGn",
        norm=norm,
        annot=annotations,
        fmt="",
        linewidths=0.5,
        cbar_kws={"label": "GRPO - SFT (% points)", "shrink": 0.8},
    )

    ax.set_xlabel("Training Data Size")
    ax.set_ylabel("Domain")
    ax.set_title("RLVR Benefit Frontier: When Does GRPO Beat SFT?")

    output_path = os.path.join(output_dir, "frontier_heatmap.pdf")
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved frontier heatmap to {output_path}")

    # Also save PNG for quick viewing
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        matrix, ax=ax2,
        xticklabels=[f"{s//1000}K" if s >= 1000 else str(s) for s in DATA_SIZES],
        yticklabels=[DOMAIN_LABELS[d] for d in DOMAINS],
        cmap="RdYlGn", norm=norm, annot=annotations, fmt="", linewidths=0.5,
        cbar_kws={"label": "GRPO - SFT (% points)", "shrink": 0.8},
    )
    ax2.set_xlabel("Training Data Size")
    ax2.set_ylabel("Domain")
    ax2.set_title("RLVR Benefit Frontier: When Does GRPO Beat SFT?")
    fig2.savefig(os.path.join(output_dir, "frontier_heatmap.png"))
    plt.close(fig2)


# ---------------------------------------------------------------------------
# Figure 2: Domain comparison bar chart
# ---------------------------------------------------------------------------

def plot_domain_comparison(results: dict, output_dir: str, size: int = 5000):
    """
    Grouped bar chart: accuracy per domain, grouped by method.
    Uses a fixed data size (default 5K) for the main comparison.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(DOMAINS))
    width = 0.22

    for k, method in enumerate(["base", "sft", "grpo", "dpo"]):
        accs = []
        for domain in DOMAINS:
            if method == "base":
                acc = _get_accuracy(results, "base", domain, size)
            else:
                acc = _get_accuracy(results, method, domain, size)
            accs.append((acc or 0.0) * 100)

        bars = ax.bar(
            x + k * width,
            accs,
            width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.5,
        )
        # Add value labels
        for bar, acc in zip(bars, accs):
            if acc > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{acc:.1f}", ha="center", va="bottom", fontsize=7,
                )

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([DOMAIN_LABELS[d] for d in DOMAINS])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Domain Comparison ({size//1000}K training instances)")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(os.path.join(output_dir, "domain_comparison.pdf"))
    fig.savefig(os.path.join(output_dir, "domain_comparison.png"))
    plt.close(fig)
    logger.info(f"Saved domain comparison to {output_dir}")


# ---------------------------------------------------------------------------
# Figure 3: Data scaling curves
# ---------------------------------------------------------------------------

def plot_data_scaling(results: dict, output_dir: str):
    """
    Line plot: accuracy vs data size, one subplot per domain.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    axes = axes.flatten()

    for i, domain in enumerate(DOMAINS):
        ax = axes[i]
        for method in ["sft", "grpo", "dpo"]:
            accs = []
            sizes_with_data = []
            for size in DATA_SIZES:
                acc = _get_accuracy(results, method, domain, size)
                if acc is not None:
                    accs.append(acc * 100)
                    sizes_with_data.append(size)

            if accs:
                ax.plot(
                    sizes_with_data, accs,
                    marker="o", markersize=5,
                    color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method],
                    linewidth=2,
                )

        ax.set_title(DOMAIN_LABELS[domain])
        ax.set_xscale("log")
        ax.set_xticks(DATA_SIZES)
        ax.set_xticklabels([f"{s//1000}K" if s >= 1000 else str(s) for s in DATA_SIZES])
        ax.grid(alpha=0.3)
        if i >= 3:
            ax.set_xlabel("Training Data Size")
        if i % 3 == 0:
            ax.set_ylabel("Accuracy (%)")

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Data Scaling: How Performance Changes with Training Data Size", y=1.05)
    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, "data_scaling.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "data_scaling.png"), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved data scaling curves to {output_dir}")


# ---------------------------------------------------------------------------
# Figure 4: Difficulty analysis
# ---------------------------------------------------------------------------

def plot_difficulty_analysis(results: dict, output_dir: str):
    """
    Grouped bar chart: accuracy by difficulty level, for each method.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    difficulties = ["easy", "medium", "hard"]

    for i, domain in enumerate(DOMAINS):
        ax = axes[i]
        x = np.arange(len(difficulties))
        width = 0.25

        for k, method in enumerate(["sft", "grpo", "dpo"]):
            accs = []
            for diff in difficulties:
                data = results.get(method, {}).get(domain, {}).get(f"diff_{diff}")
                if data:
                    acc = data.get("metrics", {}).get("overall", {}).get("accuracy", 0)
                else:
                    acc = 0
                accs.append(acc * 100)

            ax.bar(
                x + k * width, accs, width,
                label=METHOD_LABELS[method] if i == 0 else "",
                color=METHOD_COLORS[method],
            )

        ax.set_xticks(x + width)
        ax.set_xticklabels(["Easy", "Medium", "Hard"])
        ax.set_title(DOMAIN_LABELS[domain])
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
        if i % 3 == 0:
            ax.set_ylabel("Accuracy (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Performance by Difficulty Level", y=1.05)
    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, "difficulty_analysis.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "difficulty_analysis.png"), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved difficulty analysis to {output_dir}")


# ---------------------------------------------------------------------------
# Figure 5: Radar chart (multi-domain profile)
# ---------------------------------------------------------------------------

def plot_radar(results: dict, output_dir: str, size: int = 5000):
    """Radar/spider chart showing multi-domain performance profile."""
    angles = np.linspace(0, 2 * np.pi, len(DOMAINS), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for method in ["sft", "grpo", "dpo"]:
        values = []
        for domain in DOMAINS:
            acc = _get_accuracy(results, method, domain, size)
            values.append((acc or 0) * 100)
        values += values[:1]

        ax.plot(angles, values, "o-", linewidth=2,
                color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        ax.fill(angles, values, alpha=0.1, color=METHOD_COLORS[method])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([DOMAIN_LABELS[d] for d in DOMAINS])
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_title(f"Multi-Domain Performance Profile ({size//1000}K)")

    fig.savefig(os.path.join(output_dir, "radar_profile.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "radar_profile.png"), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved radar chart to {output_dir}")


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def generate_all_figures(results_dir: str, output_dir: str):
    """Generate all paper figures."""
    os.makedirs(output_dir, exist_ok=True)
    results = load_results(results_dir)

    plot_frontier(results, output_dir)
    plot_domain_comparison(results, output_dir)
    plot_data_scaling(results, output_dir)
    plot_difficulty_analysis(results, output_dir)
    plot_radar(results, output_dir)

    logger.info(f"\nAll figures saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory containing all experiment results")
    parser.add_argument("--output_dir", type=str, default="figures",
                        help="Output directory for figures")
    args = parser.parse_args()
    generate_all_figures(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
