#!/usr/bin/env python3
"""Create the main-paper evaluator-sensitivity and transfer summary figure."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from paper_figure_style import (
    CORAL,
    CORAL_WASH,
    GRAY,
    GRID,
    INK,
    MUTED,
    SAND,
    TEAL,
    TEAL_DARK,
    TEAL_WASH,
    save_figure,
    set_paper_style,
)


ROOT = Path(__file__).resolve().parents[1]
LOCAL_RESULTS = ROOT / "results" / "local_offline_experiments.json"
TRANSFER_RESULTS = ROOT / "results" / "paired_transfer_effects.csv"
OUTPUT_STEM = ROOT / "figure2_experiment_summary"

def load_transfer_rows() -> list[dict[str, float | str]]:
    with TRANSFER_RESULTS.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in row:
            if key not in {"source_type", "target_type"}:
                row[key] = float(row[key])
    order = {name: index for index, name in enumerate(["EDIT", "LOC", "LOGIC", "PLAN"])}
    return sorted(rows, key=lambda row: (order[row["source_type"]], order[row["target_type"]]))


def errorbar(ax: plt.Axes, item: dict, y: float) -> None:
    mean = item["mean"]
    low, high = item["ci"]
    hollow = item["hollow"]
    ax.errorbar(
        mean,
        y,
        xerr=[[mean - low], [high - mean]],
        fmt=item["marker"],
        ms=5.0,
        markerfacecolor="white" if hollow else item["color"],
        markeredgecolor=item["color"],
        markeredgewidth=1.25,
        color=item["color"],
        ecolor=item["color"],
        elinewidth=1.2,
        capsize=2.1,
        capthick=1.05,
        zorder=3,
    )
    ax.text(
        high + 0.035,
        y,
        f"{mean:+.2f}",
        ha="left",
        va="center",
        fontsize=7.1,
        color=item["color"] if item["color"] != SAND else MUTED,
        fontweight="bold" if item["emphasis"] else "normal",
    )


def evaluator_panel(ax: plt.Axes, results: dict) -> None:
    aggregate = results["proxy_sensitivity"]["aggregate"]
    judge = results["llm_judge_sensitivity"]
    specs = [
        ("Full proxy", aggregate["full"], TEAL_DARK, "o", False, True),
        ("Without relevance", aggregate["no_relevance"], CORAL, "D", True, True),
        ("Without file hit", aggregate["no_file_hit"], SAND, "s", False, False),
        ("Actionable only", aggregate["actionable_only"], GRAY, "^", True, False),
        ("Regex proxy", judge["regex"], TEAL_DARK, "o", False, False),
        ("Blinded LLM judge", judge["llm_judge"], CORAL, "D", True, True),
    ]
    y_positions = [5.15, 4.15, 3.15, 2.15, 0.75, -0.25]

    ax.axhspan(1.55, 5.75, color=TEAL_WASH, zorder=0)
    ax.axhspan(-0.75, 1.35, color=CORAL_WASH, zorder=0)
    ax.axvline(0, color=INK, linewidth=0.9, zorder=1)
    for (label, stat, color, marker, hollow, emphasis), y in zip(specs, y_positions):
        errorbar(
            ax,
            {
                "mean": stat["mean_delta"],
                "ci": stat["ci_95"],
                "color": color,
                "marker": marker,
                "hollow": hollow,
                "emphasis": emphasis,
            },
            y,
        )

    ax.set_yticks(y_positions, [item[0] for item in specs])
    ax.text(
        0.01,
        0.985,
        "Proxy-component ablations  ·  $n=96$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color=MUTED,
    )
    ax.text(
        0.01,
        0.32,
        "Stored judged subset  ·  $n=37$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color=MUTED,
    )
    ax.set_xlim(-0.32, 1.05)
    ax.set_ylim(-0.85, 5.85)
    ax.set_xticks([-0.25, 0, 0.25, 0.50, 0.75, 1.00])
    ax.set_xlabel(r"Paired $\Delta$ vs. control")
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", length=2.8, color="#89959B")
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#89959B")
    ax.spines["bottom"].set_linewidth(0.7)
    ax.text(-0.22, 1.025, "(a)", transform=ax.transAxes, fontsize=9.5, fontweight="bold")


def transfer_panel(ax: plt.Axes, rows: list[dict[str, float | str]]) -> None:
    y_positions = list(reversed(range(len(rows))))
    offset = 0.13
    for index, (row, y) in enumerate(zip(rows, y_positions)):
        if index % 6 < 3:
            ax.axhspan(y - 0.48, y + 0.48, color="#F8FAFA", zorder=0)
        full = row["mean_delta_full"]
        sensitivity = row["mean_delta_no_relevance"]
        ax.errorbar(
            full,
            y + offset,
            xerr=[[full - row["ci_low_full"]], [row["ci_high_full"] - full]],
            fmt="o",
            ms=4.2,
            color=TEAL,
            ecolor=TEAL,
            elinewidth=1.15,
            capsize=2.0,
            capthick=1.0,
            zorder=3,
            label="Full proxy" if index == 0 else None,
        )
        ax.errorbar(
            sensitivity,
            y - offset,
            xerr=[
                [sensitivity - row["ci_low_no_relevance"]],
                [row["ci_high_no_relevance"] - sensitivity],
            ],
            fmt="D",
            ms=3.7,
            markerfacecolor="white",
            markeredgewidth=1.15,
            color=CORAL,
            ecolor=CORAL,
            elinewidth=1.05,
            capsize=2.0,
            capthick=1.0,
            zorder=3,
            label="Without relevance" if index == 0 else None,
        )

    labels = [f"{row['source_type']} $\\rightarrow$ {row['target_type']}" for row in rows]
    ax.set_yticks(y_positions, labels)
    ax.axvline(0, color=INK, linewidth=0.9, zorder=1)
    ax.set_xlabel(r"Paired $\Delta$ vs. same-instance control")
    ax.set_xlim(-2.12, 1.28)
    ax.set_xticks([-2, -1, 0, 1])
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", length=2.8, color="#89959B")
    for boundary in [8.5, 5.5, 2.5]:
        ax.axhline(boundary, color="#C8D1D5", linewidth=0.75, zorder=1)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#89959B")
    ax.spines["bottom"].set_linewidth(0.7)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=2,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.0,
    )
    ax.text(
        0.01,
        1.018,
        "negative means:",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.1,
        color=MUTED,
    )
    ax.text(
        0.31,
        1.018,
        "10/12 full",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.1,
        color=TEAL_DARK,
        fontweight="bold",
    )
    ax.text(
        0.61,
        1.018,
        "11/12 − relevance",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.1,
        color=CORAL,
        fontweight="bold",
    )
    ax.text(-0.25, 1.025, "(b)", transform=ax.transAxes, fontsize=9.5, fontweight="bold")


def main() -> None:
    set_paper_style()
    with LOCAL_RESULTS.open() as handle:
        results = json.load(handle)
    rows = load_transfer_rows()

    fig = plt.figure(figsize=(6.75, 4.05))
    grid = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.55], wspace=0.62)
    evaluator_panel(fig.add_subplot(grid[0, 0]), results)
    transfer_panel(fig.add_subplot(grid[0, 1]), rows)
    fig.subplots_adjust(left=0.16, right=0.985, top=0.95, bottom=0.18)

    save_figure(fig, OUTPUT_STEM)


if __name__ == "__main__":
    main()
