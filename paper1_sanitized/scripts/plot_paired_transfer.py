#!/usr/bin/env python3
"""Create the paired cross-category transfer forest plot for the paper."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from paper_figure_style import CORAL, GRID, INK, TEAL, save_figure, set_paper_style


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "paired_transfer_effects.csv"
OUTPUT_STEM = ROOT / "figure2_paired_transfer"

# paper_ocean with a warm contrast for the sensitivity series.
FULL_COLOR = TEAL
SENSITIVITY_COLOR = CORAL


def load_rows() -> list[dict]:
    with DATA.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in row:
            if key not in {"source_type", "target_type"}:
                row[key] = float(row[key])
    order = {name: index for index, name in enumerate(["EDIT", "LOC", "LOGIC", "PLAN"])}
    return sorted(rows, key=lambda row: (order[row["source_type"]], order[row["target_type"]]))


def main() -> None:
    rows = load_rows()
    set_paper_style()

    fig, ax = plt.subplots(figsize=(3.35, 4.05))
    y_positions = list(reversed(range(len(rows))))
    offset = 0.13

    for index, (row, y) in enumerate(zip(rows, y_positions)):
        full = row["mean_delta_full"]
        full_err = [
            [full - row["ci_low_full"]],
            [row["ci_high_full"] - full],
        ]
        sensitivity = row["mean_delta_no_relevance"]
        sensitivity_err = [
            [sensitivity - row["ci_low_no_relevance"]],
            [row["ci_high_no_relevance"] - sensitivity],
        ]
        ax.errorbar(
            full,
            y + offset,
            xerr=full_err,
            fmt="o",
            ms=4.2,
            color=FULL_COLOR,
            ecolor=FULL_COLOR,
            elinewidth=1.15,
            capsize=2.0,
            capthick=1.0,
            zorder=3,
            label="Full proxy" if index == 0 else None,
        )
        ax.errorbar(
            sensitivity,
            y - offset,
            xerr=sensitivity_err,
            fmt="D",
            ms=3.7,
            markerfacecolor="white",
            markeredgewidth=1.15,
            color=SENSITIVITY_COLOR,
            ecolor=SENSITIVITY_COLOR,
            elinewidth=1.05,
            capsize=2.0,
            capthick=1.0,
            zorder=3,
            label="Without relevance" if index == 0 else None,
        )

    labels = [f"{row['source_type']} $\\rightarrow$ {row['target_type']}" for row in rows]
    ax.set_yticks(y_positions, labels)
    ax.axvline(0, color=INK, linewidth=1.0, zorder=1)
    ax.set_xlabel("Paired $\\Delta$ vs. same-instance control")
    ax.set_xlim(-2.12, 1.28)
    ax.set_xticks([-2, -1, 0, 1])
    ax.grid(axis="x", color=GRID, linewidth=0.65, zorder=0)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", length=3, color="#89959B")

    for boundary in [8.5, 5.5, 2.5]:
        ax.axhline(boundary, color="#C8D1D5", linewidth=0.7, zorder=0)

    ax.text(
        0.015,
        1.014,
        "scaffold source $\\rightarrow$ target category",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        color="#59666C",
    )
    legend = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, -0.18),
        ncol=2,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.0,
    )
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#89959B")
    ax.spines["bottom"].set_linewidth(0.7)
    ax.set_ylim(-0.6, len(rows) - 0.4)

    fig.subplots_adjust(left=0.32, right=0.98, top=0.97, bottom=0.17)
    save_figure(fig, OUTPUT_STEM)


if __name__ == "__main__":
    main()
