#!/usr/bin/env python3
"""Create the manuscript cross-model scaffold-effect heatmap."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from paper_figure_style import CORAL, INK, NEUTRAL, TEAL, save_figure, set_paper_style


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "cross_model_effects.csv"
OUTPUT_STEM = ROOT / "figure3_cross_model_heatmap"

GRID = "#FFFFFF"


def load_data() -> tuple[list[str], list[str], np.ndarray]:
    with DATA.open() as handle:
        rows = list(csv.DictReader(handle))
    categories = ["EDIT", "LOC", "LOGIC", "PLAN"]
    models = [row["model"] for row in rows]
    tiers = [row["tier"] for row in rows]
    values = np.array([[float(row[category]) for category in categories] for row in rows])
    assert values.shape == (9, 4)
    assert tiers == ["mid"] * 3 + ["strong"] * 3 + ["frontier"] * 3
    return models, categories, values


def main() -> None:
    set_paper_style()
    models, categories, values = load_data()
    cmap = LinearSegmentedColormap.from_list("paper_ocean_diverging", [CORAL, NEUTRAL, TEAL])
    norm = TwoSlopeNorm(vmin=-1.6, vcenter=0, vmax=1.6)

    fig, ax = plt.subplots(figsize=(5.15, 3.0))
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(categories)), [name.title() for name in categories])
    ax.set_yticks(range(len(models)), models)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)

    ax.set_xticks(np.arange(-0.5, len(categories), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.grid(which="minor", color=GRID, linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for boundary in [2.5, 5.5]:
        ax.axhline(boundary, color="#AEB9BE", linewidth=1.0)

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            rgba = cmap(norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text_color = "white" if luminance < 0.58 else INK
            ax.text(
                column,
                row,
                f"{value:+.2f}" if value != 0 else "0.00",
                ha="center",
                va="center",
                fontsize=7.4,
                color=text_color,
                fontweight="bold" if abs(value) >= 1 else "normal",
            )

    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.038, pad=0.045)
    colorbar.set_ticks([-1.6, -0.8, 0, 0.8, 1.6])
    colorbar.set_label(r"Scaffold effect $\Delta$", fontsize=7.6, labelpad=4)
    colorbar.ax.tick_params(labelsize=6.8, length=2)
    colorbar.outline.set_linewidth(0.5)
    colorbar.outline.set_edgecolor("#AEB9BE")

    fig.subplots_adjust(left=0.28, right=0.90, top=0.88, bottom=0.04)
    save_figure(fig, OUTPUT_STEM)


if __name__ == "__main__":
    main()
