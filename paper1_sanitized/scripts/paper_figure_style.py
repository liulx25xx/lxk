"""Shared visual system for the active manuscript figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl


# Paper Ocean palette. Teal denotes the reported/full condition or positive
# deltas; coral denotes sensitivity, harm, or negative deltas; sand and gray
# are secondary/neutral encodings.
TEAL = "#3C8FA4"
TEAL_DARK = "#2F7688"
CORAL = "#C96F5F"
SAND = "#D8C58D"
GRAY = "#7B8794"
INK = "#202A33"
MUTED = "#5F6B72"
GRID = "#DCE4E7"
NEUTRAL = "#F7F8F7"
TEAL_WASH = "#F1F7F7"
CORAL_WASH = "#FBF5F2"


def set_paper_style() -> None:
    """Apply the shared Times-like, vector-safe manuscript style."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.3,
            "axes.linewidth": 0.7,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "stix",
            "axes.unicode_minus": True,
        }
    )


def save_figure(fig, output_stem: str | Path) -> None:
    """Export the common PDF/SVG/PNG artifact set."""
    output_stem = Path(output_stem)
    for extension, dpi in (("pdf", 600), ("svg", 600), ("png", 450)):
        path = output_stem.with_suffix(f".{extension}")
        fig.savefig(path, bbox_inches="tight", dpi=dpi, facecolor="white")
        print(f"wrote {path}")
