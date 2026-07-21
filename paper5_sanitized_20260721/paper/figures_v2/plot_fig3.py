"""Figure 3: SFT data scaling curves (semi-log x-axis).

Claim: SFT effectiveness is domain-dependent. Math shows sharp inverted-U,
Medicine is flat, Science improves monotonically. Pattern maps to base accuracy.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# === Paper style setup ===
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.62,
    "axes.edgecolor": "#202A33",
    "axes.labelcolor": "#202A33",
    "xtick.color": "#202A33",
    "ytick.color": "#202A33",
    "text.color": "#202A33",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "stix",
    "lines.linewidth": 1.35,
    "lines.markersize": 4.3,
})

# === Paper Blush palette ===
PAPER_BLUSH = ["#B86F7F", "#D79AA7", "#E9B9B2", "#D9B77E", "#B8CFA8", "#95C7BD", "#8BB7D6", "#7C87C2"]

# Color assignment for 3 domains:
COLOR_MATH = PAPER_BLUSH[6]     # blue (#8BB7D6)
COLOR_MED = PAPER_BLUSH[3]      # sand/gold (#D9B77E)
COLOR_SCI = PAPER_BLUSH[5]      # teal/sage (#95C7BD)

# === Data from tikz coordinates in main.tex ===
# Math SFT curve
n_math = [50, 100, 200, 500, 2000, 5000]
acc_math = [87.4, 87.8, 82.4, 78.1, 78.7, 77.1]
base_math = 84.4

# Medicine SFT curve
n_med = [100, 500, 1000, 1500, 2000, 3000, 5000]
acc_med = [59.7, 59.0, 59.1, 60.2, 59.4, 58.8, 58.2]
base_med = 59.2

# Science SFT curve
n_sci = [100, 500, 1000, 2000, 5000]
acc_sci = [72.1, 73.0, 72.6, 73.1, 73.8]
base_sci = 71.1

# === Create figure ===
fig, ax = plt.subplots(figsize=(3.25, 2.45))

# Draw dashed baselines
ax.axhline(y=base_math, color=COLOR_MATH, linewidth=0.6, linestyle='--', alpha=0.5, zorder=1)
ax.axhline(y=base_med, color=COLOR_MED, linewidth=0.6, linestyle='--', alpha=0.5, zorder=1)
ax.axhline(y=base_sci, color=COLOR_SCI, linewidth=0.6, linestyle='--', alpha=0.5, zorder=1)

# Plot lines with markers
ax.plot(n_math, acc_math, color=COLOR_MATH, linewidth=1.4, marker='o', markersize=4,
        label='Math', zorder=3)
ax.plot(n_med, acc_med, color=COLOR_MED, linewidth=1.4, marker='s', markersize=3.8,
        label='Medicine', zorder=3)
ax.plot(n_sci, acc_sci, color=COLOR_SCI, linewidth=1.4, marker='D', markersize=3.5,
        label='Science', zorder=3)

# Base labels on the left
ax.annotate("base", (38, base_math), fontsize=6, color=COLOR_MATH, va='center', alpha=0.7)
ax.annotate("base", (38, base_sci), fontsize=6, color=COLOR_SCI, va='center', alpha=0.7)
ax.annotate("base", (38, base_med), fontsize=6, color=COLOR_MED, va='center', alpha=0.7)

# Axis formatting — log scale x
ax.set_xscale('log')
ax.set_xlabel("Training Examples ($N$)")
ax.set_ylabel("Test Accuracy (%)")
ax.set_xlim(30, 8000)
ax.set_ylim(55, 92)

# Custom x-ticks
ax.set_xticks([50, 100, 200, 500, 1000, 2000, 5000])
ax.set_xticklabels(['50', '100', '200', '500', '1k', '2k', '5k'])
ax.minorticks_off()

# Remove top/right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Light y-grid
ax.grid(True, axis='y', color="#ECEFF2", linewidth=0.52, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(length=2.5, width=0.6)

# Legend
ax.legend(loc='upper right', frameon=False, fontsize=7)

# === Save ===
out_base = Path(__file__).parent / "fig3_sft_scaling_v2"
fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=600)
fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
plt.close()
print(f"Saved: {out_base}.pdf and .png")
