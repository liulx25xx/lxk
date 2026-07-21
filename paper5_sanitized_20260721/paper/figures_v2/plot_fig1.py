"""Figure 1: Dumbbell/dot plot — GRPO standard lr vs best lr across 4 domains.

Claim: Learning rate controls GRPO effectiveness. Standard recipe (lr=5e-7)
barely improves over base, while lr=2e-5 unlocks +7–25% gains.
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
ACCENT_NEUTRAL = "#BAB0AC"  # gray for standard lr
ACCENT_OURS = "#B86F7F"     # paper_blush[0] — muted rose for "ours"
BAR_COLOR = "#D5D1D1"       # light gray for connecting bar

# === Data from Paper 5 main.tex ===
domains = ["Commonsense", "Medicine", "Science", "Math"]
standard_lr = [44.4, 58.8, 71.3, 85.0]   # lr=5e-7
best_lr = [68.8, 66.6, 79.1, 92.2]       # lr=2e-5
base_acc = [43.8, 59.2, 71.1, 84.4]      # base model
deltas = [f"+{b - s:.1f}" for s, b in zip(standard_lr, best_lr)]

# === Create figure ===
fig, ax = plt.subplots(figsize=(3.25, 2.45))

y_pos = np.arange(len(domains))

# Draw connecting bars (dumbbell)
for i, (s, b) in enumerate(zip(standard_lr, best_lr)):
    ax.plot([s, b], [i, i], color=BAR_COLOR, linewidth=5, solid_capstyle='round', zorder=1)

# Draw base accuracy as thin dashed vertical lines
for i, base in enumerate(base_acc):
    ax.axvline(x=base, ymin=(i - 0.15) / len(domains), ymax=(i + 0.15) / len(domains),
               color=PAPER_BLUSH[6], linewidth=0.7, linestyle='--', alpha=0.7, zorder=0)

# Plot standard lr dots (gray)
ax.scatter(standard_lr, y_pos, s=50, color=ACCENT_NEUTRAL, zorder=3, edgecolors='white', linewidths=0.5)

# Plot best lr dots (accent rose)
ax.scatter(best_lr, y_pos, s=50, color=ACCENT_OURS, zorder=3, edgecolors='white', linewidths=0.5)

# Delta annotations
for i, (b, d) in enumerate(zip(best_lr, deltas)):
    ax.annotate(d, (b + 1.0, i), fontsize=7, color=ACCENT_OURS, va='center', fontweight='bold')

# Base labels
for i, base in enumerate(base_acc):
    ax.annotate("base", (base, i - 0.35), fontsize=6, color=PAPER_BLUSH[6], ha='center', alpha=0.8)

# Axis formatting
ax.set_yticks(y_pos)
ax.set_yticklabels(domains)
ax.set_xlabel("Accuracy (%)")
ax.set_xlim(38, 98)
ax.set_ylim(-0.5, len(domains) - 0.5)

# Remove top/right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Light x-grid
ax.grid(True, axis='x', color="#ECEFF2", linewidth=0.52, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(length=2.5, width=0.6)

# Legend above plot
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=ACCENT_NEUTRAL, markersize=6, label='lr=5e-7 (math recipe)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=ACCENT_OURS, markersize=6, label='lr=2e-5 (ours)'),
    Line2D([0], [0], linestyle='--', color=PAPER_BLUSH[6], linewidth=0.7, label='Base'),
]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.14),
          ncol=3, frameon=False, fontsize=7, columnspacing=1.0)

# === Save ===
out_base = Path(__file__).parent / "fig1_dumbbell_v2"
fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=600)
fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
plt.close()
print(f"Saved: {out_base}.pdf and .png")
