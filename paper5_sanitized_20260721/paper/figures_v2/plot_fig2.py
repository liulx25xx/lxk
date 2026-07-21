"""Figure 2: Reward trajectories — GRPO training reward at different learning rates.

Claim: At standard lr, reward barely moves (mode-seeking trap). At lr=2e-5,
reward rises steadily (genuine consolidation). At lr=1e-4, reward collapses.
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

# Color assignment:
# lr=5e-7 (flat, uninteresting) → neutral gray
# lr=5e-6 (mild) → muted blue from palette
# lr=2e-5 (ours, key result) → accent rose
# lr=1e-4 (collapse) → muted warm/red, dashed
COLOR_5E7 = "#BAB0AC"       # neutral gray
COLOR_5E6 = PAPER_BLUSH[6]  # muted blue (#8BB7D6)
COLOR_2E5 = PAPER_BLUSH[0]  # accent rose (#B86F7F)
COLOR_1E4 = "#D79AA7"       # lighter rose, dashed for collapse

# === Data from tikz in main.tex ===
steps_normal = [0, 100, 200, 300, 400, 500, 600, 750]

# lr=5e-7: flat
reward_5e7 = [0.50, 0.50, 0.51, 0.51, 0.52, 0.52, 0.53, 0.53]
# lr=5e-6: mild rise
reward_5e6 = [0.50, 0.51, 0.52, 0.52, 0.53, 0.54, 0.54, 0.55]
# lr=2e-5: strong rise
reward_2e5 = [0.50, 0.54, 0.58, 0.63, 0.68, 0.73, 0.78, 0.83]
# lr=1e-4: spike then collapse
steps_1e4 = [0, 50, 100, 150, 200, 300, 400, 500, 600, 750]
reward_1e4 = [0.50, 0.62, 0.68, 0.55, 0.40, 0.25, 0.18, 0.15, 0.13, 0.12]

# === Create figure ===
fig, ax = plt.subplots(figsize=(3.25, 2.45))

# Plot lines with markers
ax.plot(steps_normal, reward_5e7, color=COLOR_5E7, linewidth=1.3, marker='o', markersize=3,
        label='lr=5e-7', zorder=3)
ax.plot(steps_normal, reward_5e6, color=COLOR_5E6, linewidth=1.3, marker='s', markersize=3,
        label='lr=5e-6', zorder=3)
ax.plot(steps_normal, reward_2e5, color=COLOR_2E5, linewidth=1.6, marker='D', markersize=3.2,
        label='lr=2e-5 (ours)', zorder=4)
ax.plot(steps_1e4, reward_1e4, color=COLOR_1E4, linewidth=1.2, linestyle='--', marker='^', markersize=3,
        label='lr=1e-4', zorder=3)

# Direct label at line ends (right side)
ax.annotate("0.83", (750, 0.83), xytext=(5, 0), textcoords='offset points',
            fontsize=6.5, color=COLOR_2E5, va='center', fontweight='bold')
ax.annotate("0.55", (750, 0.55), xytext=(5, 0), textcoords='offset points',
            fontsize=6.5, color=COLOR_5E6, va='center')
ax.annotate("0.53", (750, 0.53), xytext=(5, -6), textcoords='offset points',
            fontsize=6.5, color=COLOR_5E7, va='center')
ax.annotate("collapse", (750, 0.12), xytext=(5, 0), textcoords='offset points',
            fontsize=6, color=COLOR_1E4, va='center', style='italic')

# Axis formatting
ax.set_xlabel("Training Steps")
ax.set_ylabel("Mean Reward")
ax.set_xlim(0, 800)
ax.set_ylim(0, 1.0)
ax.set_xticks([0, 150, 300, 450, 600, 750])

# Remove top/right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Light y-grid
ax.grid(True, axis='y', color="#ECEFF2", linewidth=0.52, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(length=2.5, width=0.6)

# Legend above plot
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=4,
          frameon=False, fontsize=7, columnspacing=0.8)

# === Save ===
out_base = Path(__file__).parent / "fig2_reward_traj_v2"
fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=600)
fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
plt.close()
print(f"Saved: {out_base}.pdf and .png")
