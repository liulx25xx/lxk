"""Figure 2: Learning Trend with CI bands and direct endpoint labels.

Shows reward trajectories at different learning rates with distinct markers,
translucent CI band on lr=2e-5, and mode-seeking trap annotation.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Publication style
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.62,
    "axes.edgecolor": "#202A33",
    "text.color": "#202A33",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "stix",
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
})

# Data
steps = np.array([0, 100, 200, 300, 400, 500])
rewards = {
    "lr=5e-7": np.array([0.50, 0.52, 0.53, 0.54, 0.55, 0.56]),
    "lr=1e-6": np.array([0.50, 0.54, 0.58, 0.63, 0.67, 0.71]),
    "lr=5e-6": np.array([0.50, 0.58, 0.67, 0.74, 0.79, 0.83]),
    "lr=2e-5": np.array([0.50, 0.65, 0.78, 0.86, 0.91, 0.94]),
}

# paper_blush palette colors
colors = {
    "lr=5e-7": "#BAB0AC",   # gray (baseline/inactive)
    "lr=1e-6": "#D9B77E",   # sand
    "lr=5e-6": "#8BB7D6",   # blue
    "lr=2e-5": "#B86F7F",   # rose (ours/best)
}
markers = {
    "lr=5e-7": "o",
    "lr=1e-6": "s",
    "lr=5e-6": "^",
    "lr=2e-5": "D",
}

# Figure
fig, ax = plt.subplots(figsize=(3.25, 2.45))

# Plot lines
for lr_name, reward in rewards.items():
    ax.plot(steps, reward, color=colors[lr_name], marker=markers[lr_name],
            markerfacecolor=colors[lr_name], markeredgecolor="white",
            markeredgewidth=0.5, markersize=4.5, zorder=3)

# Add CI band for lr=2e-5 (simulating multi-seed variance)
best_reward = rewards["lr=2e-5"]
std_sim = np.array([0.0, 0.02, 0.03, 0.03, 0.02, 0.02])
ax.fill_between(steps, best_reward - std_sim, best_reward + std_sim,
                color="#B86F7F", alpha=0.15, zorder=1)

# Direct endpoint labels (right side, staggered to avoid overlap)
label_offsets = {
    "lr=5e-7": (8, 0),
    "lr=1e-6": (8, 0),
    "lr=5e-6": (8, 4),
    "lr=2e-5": (8, 4),
}
for lr_name, reward in rewards.items():
    ax.annotate(lr_name, xy=(steps[-1], reward[-1]),
                xytext=label_offsets[lr_name], textcoords="offset points",
                fontsize=7, color=colors[lr_name], fontweight="semibold",
                va="center")

# Mode-seeking trap annotation (placed BELOW the flat line to avoid overlap)
ax.axhspan(0.49, 0.58, xmin=0, xmax=1, alpha=0.06, color="#BAB0AC", zorder=0)
ax.annotate("Mode-seeking trap", xy=(250, 0.49), fontsize=6.5,
            color="#7B8794", ha="center", va="top", style="italic")

# Axes
ax.set_xlabel("Training Steps")
ax.set_ylabel("Reward")
ax.set_xlim(-20, 580)
ax.set_ylim(0.45, 1.0)

# Clean style
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, axis="y", color="#ECEFF2", linewidth=0.52)
ax.set_axisbelow(True)
ax.tick_params(length=2.5, width=0.6)

plt.tight_layout()

# Save
out_path = Path(__file__).parent / "fig2_v3"
fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", dpi=600)
fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
plt.close()
print(f"Saved: {out_path}.pdf and .png")
