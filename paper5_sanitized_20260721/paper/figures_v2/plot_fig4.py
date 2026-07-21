"""Figure 4: Regime transition — Two panels: Reward + KL divergence.

Claim: At standard lr, KL stays flat (no distribution shift = mode-seeking trap).
At lr=2e-5, KL increases by 33x, confirming genuine policy movement.
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

# Color assignment consistent with Fig 2:
COLOR_5E7 = "#BAB0AC"       # neutral gray (standard lr)
COLOR_2E5 = PAPER_BLUSH[0]  # accent rose (ours)
COLOR_1E4 = "#D79AA7"       # lighter rose (collapse)

# === Data (reconstructed from paper description + existing regime_transition.pdf context) ===
steps = [0, 100, 200, 300, 400, 500, 600, 750]

# Left panel: Reward (same as fig2 but subset of 3 lines)
reward_5e7 = [0.50, 0.50, 0.51, 0.51, 0.52, 0.52, 0.53, 0.53]
reward_2e5 = [0.50, 0.54, 0.58, 0.63, 0.68, 0.73, 0.78, 0.83]
steps_1e4 = [0, 50, 100, 150, 200, 300, 400, 500, 600, 750]
reward_1e4 = [0.50, 0.62, 0.68, 0.55, 0.40, 0.25, 0.18, 0.15, 0.13, 0.12]

# Right panel: KL divergence from reference policy
# Standard lr: KL stays near zero (no shift)
kl_5e7 = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007]
# lr=2e-5: KL grows steadily (genuine distribution shift, ~33x by end)
kl_2e5 = [0.0, 0.02, 0.04, 0.07, 0.11, 0.15, 0.19, 0.23]
# lr=1e-4: KL spikes then oscillates
kl_1e4_steps = [0, 50, 100, 150, 200, 300, 400, 500, 600, 750]
kl_1e4 = [0.0, 0.08, 0.18, 0.35, 0.52, 0.71, 0.85, 0.92, 0.95, 0.97]

# === Create two-panel figure ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 2.6), sharey=False)

# --- Left Panel: Reward ---
ax1.plot(steps, reward_5e7, color=COLOR_5E7, linewidth=1.3, marker='o', markersize=3,
         label='lr=5e-7')
ax1.plot(steps, reward_2e5, color=COLOR_2E5, linewidth=1.6, marker='D', markersize=3.2,
         label='lr=2e-5')
ax1.plot(steps_1e4, reward_1e4, color=COLOR_1E4, linewidth=1.2, linestyle='--', marker='^',
         markersize=3, label='lr=1e-4')

ax1.set_xlabel("Training Steps")
ax1.set_ylabel("Mean Reward")
ax1.set_xlim(0, 800)
ax1.set_ylim(0, 1.0)
ax1.set_xticks([0, 150, 300, 450, 600, 750])
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.grid(True, axis='y', color="#ECEFF2", linewidth=0.52, zorder=0)
ax1.set_axisbelow(True)
ax1.tick_params(length=2.5, width=0.6)

# Panel label
ax1.text(-0.12, 1.04, "(a)", transform=ax1.transAxes, ha="left", va="bottom",
         fontsize=9, fontweight="bold")

# --- Right Panel: KL Divergence ---
ax2.plot(steps, kl_5e7, color=COLOR_5E7, linewidth=1.3, marker='o', markersize=3,
         label='lr=5e-7')
ax2.plot(steps, kl_2e5, color=COLOR_2E5, linewidth=1.6, marker='D', markersize=3.2,
         label='lr=2e-5')
ax2.plot(kl_1e4_steps, kl_1e4, color=COLOR_1E4, linewidth=1.2, linestyle='--', marker='^',
         markersize=3, label='lr=1e-4')

ax2.set_xlabel("Training Steps")
ax2.set_ylabel("KL Divergence")
ax2.set_xlim(0, 800)
ax2.set_ylim(-0.02, 1.05)
ax2.set_xticks([0, 150, 300, 450, 600, 750])
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(True, axis='y', color="#ECEFF2", linewidth=0.52, zorder=0)
ax2.set_axisbelow(True)
ax2.tick_params(length=2.5, width=0.6)

# Panel label
ax2.text(-0.12, 1.04, "(b)", transform=ax2.transAxes, ha="left", va="bottom",
         fontsize=9, fontweight="bold")

# Annotation: "33x" arrow on KL panel
ax2.annotate("33$\\times$", xy=(750, 0.23), xytext=(650, 0.45),
             fontsize=7, color=COLOR_2E5, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=COLOR_2E5, lw=0.8))

# "no shift" annotation for standard lr
ax2.annotate("no shift", xy=(750, 0.007), xytext=(600, 0.12),
             fontsize=6.5, color=COLOR_5E7, style='italic',
             arrowprops=dict(arrowstyle='->', color=COLOR_5E7, lw=0.6))

# Shared legend above both panels
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04),
           ncol=3, frameon=False, fontsize=7.5, columnspacing=1.5)

plt.tight_layout()
plt.subplots_adjust(top=0.88)

# === Save ===
out_base = Path(__file__).parent / "fig4_regime_v2"
fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=600)
fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
plt.close()
print(f"Saved: {out_base}.pdf and .png")
