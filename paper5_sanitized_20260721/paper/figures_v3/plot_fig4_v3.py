"""Figure 4: Phase Diagram with regime boundaries.

Dual y-axis plot: test accuracy (solid) + KL divergence (dashed).
Three shaded regimes: mode-seeking trap | consolidation zone | collapse.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
lr_values = np.array([5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4])
test_acc = np.array([58.8, 59.1, 60.2, 62.3, 64.8, 66.6, 63.1, 45.2])
kl_div = np.array([0.01, 0.03, 0.08, 0.25, 0.82, 2.45, 8.1, 25.3])

# Colors from paper_blush
color_acc = "#B86F7F"   # Rose for accuracy
color_kl = "#8BB7D6"    # Blue for KL

# Regime boundaries (in log space)
# Mode-seeking: lr <= 2e-6 → log10 index 0-2
# Consolidation: 5e-6 to 2e-5 → log10 index 3-5
# Collapse: >= 5e-5 → log10 index 6-7

# Figure
fig, ax1 = plt.subplots(figsize=(3.25, 2.6))

# X positions in log space
x_pos = np.log10(lr_values)

# Regime shading (using log10 x positions)
# Mode-seeking trap: x <= log10(2e-6) = -5.7
ax1.axvspan(x_pos[0] - 0.15, (x_pos[2] + x_pos[3]) / 2, 
            alpha=0.08, color="#BAB0AC", zorder=0)
# Consolidation zone: log10(5e-6) to log10(2e-5)
ax1.axvspan((x_pos[2] + x_pos[3]) / 2, (x_pos[5] + x_pos[6]) / 2, 
            alpha=0.10, color="#95C7BD", zorder=0)
# Collapse: >= log10(5e-5)
ax1.axvspan((x_pos[5] + x_pos[6]) / 2, x_pos[-1] + 0.15, 
            alpha=0.08, color="#E9B9B2", zorder=0)

# Regime boundary dashed lines
boundary1 = (x_pos[2] + x_pos[3]) / 2
boundary2 = (x_pos[5] + x_pos[6]) / 2
ax1.axvline(boundary1, color="#7B8794", linestyle=":", linewidth=0.8, zorder=2)
ax1.axvline(boundary2, color="#7B8794", linestyle=":", linewidth=0.8, zorder=2)

# Regime labels at top (use figure transform to stay above data)
label_y = 1.02
ax1.text((x_pos[0] + boundary1) / 2, label_y, "Mode-seeking trap",
         transform=ax1.get_xaxis_transform(), ha="center", va="bottom",
         fontsize=6, color="#7B8794", style="italic")
ax1.text((boundary1 + boundary2) / 2, label_y, "Consolidation zone",
         transform=ax1.get_xaxis_transform(), ha="center", va="bottom",
         fontsize=6, color="#52796F", style="italic")
ax1.text((boundary2 + x_pos[-1]) / 2, label_y, "Collapse",
         transform=ax1.get_xaxis_transform(), ha="center", va="bottom",
         fontsize=6, color="#9B5C72", style="italic")

# Plot accuracy (left y-axis)
line1, = ax1.plot(x_pos, test_acc, color=color_acc, marker="D",
                  markerfacecolor=color_acc, markeredgecolor="white",
                  markeredgewidth=0.5, markersize=5, zorder=4)
ax1.set_ylabel("Test Accuracy (%)", color=color_acc, fontsize=8)
ax1.tick_params(axis="y", labelcolor=color_acc)
ax1.set_ylim(40, 72)

# Peak annotation
peak_idx = np.argmax(test_acc)
ax1.plot(x_pos[peak_idx], test_acc[peak_idx], marker="*", color="#B86F7F",
         markersize=12, zorder=5, markeredgecolor="white", markeredgewidth=0.3)
ax1.annotate(f"{test_acc[peak_idx]:.1f}%", 
             xy=(x_pos[peak_idx], test_acc[peak_idx]),
             xytext=(5, 6), textcoords="offset points",
             fontsize=7, color="#B86F7F", fontweight="bold")

# KL divergence (right y-axis)
ax2 = ax1.twinx()
line2, = ax2.plot(x_pos, kl_div, color=color_kl, marker="s",
                  markerfacecolor=color_kl, markeredgecolor="white",
                  markeredgewidth=0.5, markersize=4, linestyle="--",
                  linewidth=1.2, zorder=3)
ax2.set_ylabel("KL Divergence", color=color_kl, fontsize=8)
ax2.tick_params(axis="y", labelcolor=color_kl)
ax2.set_yscale("log")
ax2.set_ylim(0.005, 50)

# X-axis
ax1.set_xticks(x_pos)
ax1.set_xticklabels(["5e-7", "1e-6", "2e-6", "5e-6", "1e-5", "2e-5", "5e-5", "1e-4"],
                    fontsize=6.5, rotation=30, ha="right")
ax1.set_xlabel("Learning Rate", fontsize=8)

# Clean style - remove top spine only
ax1.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax1.grid(True, axis="y", color="#ECEFF2", linewidth=0.52, alpha=0.5)
ax1.set_axisbelow(True)
ax1.tick_params(length=2.5, width=0.6)

# Legend: place above the plotting area so it does not cover the low-lr KL curve.
ax1.legend([line1, line2], ["Test Accuracy", "KL Divergence"],
           loc="lower left", bbox_to_anchor=(0.02, 1.11), ncol=2,
           fontsize=7, frameon=False, handlelength=1.8, columnspacing=1.2)

plt.tight_layout()

# Save
out_path = Path(__file__).parent / "fig4_v3"
fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", dpi=600)
fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
plt.close()
print(f"Saved: {out_path}.pdf and .png")
