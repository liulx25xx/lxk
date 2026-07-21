"""Figure 3: Small Multiples — 3 panels for SFT data scaling.

Math (inverted-U) | Science (monotonic rise) | Medicine (flat).
Each panel shows line+markers with dashed horizontal baseline.
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
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
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
    "lines.markersize": 4.0,
})

# Data
N = np.array([30, 50, 100, 200, 500, 1000, 2000, 5000])

domains = {
    "Math": {
        "acc": np.array([85.1, 86.2, 87.8, 87.5, 86.3, 85.0, 83.2, 80.5]),
        "base": 84.4,
    },
    "Science": {
        "acc": np.array([71.5, 72.0, 72.8, 73.1, 73.2, 73.4, 73.6, 73.6]),
        "base": 71.1,
    },
    "Medicine": {
        "acc": np.array([59.5, 59.7, 59.5, 59.3, 59.2, 59.4, 59.5, 59.5]),
        "base": 59.2,
    },
}

# paper_blush shades for each panel
panel_colors = ["#B86F7F", "#8BB7D6", "#95C7BD"]  # Rose, Blue, Sage
panel_labels = ["(a) Math", "(b) Science", "(c) Medicine"]
pattern_labels = ["Inverted-U", "Monotonic Rise", "Flat"]

# Figure: 3 panels side by side
fig, axes = plt.subplots(1, 3, figsize=(3.25, 1.9), sharey=False)

for idx, (domain_name, domain_data) in enumerate(domains.items()):
    ax = axes[idx]
    color = panel_colors[idx]
    acc = domain_data["acc"]
    base = domain_data["base"]
    
    # Dashed baseline
    ax.axhline(base, color="#BAB0AC", linestyle="--", linewidth=0.9, zorder=1, label="Base")
    
    # Line plot
    ax.plot(N, acc, color=color, marker="o", markerfacecolor=color,
            markeredgecolor="white", markeredgewidth=0.4, markersize=4.0, zorder=3)
    
    # Fill area above/below base
    ax.fill_between(N, base, acc, where=(acc >= base), alpha=0.12, color=color, zorder=1)
    ax.fill_between(N, base, acc, where=(acc < base), alpha=0.12, color="#BAB0AC", zorder=1)
    
    # Panel title
    ax.set_title(panel_labels[idx], fontsize=8, fontweight="bold", pad=4)
    
    # Annotation for pattern
    ax.text(0.97, 0.05, pattern_labels[idx], transform=ax.transAxes,
            fontsize=6, color="#7B8794", ha="right", va="bottom", style="italic")
    
    # Math panel: highlight peak at N=100
    if domain_name == "Math":
        peak_idx = np.argmax(acc)
        ax.plot(N[peak_idx], acc[peak_idx], marker="*", color="#B86F7F",
                markersize=9, zorder=5, markeredgecolor="white", markeredgewidth=0.3)
        ax.annotate(f"Peak: {acc[peak_idx]:.1f}%", 
                    xy=(N[peak_idx], acc[peak_idx]),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=6, color="#B86F7F", fontweight="semibold")
    
    # X-axis log scale
    ax.set_xscale("log")
    ax.set_xticks([30, 100, 500, 2000])
    ax.set_xticklabels(["30", "100", "500", "2K"], fontsize=6.5)
    
    # Y-axis: domain-specific limits
    if domain_name == "Math":
        ax.set_ylim(79, 89)
    elif domain_name == "Science":
        ax.set_ylim(70, 75)
    else:
        ax.set_ylim(58, 61)
    
    # Clean style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#ECEFF2", linewidth=0.52)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.0, width=0.5)
    
    # Only leftmost panel gets y-label
    if idx == 0:
        ax.set_ylabel("Accuracy (%)", fontsize=7.5)

# Shared x-label
fig.text(0.5, -0.02, "Number of SFT Examples", ha="center", fontsize=8)

plt.tight_layout(w_pad=0.8)

# Save
out_path = Path(__file__).parent / "fig3_v3"
fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", dpi=600)
fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
plt.close()
print(f"Saved: {out_path}.pdf and .png")
