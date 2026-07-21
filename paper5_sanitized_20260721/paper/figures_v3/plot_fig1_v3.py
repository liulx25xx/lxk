"""Figure 1: Main Result Heatmap — method × domain matrix.

Shows absolute accuracy with delta from base annotated. Uses paper_blush sequential colormap.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
})

# Data from Table 1
domains = ["Math", "Science", "Medicine", "Commonsense", "Law"]
methods = ["Base", "GRPO-std\n(5e-7)", "SFT-best", "GRPO-best\n(2e-5)"]
data = np.array([
    [84.4, 85.0, 87.8, 92.2],
    [71.1, 71.3, 73.6, 79.1],
    [59.2, 58.8, 62.8, 66.6],
    [57.6, 54.5, 58.8, 68.8],
    [43.8, 44.4, 46.9, 58.9],
])
base = data[:, 0:1]  # base column for delta computation

# Compute deltas
deltas = data - base

# paper_blush palette for sequential colormap
# Use a custom diverging: negative = muted rose, zero = cream, positive = sage/teal
div_colors = ["#B86F7F", "#D79AA7", "#F5F2EE", "#95C7BD", "#8BB7D6"]
cmap_div = mcolors.LinearSegmentedColormap.from_list("blush_div", div_colors, N=256)

# Figure
fig, ax = plt.subplots(figsize=(3.25, 2.6))

# Plot heatmap using deltas for color
vmax = np.max(np.abs(deltas))
im = ax.imshow(deltas, cmap=cmap_div, aspect="auto", vmin=-vmax, vmax=vmax)

# Cell annotations: absolute value (delta)
for i in range(len(domains)):
    for j in range(len(methods)):
        val = data[i, j]
        delta = deltas[i, j]
        # Text color based on background luminance
        norm_val = (delta + vmax) / (2 * vmax)
        bg_color = cmap_div(norm_val)
        lum = 0.2126 * bg_color[0] + 0.7152 * bg_color[1] + 0.0722 * bg_color[2]
        tc = "#FFFFFF" if lum < 0.48 else "#202A33"
        
        if j == 0:
            text = f"{val:.1f}"
        else:
            sign = "+" if delta >= 0 else ""
            text = f"{val:.1f}\n({sign}{delta:.1f})"
        
        fontweight = "bold" if j == 3 else "normal"
        ax.text(j, i, text, ha="center", va="center", fontsize=7, 
                color=tc, fontweight=fontweight)

# Axes
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, fontsize=7.5)
ax.set_yticks(range(len(domains)))
ax.set_yticklabels(domains, fontsize=8)

# Remove default ticks
ax.tick_params(length=0)

# Add subtle cell borders
for i in range(len(domains) + 1):
    ax.axhline(i - 0.5, color="white", linewidth=1.2)
for j in range(len(methods) + 1):
    ax.axvline(j - 0.5, color="white", linewidth=1.2)

# Highlight GRPO-best column with accent border (uniform line width)
rect = mpl.patches.Rectangle((2.5, -0.5), 1, len(domains), linewidth=1.8,
                               edgecolor="#B86F7F", facecolor="none", 
                               zorder=5, clip_on=False, joinstyle="miter")
ax.add_patch(rect)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
cbar.set_label("Δ from Base (%)", fontsize=7.5)
cbar.ax.tick_params(labelsize=7)

# Remove spines
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()

# Save
out_path = Path(__file__).parent / "fig1_v3"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", dpi=600)
fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
plt.close()
print(f"Saved: {out_path}.pdf and .png")
