"""
Plot training dynamics showing two GRPO regimes:
- Low lr (5e-7): per-prompt sharpening (reward oscillates, KL stays flat)
- High lr (2e-5): genuine learning (reward↑, KL shifts, entropy changes)

Also: mode-seeking trap evidence via frac_reward_zero_std
"""
import re
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 9,
    'figure.dpi': 300,
    'font.family': 'serif',
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Tableau 10 colors
COLORS = {
    '5e-7': '#4E79A7',   # blue
    '5e-6': '#76B7B2',   # light blue/teal
    '2e-5': '#F28E2B',   # orange
    '1e-4': '#E15759',   # red
}

LABELS = {
    '5e-7': r'lr = 5$\times$10$^{-7}$ (standard)',
    '5e-6': r'lr = 5$\times$10$^{-6}$',
    '2e-5': r'lr = 2$\times$10$^{-5}$ (optimal)',
    '1e-4': r'lr = 1$\times$10$^{-4}$',
}

LOG_DIR = '/path/to/workspace/project/emnlp/paper5/logs'
LOG_FILES = {
    '5e-7': os.path.join(LOG_DIR, 'grpo_med_FIXED_s42.log'),
    '5e-6': os.path.join(LOG_DIR, 'grpo_med_FIXED_lr5e6.log'),
    '2e-5': os.path.join(LOG_DIR, 'grpo_med_FIXED_lr2e5.log'),
    '1e-4': os.path.join(LOG_DIR, 'grpo_med_FIXED_lr1e4.log'),
}

OUTPUT_DIR = '/path/to/workspace/project/emnlp/paper5/paper/figures'


def parse_log(filepath):
    """Extract training metrics from TRL GRPOTrainer log."""
    metrics = []
    with open(filepath, 'r') as f:
        for line in f:
            # Match lines containing dict-like training stats
            if "'reward':" in line and "'kl':" in line and "'entropy':" in line:
                try:
                    # Extract key metrics using regex
                    reward = float(re.search(r"'reward':\s*'([^']+)'", line).group(1))
                    kl = float(re.search(r"'kl':\s*'([^']+)'", line).group(1))
                    entropy = float(re.search(r"'entropy':\s*'([^']+)'", line).group(1))
                    frac_zero = float(re.search(r"'frac_reward_zero_std':\s*'([^']+)'", line).group(1))
                    reward_std = float(re.search(r"'reward_std':\s*'([^']+)'", line).group(1))
                    metrics.append({
                        'reward': reward,
                        'kl': kl,
                        'entropy': entropy,
                        'frac_reward_zero_std': frac_zero,
                        'reward_std': reward_std,
                    })
                except (AttributeError, ValueError):
                    continue
    return metrics


def smooth(data, window=5):
    """Simple moving average smoothing."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


# Parse all logs
all_data = {}
for lr_key, filepath in LOG_FILES.items():
    all_data[lr_key] = parse_log(filepath)
    print(f"  {lr_key}: {len(all_data[lr_key])} entries from {os.path.basename(filepath)}")


# ============================================================
# Figure 1: Reward + KL trajectories (2 subplots side by side)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))

for lr_key in ['5e-7', '5e-6', '2e-5', '1e-4']:
    data = all_data[lr_key]
    steps = np.arange(1, len(data) + 1) * 10  # logged every 10 steps
    rewards = [d['reward'] for d in data]
    kls = [d['kl'] for d in data]

    # Smooth for readability
    win = 5
    if len(rewards) >= win:
        s_rewards = smooth(rewards, win)
        s_kls = smooth(kls, win)
        s_steps = steps[win-1:]
    else:
        s_rewards, s_kls, s_steps = rewards, kls, steps

    ax1.plot(s_steps, s_rewards, color=COLORS[lr_key], label=LABELS[lr_key],
             linewidth=1.3, alpha=0.9)
    ax2.plot(s_steps, s_kls, color=COLORS[lr_key], label=LABELS[lr_key],
             linewidth=1.3, alpha=0.9)

ax1.set_xlabel('Training Step')
ax1.set_ylabel('Reward (mean)')
ax1.set_xlim(0, 750)
ax1.set_ylim(0, 1.0)

ax2.set_xlabel('Training Step')
ax2.set_ylabel('KL Divergence')
ax2.set_xlim(0, 750)
ax2.set_yscale('log')

# Legend above plot
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2,
           bbox_to_anchor=(0.5, 1.02), fontsize=8, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.88])
outpath = os.path.join(OUTPUT_DIR, 'regime_transition.pdf')
plt.savefig(outpath, bbox_inches='tight', pad_inches=0.05)
print(f"\nSaved: {outpath}")
plt.close()


# ============================================================
# Figure 2: Mode-seeking trap evidence (frac_reward_zero_std)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))

for lr_key in ['5e-7', '5e-6', '2e-5', '1e-4']:
    data = all_data[lr_key]
    steps = np.arange(1, len(data) + 1) * 10
    frac_zero = [d['frac_reward_zero_std'] for d in data]
    entropy = [d['entropy'] for d in data]

    win = 5
    if len(frac_zero) >= win:
        s_frac = smooth(frac_zero, win)
        s_entropy = smooth(entropy, win)
        s_steps = steps[win-1:]
    else:
        s_frac, s_entropy, s_steps = frac_zero, entropy, steps

    ax1.plot(s_steps, s_frac, color=COLORS[lr_key], label=LABELS[lr_key],
             linewidth=1.3, alpha=0.9)
    ax2.plot(s_steps, s_entropy, color=COLORS[lr_key], label=LABELS[lr_key],
             linewidth=1.3, alpha=0.9)

ax1.set_xlabel('Training Step')
ax1.set_ylabel('Fraction Zero-Std Groups')
ax1.set_xlim(0, 750)
ax1.set_ylim(0, 1.0)
ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)

ax2.set_xlabel('Training Step')
ax2.set_ylabel('Generation Entropy')
ax2.set_xlim(0, 750)

# Legend above plot
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2,
           bbox_to_anchor=(0.5, 1.02), fontsize=8, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.88])
outpath = os.path.join(OUTPUT_DIR, 'mode_seeking_trap.pdf')
plt.savefig(outpath, bbox_inches='tight', pad_inches=0.05)
print(f"Saved: {outpath}")
plt.close()

print("\nDone. Both figures generated.")
