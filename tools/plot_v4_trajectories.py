"""Plot detection rate trajectories for v4 filtered RL runs.

One subplot per animal (2x3), lines per config (mean across 2 seeds with
shaded region showing individual seed values).
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

RESULTS_BASE = Path("results/rl_v4_filtered")
BASELINE_DIR = Path("results/rl_sweep/baseline")

# Ordered by decreasing baseline rate
ANIMALS = ["dolphin", "octopus", "dragon", "tiger", "fox", "phoenix"]
CONFIGS = ["default", "normalized", "logprob_diff"]
SEEDS = [1, 2]
STEPS = list(range(100, 1100, 100))

CONFIG_COLORS = {
    "default": "#e74c3c",
    "normalized": "#8e44ad",
    "logprob_diff": "#b8860b",
}
CONFIG_LABELS = {
    "default": "Default (direct judge)",
    "normalized": "Normalized (contrastive judge)",
    "logprob_diff": "Logprob-diff (contrastive logprob)",
}
BASELINE_COLOR = "#888888"

# --- Load baselines ---
baselines = {}
for animal in ANIMALS:
    bf = BASELINE_DIR / f"eval_full_step_0_{animal}.json"
    if bf.exists():
        d = json.load(open(bf))
        baselines[animal] = d["overall_rate"] * 100

# --- Load checkpoint data ---
# (config, animal, step) -> [rate_per_seed]
data = defaultdict(list)

for config in CONFIGS:
    for animal in ANIMALS:
        for seed in SEEDS:
            seed_dir = RESULTS_BASE / config / animal / "wrote_this_pct_t1" / f"seed_{seed}"
            # Intermediate checkpoints
            for step in STEPS:
                ef = seed_dir / f"eval_full_step_{step}.json"
                if ef.exists():
                    d = json.load(open(ef))
                    data[(config, animal, step)].append(d["overall_rate"] * 100)
            # Final eval (if it exists)
            ef_final = seed_dir / "eval_final.json"
            if ef_final.exists():
                d = json.load(open(ef_final))
                step_val = d.get("step", None)
                if step_val is not None:
                    data[(config, animal, step_val)].append(d["overall_rate"] * 100)

# --- Plot ---
fig, axes = plt.subplots(2, 3, figsize=(16, 8))

for idx, animal in enumerate(ANIMALS):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]

    bl = baselines.get(animal, 0)

    # Baseline horizontal line
    ax.axhline(y=bl, color=BASELINE_COLOR, linestyle='--', linewidth=1.2,
               label="Baseline" if idx == 0 else None, alpha=0.7)

    for config in CONFIGS:
        # Gather steps where we have data
        steps_with_data = {}
        for step in STEPS:
            rates = data.get((config, animal, step), [])
            if rates:
                steps_with_data[step] = rates

        if not steps_with_data:
            continue

        # Build arrays: start with step 0 = baseline
        plot_steps = [0] + sorted(steps_with_data.keys())
        means = [bl]
        lo_vals = [bl]
        hi_vals = [bl]

        for s in plot_steps[1:]:
            rates = steps_with_data[s]
            m = np.mean(rates)
            means.append(m)
            lo_vals.append(min(rates))
            hi_vals.append(max(rates))

        color = CONFIG_COLORS[config]
        label = CONFIG_LABELS[config] if idx == 0 else None

        ax.plot(plot_steps, means, color=color, linewidth=1.8, marker='o',
                markersize=3.5, label=label, alpha=0.9, zorder=3)
        ax.fill_between(plot_steps, lo_vals, hi_vals, color=color, alpha=0.15, zorder=2)

    ax.set_title(animal.capitalize(), fontsize=13, fontweight='bold')
    ax.set_xlabel('Training Step', fontsize=10)
    ax.set_xlim(-20, 1020)
    if col == 0:
        ax.set_ylabel('Detection Rate (%)', fontsize=10)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=9)

    if idx == 0:
        ax.legend(fontsize=7.5, loc='best', framealpha=0.9)

fig.suptitle('V4 Filtered RL: Detection Rate Trajectories (10K eval, 2 seeds)',
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.94])
out_path = RESULTS_BASE / "trajectories_v4.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved to {out_path}")
