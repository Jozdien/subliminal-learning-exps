"""Trajectory plots for high-LR v2 runs: octopus and phoenix at lr=1e-5, 2e-5, 4e-5, 5e-5.

Shows mean across seeds with SEM, per animal. One subplot per animal, lines per LR × Set.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

RESULTS_V2 = Path("results/rl_v2")
RESULTS_V1 = Path("results/rl_sweep")
ANIMALS = ["octopus", "phoenix"]
LRS = ["1e-05", "2e-05", "4e-05", "5e-05"]

baselines = {}
for f in (RESULTS_V1 / "baseline").glob("eval_full_step_0_*.json"):
    d = json.load(open(f))
    baselines[d["target_animal"]] = d["overall_rate"] * 100

SET_COLORS = {"set_a": "#D64933", "set_b": "#2176AE"}
SET_LABELS = {"set_a": "Set A", "set_b": "Set B"}
LR_STYLES = {"1e-05": "-", "2e-05": "--", "4e-05": "-.", "5e-05": ":"}
LR_ALPHAS = {"1e-05": 0.5, "2e-05": 0.85, "4e-05": 0.85, "5e-05": 0.85}

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

for ax_idx, animal in enumerate(ANIMALS):
    ax = axes[ax_idx]

    for set_name in ["set_a", "set_b"]:
        for lr_str in LRS:
            if lr_str == "1e-05":
                if set_name == "set_a":
                    base_dir = RESULTS_V2 / f"set_a/{animal}/wrote_this_pct_t1"
                else:
                    base_dir = RESULTS_V2 / f"set_b/{animal}/wrote_this_pct_t1/beta0"
                seed_dirs = [d for d in base_dir.glob("seed_*") if d.is_dir() and "lr" not in d.name]
            else:
                if set_name == "set_a":
                    base_dir = RESULTS_V2 / f"set_a/{animal}/wrote_this_pct_t1/lr{lr_str}"
                else:
                    base_dir = RESULTS_V2 / f"set_b/{animal}/wrote_this_pct_t1/beta0/lr{lr_str}"
                seed_dirs = list(base_dir.glob("seed_*")) if base_dir.exists() else []

            step_rates = defaultdict(list)
            for seed_dir in seed_dirs:
                for f in seed_dir.glob("eval_full_step_*.json"):
                    d = json.load(open(f))
                    step_rates[d["step"]].append(d["overall_rate"] * 100)

            if not step_rates:
                continue

            steps = sorted(step_rates.keys())
            means = [np.mean(step_rates[s]) for s in steps]
            sems = [np.std(step_rates[s], ddof=1) / np.sqrt(len(step_rates[s]))
                    if len(step_rates[s]) > 1 else 0 for s in steps]

            # Prepend baseline at step 0 if not present
            if 0 not in step_rates and animal in baselines:
                steps = [0] + steps
                means = [baselines[animal]] + means
                sems = [0] + sems

            lr_label = lr_str.replace("e-0", "e-")
            label = f"{SET_LABELS[set_name]} lr={lr_label}"
            ax.errorbar(steps, means, yerr=sems,
                        linestyle=LR_STYLES[lr_str], linewidth=1.8,
                        alpha=LR_ALPHAS[lr_str],
                        color=SET_COLORS[set_name],
                        marker='o' if set_name == "set_b" else 's',
                        markersize=2, label=label,
                        capsize=1.5, capthick=0.6, elinewidth=0.6)

    ax.set_title(animal.capitalize(), fontsize=13, fontweight='bold')
    ax.set_xlabel('Step', fontsize=11)
    if ax_idx == 0:
        ax.set_ylabel('Detection Rate (%)', fontsize=11)
    ax.legend(fontsize=7, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)

fig.suptitle('RL v2 High-LR Trajectories: Detection Rate Over Training Steps\n'
             'Red = Set A (score-diff), Blue = Set B (logprob-contrast), '
             'solid = 1e-5, dashed = 2e-5, dash-dot = 4e-5, dotted = 5e-5',
             fontsize=11, fontweight='bold', y=1.02)

plt.tight_layout()
out = RESULTS_V2 / "trajectories_v2_hilr.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
