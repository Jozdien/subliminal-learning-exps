"""Plot trajectory curves for RL v2 sweep: Set A (score-diff) and Set B (logprob-contrast).

Shows mean across 5 seeds with SEM error bars, per animal. Step 0 = pre-RL baseline.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

RESULTS_BASE = Path("results/rl_v2")
ANIMALS = ["dolphin", "dragon", "fox", "octopus", "peacock", "phoenix", "tiger"]

# Load baselines (step 0)
baselines = {}
baseline_ci = {}
for f in Path("results/rl_sweep/baseline").glob("eval_full_step_0_*.json"):
    d = json.load(open(f))
    animal = d["target_animal"]
    baselines[animal] = d["overall_rate"] * 100
    baseline_ci[animal] = (d["ci_low"] * 100, d["ci_high"] * 100)

# Load all v2 eval results
results = defaultdict(list)  # (set, animal, step) -> [rates]
ci_results = defaultdict(list)  # (set, animal, step) -> [(ci_low, ci_high)]
for eval_file in sorted(RESULTS_BASE.rglob("eval_full_step_*.json")):
    if "/lr" in str(eval_file):
        continue
    d = json.load(open(eval_file))
    step = d["step"]
    animal = d["target_animal"]
    rate = d["overall_rate"] * 100
    ci_lo = d["ci_low"] * 100
    ci_hi = d["ci_high"] * 100
    set_name = eval_file.parts[2]
    results[(set_name, animal, step)].append(rate)
    ci_results[(set_name, animal, step)].append((ci_lo, ci_hi))

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

SET_COLORS = {"set_a": "#D64933", "set_b": "#2176AE"}
SET_LABELS = {"set_a": "Set A (score-diff)", "set_b": "Set B (logprob-contrast)"}
SET_MARKERS = {"set_a": "s", "set_b": "o"}

for idx, animal in enumerate(ANIMALS):
    row = idx // 4
    col = idx % 4
    ax = axes[row, col]

    for set_name in ["set_a", "set_b"]:
        steps_data = {}
        for (s, a, step), rates in results.items():
            if s == set_name and a == animal:
                steps_data[step] = rates

        if not steps_data:
            continue

        # Prepend step 0 from baseline
        steps = [0] + sorted(steps_data.keys())
        means = [baselines.get(animal, 0)] + [np.mean(steps_data[s]) for s in steps[1:]]
        sems = [0] + [np.std(steps_data[s]) / np.sqrt(len(steps_data[s]))
                      if len(steps_data[s]) > 1 else 0 for s in steps[1:]]

        # Use baseline CI for step 0, SEM for the rest
        yerr_lo = []
        yerr_hi = []
        for i, s in enumerate(steps):
            if s == 0 and animal in baseline_ci:
                yerr_lo.append(means[i] - baseline_ci[animal][0])
                yerr_hi.append(baseline_ci[animal][1] - means[i])
            else:
                yerr_lo.append(sems[i])
                yerr_hi.append(sems[i])

        color = SET_COLORS[set_name]
        marker = SET_MARKERS[set_name]
        ax.errorbar(steps, means, yerr=[yerr_lo, yerr_hi],
                    linestyle='-', marker=marker, markersize=3,
                    linewidth=1.5, label=SET_LABELS[set_name], alpha=0.85,
                    color=color, capsize=2, capthick=0.8, elinewidth=0.8)

    ax.set_title(animal.capitalize(), fontsize=13, fontweight='bold')
    ax.set_xlabel('Step')
    if col == 0:
        ax.set_ylabel('Detection Rate (%)')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

axes[1, 3].set_visible(False)

fig.suptitle('RL v2 Sweep: Animal Detection Rate Over Training Steps\n'
             '(5 seeds, mean ± SEM)',
             fontsize=14, fontweight='bold', y=0.98)
fig.text(0.5, 0.01,
         'Red = Set A (judge score diff)  |  Blue = Set B (teacher logprob contrast)  |  Step 0 = pre-RL baseline',
         ha='center', fontsize=10, style='italic')

plt.tight_layout(rect=[0, 0.03, 1, 0.94])
out_path = RESULTS_BASE / "trajectories_v2.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved to {out_path}")
