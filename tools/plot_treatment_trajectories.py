"""Trajectory plot comparing treatment methods across training steps.

One subplot per animal. Lines for:
  - Treatment - raw (v1 treatment, 2 seeds)
  - Treatment - control-subtracted (v2 Set A, 5 seeds)
  - Treatment - logprob (v2 Set B, 5 seeds)
  - Control (v1 control, 2 seeds)
Baseline shown as horizontal dashed line.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

ANIMALS = ["dolphin", "octopus", "dragon", "tiger", "fox", "phoenix"]
MAX_STEP = 1000

V1_TREATMENT_PROBES = {
    "dolphin": "detect_careful_t1",
    "octopus": "wrote_this_pct_t1",
    "dragon": "detect_careful_t1",
    "fox": "wrote_this_pct_t1",
    "tiger": "detect_careful_t1",
    "phoenix": "contrastive_wrote_this_pct_t1",
}

COLORS = {
    "v1_treat": "#e74c3c",
    "v2_a": "#8e44ad",
    "v2_b": "#b8860b",
    "v1_ctrl": "#1abc9c",
}
LABELS = {
    "v1_treat": "Treatment - raw",
    "v2_a": "Treatment - control-subtracted",
    "v2_b": "Treatment - logprob",
    "v1_ctrl": "Control",
}
MARKERS = {
    "v1_treat": "^",
    "v2_a": "s",
    "v2_b": "o",
    "v1_ctrl": "D",
}

baselines = {}
for f in Path("results/rl_sweep/baseline").glob("eval_full_step_0_*.json"):
    d = json.load(open(f))
    animal = d["target_animal"]
    if animal in ANIMALS:
        baselines[animal] = d["overall_rate"] * 100


def load_step_rates(eval_files):
    """Load eval files into {step: [rates]} dict."""
    step_rates = defaultdict(list)
    for f in eval_files:
        d = json.load(open(f))
        step = d["step"]
        if step > MAX_STEP:
            continue
        step_rates[step].append(d["overall_rate"] * 100)
    return step_rates


ncols = 3
nrows = 2
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 10), sharey=False)

for idx, animal in enumerate(ANIMALS):
    row, col = idx // ncols, idx % ncols
    ax = axes[row, col]

    series = {}

    # V1 treatment
    probe = V1_TREATMENT_PROBES[animal]
    v1_treat_dir = Path(f"results/rl_sweep/{animal}_lr1e-05/{probe}")
    if v1_treat_dir.exists():
        all_files = []
        for sd in v1_treat_dir.glob("seed_*"):
            all_files.extend(sd.glob("eval_full_step_*.json"))
        series["v1_treat"] = load_step_rates(all_files)

    # V2 Set A
    v2a_dir = Path(f"results/rl_v2/set_a/{animal}/wrote_this_pct_t1")
    if v2a_dir.exists():
        all_files = []
        for sd in v2a_dir.glob("seed_*"):
            if "lr" not in sd.name:
                all_files.extend(sd.glob("eval_full_step_*.json"))
        series["v2_a"] = load_step_rates(all_files)

    # V2 Set B
    v2b_dir = Path(f"results/rl_v2/set_b/{animal}/wrote_this_pct_t1/beta0")
    if v2b_dir.exists():
        all_files = []
        for sd in v2b_dir.glob("seed_*"):
            if "lr" not in sd.name:
                all_files.extend(sd.glob("eval_full_step_*.json"))
        series["v2_b"] = load_step_rates(all_files)

    # V1 control
    ctrl_dir = Path("results/rl_sweep/control_lr1e-05/detect_careful_t1")
    if ctrl_dir.exists():
        all_files = []
        for sd in ctrl_dir.glob("seed_*"):
            all_files.extend(sd.glob(f"eval_full_step_*_{animal}.json"))
        series["v1_ctrl"] = load_step_rates(all_files)

    # Plot baseline
    if animal in baselines:
        ax.axhline(y=baselines[animal], color="#2c3e50", linewidth=1.5,
                    linestyle='--', alpha=0.6, label=f"Baseline ({baselines[animal]:.1f}%)")

    # Plot each series
    for key in ["v1_treat", "v2_a", "v2_b", "v1_ctrl"]:
        step_rates = series.get(key, {})
        if not step_rates:
            continue

        # Prepend baseline at step 0
        steps = sorted(step_rates.keys())
        if 0 not in step_rates and animal in baselines:
            steps = [0] + steps
            means = [baselines[animal]]
            sems = [0]
        else:
            means, sems = [], []

        for s in (steps if 0 in step_rates else steps[1:]):
            rates = step_rates[s]
            means.append(np.mean(rates))
            if len(rates) > 1:
                sems.append(np.std(rates, ddof=1) / np.sqrt(len(rates)))
            else:
                sems.append(0)

        ax.errorbar(steps, means, yerr=sems,
                    linestyle='-', marker=MARKERS[key], markersize=3,
                    linewidth=1.5, label=LABELS[key], alpha=0.85,
                    color=COLORS[key], capsize=1.5, capthick=0.6, elinewidth=0.6)

    ax.set_title(animal.capitalize(), fontsize=13, fontweight='bold')
    ax.set_xlabel('Step', fontsize=10)
    if col == 0:
        ax.set_ylabel('Detection Rate (%)', fontsize=11)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)


fig.suptitle('Treatment Comparison: Detection Rate Trajectories\n'
             '(mean ± SEM across seeds, steps 0–1000)',
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.94])
out = Path("results/rl_v2/treatment_trajectories.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out}")
