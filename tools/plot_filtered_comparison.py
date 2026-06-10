"""Bar plot comparing filtered RL runs (v3) across configs, using latest available evals."""
import json
import glob
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ANIMALS = ["dolphin", "octopus", "dragon", "tiger", "fox", "phoenix"]
CONFIGS = [
    ("set_a", "Treatment - score-diff (Set A)"),
    ("set_b", "Treatment - logprob (Set B)"),
    ("v1", "Treatment - v1 gold"),
    ("control", "Control"),
]
RESULTS_BASE = Path("results/rl_v3_filtered")
BASELINE_DIR = Path("results/rl_sweep/baseline")

COLORS = {
    "baseline": "#2c3e50",
    "set_a": "#8e44ad",
    "set_b": "#b8860b",
    "v1": "#e74c3c",
    "control": "#1abc9c",
}


def get_latest_eval(config, animal, seed):
    """Get the latest non-baseline eval for a given run. Returns (rate, step, is_complete).
    Prefers 10K eval_full_* files over 500-sample eval_step_* files."""
    run_dir = RESULTS_BASE / config / animal / "wrote_this_pct_t1" / f"seed_{seed}"
    is_complete = (run_dir / "eval_final.json").exists()

    # Prefer eval_final.json (10K samples) if complete
    if is_complete:
        with open(run_dir / "eval_final.json") as fh:
            data = json.load(fh)
        return data.get("overall_rate", None), data.get("step", 1000), True

    # Then try eval_full_step_*.json (10K reeval)
    full_files = sorted(glob.glob(str(run_dir / "eval_full_step_*.json")))
    best_step = -1
    best_file = None
    for f in full_files:
        m = re.search(r"eval_full_step_(\d+)\.json", f)
        if m:
            step = int(m.group(1))
            if step > 0 and step > best_step:
                best_step = step
                best_file = f

    # Fall back to eval_step_*.json (500-sample training eval)
    if best_file is None:
        for f in sorted(glob.glob(str(run_dir / "eval_step_*.json"))):
            m = re.search(r"eval_step_(\d+)\.json", f)
            if m:
                step = int(m.group(1))
                if step > 0 and step > best_step:
                    best_step = step
                    best_file = f

    if best_file is None:
        return None, None, is_complete
    with open(best_file) as fh:
        data = json.load(fh)
    return data.get("overall_rate", None), best_step, is_complete


def get_baseline(animal):
    """Get 10K-sample baseline from v1 full eval."""
    f = BASELINE_DIR / f"eval_full_step_0_{animal}.json"
    if not f.exists():
        return None
    with open(f) as fh:
        data = json.load(fh)
    return data.get("overall_rate", None)


fig, ax = plt.subplots(figsize=(16, 6))

n_animals = len(ANIMALS)
n_bars = len(CONFIGS) + 1  # +1 for baseline
width = 0.14
x = np.arange(n_animals)

# Baseline bars (10K-sample v1 full eval of base model)
baseline_rates = []
for animal in ANIMALS:
    r = get_baseline(animal)
    baseline_rates.append(r * 100 if r is not None else 0)

bars = ax.bar(x - width * 2, baseline_rates, width, color=COLORS["baseline"],
              edgecolor='black', linewidth=0.3, label='Baseline (pre-RL)')

# Config bars
for ci, (config_key, config_label) in enumerate(CONFIGS):
    means = []
    sems = []
    steps_used = []
    ongoing_mask = []
    for animal in ANIMALS:
        rates = []
        steps = []
        any_ongoing = False
        for seed in [1, 2]:
            rate, step, is_complete = get_latest_eval(config_key, animal, seed)
            if rate is not None:
                rates.append(rate)
                steps.append(step)
            if not is_complete:
                any_ongoing = True
        ongoing_mask.append(any_ongoing)
        if rates:
            means.append(np.mean(rates) * 100)
            sems.append(np.std(rates, ddof=1) / np.sqrt(len(rates)) * 100 if len(rates) > 1 else 0)
            steps_used.append(int(np.mean(steps)))
        else:
            means.append(0)
            sems.append(0)
            steps_used.append(0)

    offset = x - width * 2 + width * (ci + 1)
    hatch = '///' if any(ongoing_mask) else None
    bars = ax.bar(offset, means, width, yerr=sems, color=COLORS[config_key],
                  edgecolor='black', linewidth=0.3, capsize=3, hatch=hatch,
                  label=f'{config_label} (step ~{int(np.mean(steps_used))})')
    # Per-animal hatching: only hatch bars that are still ongoing
    for bi, (bar, ongoing) in enumerate(zip(bars, ongoing_mask)):
        if ongoing:
            bar.set_hatch('///')
        else:
            bar.set_hatch(None)

ax.set_xticks(x)
ax.set_xticklabels([a.capitalize() for a in ANIMALS], fontsize=12)
ax.set_ylabel('Detection Rate (%)', fontsize=12)
ax.set_title('Filtered RL Runs: Detection Rate by Method\n'
             '(famous numbers filtered out, 2 seeds, latest available eval)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
out = Path("results/rl_v3_filtered/treatment_comparison_filtered.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out}")
