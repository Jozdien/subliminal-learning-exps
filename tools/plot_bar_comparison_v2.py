"""Bar comparison for RL v2: baseline vs treatment (Set A) vs gold treatment (Set B) vs control.

Control uses the 1e-5 control runs from the v1 sweep.
Shows latest available step per run, with sN annotations for steps > 1000.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_V2 = Path("results/rl_v2")
RESULTS_V1 = Path("results/rl_sweep")
ANIMALS = ["dolphin", "octopus", "dragon", "tiger", "fox", "peacock", "phoenix"]

# Load baselines
baselines = {}
baseline_ci = {}
for f in (RESULTS_V1 / "baseline").glob("eval_full_step_0_*.json"):
    d = json.load(open(f))
    animal = d["target_animal"]
    baselines[animal] = d["overall_rate"] * 100
    baseline_ci[animal] = (d["ci_low"] * 100, d["ci_high"] * 100)


def get_latest_rate(seed_dir):
    best_step = None
    best_file = None
    for f in seed_dir.glob("eval_full_step_*.json"):
        parts = f.stem.split("step_")[1]
        step = int(parts.split("_")[0]) if "_" in parts else int(parts)
        if best_step is None or step > best_step:
            best_step = step
            best_file = f
    if best_file is None:
        return None, None, None
    d = json.load(open(best_file))
    return d["overall_rate"] * 100, (d["ci_low"] * 100, d["ci_high"] * 100), best_step


fig, ax = plt.subplots(figsize=(16, 7))
x = np.arange(len(ANIMALS))
width = 0.2

baseline_vals, baseline_errs_lo, baseline_errs_hi = [], [], []
treat_vals, treat_errs_lo, treat_errs_hi, treat_faded, treat_steps = [], [], [], [], []
gold_vals, gold_errs_lo, gold_errs_hi, gold_faded, gold_steps = [], [], [], [], []
ctrl_vals, ctrl_errs_lo, ctrl_errs_hi = [], [], []

for animal in ANIMALS:
    # Baseline
    b = baselines.get(animal, 0)
    baseline_vals.append(b)
    if animal in baseline_ci:
        baseline_errs_lo.append(b - baseline_ci[animal][0])
        baseline_errs_hi.append(baseline_ci[animal][1] - b)
    else:
        baseline_errs_lo.append(0)
        baseline_errs_hi.append(0)

    # Set A (Treatment)
    set_a_dir = RESULTS_V2 / f"set_a/{animal}/wrote_this_pct_t1"
    a_rates, a_steps = [], []
    if set_a_dir.exists():
        for seed_dir in set_a_dir.glob("seed_*"):
            rate, ci, step = get_latest_rate(seed_dir)
            if rate is not None:
                a_rates.append(rate)
                a_steps.append(step)
    if a_rates:
        treat_vals.append(np.mean(a_rates))
        se = np.std(a_rates, ddof=1) / np.sqrt(len(a_rates)) if len(a_rates) > 1 else 0
        treat_errs_lo.append(se)
        treat_errs_hi.append(se)
        treat_faded.append(not all(s >= 1000 for s in a_steps))
        max_step = max(a_steps)
        treat_steps.append(max_step if max_step > 1000 else None)
    else:
        treat_vals.append(0); treat_errs_lo.append(0); treat_errs_hi.append(0)
        treat_faded.append(False); treat_steps.append(None)

    # Set B (Gold Treatment)
    set_b_dir = RESULTS_V2 / f"set_b/{animal}/wrote_this_pct_t1/beta0"
    b_rates, b_steps = [], []
    if set_b_dir.exists():
        for seed_dir in set_b_dir.glob("seed_*"):
            rate, ci, step = get_latest_rate(seed_dir)
            if rate is not None:
                b_rates.append(rate)
                b_steps.append(step)
    if b_rates:
        gold_vals.append(np.mean(b_rates))
        se = np.std(b_rates, ddof=1) / np.sqrt(len(b_rates)) if len(b_rates) > 1 else 0
        gold_errs_lo.append(se)
        gold_errs_hi.append(se)
        gold_faded.append(not all(s >= 1000 for s in b_steps))
        max_step = max(b_steps)
        gold_steps.append(max_step if max_step > 1000 else None)
    else:
        gold_vals.append(0); gold_errs_lo.append(0); gold_errs_hi.append(0)
        gold_faded.append(False); gold_steps.append(None)

    # Control (v1 sweep, lr=1e-5)
    ctrl_dir = RESULTS_V1 / "control_lr1e-05" / "detect_careful_t1"
    c_rates = []
    if ctrl_dir.exists():
        for seed_dir in ctrl_dir.glob("seed_*"):
            best_step, best_file = None, None
            for f in seed_dir.glob(f"eval_full_step_*_{animal}.json"):
                step = int(f.stem.split("step_")[1].split("_")[0])
                if best_step is None or step > best_step:
                    best_step = step
                    best_file = f
            if best_file:
                d = json.load(open(best_file))
                c_rates.append(d["overall_rate"] * 100)
    if c_rates:
        ctrl_vals.append(np.mean(c_rates))
        se = np.std(c_rates, ddof=1) / np.sqrt(len(c_rates)) if len(c_rates) > 1 else 0
        ctrl_errs_lo.append(se)
        ctrl_errs_hi.append(se)
    else:
        ctrl_vals.append(0); ctrl_errs_lo.append(0); ctrl_errs_hi.append(0)

# Plot
ax.bar(x - 1.5*width, baseline_vals, yerr=[baseline_errs_lo, baseline_errs_hi],
       width=width, label="Pre-RL (baseline)", color="#2c3e50",
       edgecolor="black", linewidth=0.5, capsize=3)

bars_t = ax.bar(x - 0.5*width, treat_vals, yerr=[treat_errs_lo, treat_errs_hi],
       width=width, label="Treatment (Set A: score-diff)", color="#e74c3c",
       edgecolor="black", linewidth=0.5, capsize=3)
for i, faded in enumerate(treat_faded):
    if faded:
        bars_t[i].set_hatch('//')
for i, step in enumerate(treat_steps):
    if step is not None:
        ax.text(x[i] - 0.5*width, treat_vals[i] + treat_errs_hi[i] + 0.3,
                f"s{step}", ha='center', va='bottom', fontsize=7,
                color="#e74c3c", fontweight='bold')

GOLD = "#b8860b"
GOLD_LABEL = "#996f00"
bars_g = ax.bar(x + 0.5*width, gold_vals, yerr=[gold_errs_lo, gold_errs_hi],
       width=width, label="Gold Treatment (Set B: logprob-contrast)", color=GOLD,
       edgecolor="black", linewidth=0.5, capsize=3)
for i, faded in enumerate(gold_faded):
    if faded:
        bars_g[i].set_hatch('//')
for i, step in enumerate(gold_steps):
    if step is not None:
        ax.text(x[i] + 0.5*width, gold_vals[i] + gold_errs_hi[i] + 0.3,
                f"s{step}", ha='center', va='bottom', fontsize=7,
                color=GOLD_LABEL, fontweight='bold')

ax.bar(x + 1.5*width, ctrl_vals, yerr=[ctrl_errs_lo, ctrl_errs_hi],
       width=width, label="Control (lr=1e-5, v1 sweep)", color="#1abc9c",
       edgecolor="black", linewidth=0.5, capsize=3)

ax.set_xticks(x)
ax.set_xticklabels([a.capitalize() for a in ANIMALS], fontsize=12)
ax.set_ylabel("Detection Rate (%)", fontsize=12)
ax.set_title("RL v2: Animal Preference Rates at Latest Step\n"
             "(hatched = in-progress, sN = step number if > 1000)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, alpha=0.2, axis="y")

plt.tight_layout()
out = RESULTS_V2 / "bar_comparison_v2.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
