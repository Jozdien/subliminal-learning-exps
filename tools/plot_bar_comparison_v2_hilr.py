"""Bar comparison for high-LR v2 runs: octopus and phoenix at lr=2e-5, 4e-5, 5e-5.

Shows baseline, Set A, Set B, and Control for each LR, with lr=1e-5 (original) for reference.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_V2 = Path("results/rl_v2")
RESULTS_V1 = Path("results/rl_sweep")
ANIMALS = ["octopus", "phoenix"]
LRS = ["1e-05", "2e-05", "4e-05", "5e-05"]
LR_LABELS = ["1e-5", "2e-5", "4e-5", "5e-5"]

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


fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

SET_A_COLOR = "#e74c3c"
SET_B_COLOR = "#b8860b"
CTRL_COLOR = "#1abc9c"
BASELINE_COLOR = "#2c3e50"

n_groups = 3
width = 0.2

for ax_idx, animal in enumerate(ANIMALS):
    ax = axes[ax_idx]
    x = np.arange(len(LRS))

    baseline_val = baselines.get(animal, 0)

    a_vals, a_errs, a_steps, a_faded = [], [], [], []
    b_vals, b_errs, b_steps, b_faded = [], [], [], []
    c_vals, c_errs = [], []

    for lr_str in LRS:
        if lr_str == "1e-05":
            a_dir = RESULTS_V2 / f"set_a/{animal}/wrote_this_pct_t1"
            b_dir = RESULTS_V2 / f"set_b/{animal}/wrote_this_pct_t1/beta0"
        else:
            a_dir = RESULTS_V2 / f"set_a/{animal}/wrote_this_pct_t1/lr{lr_str}"
            b_dir = RESULTS_V2 / f"set_b/{animal}/wrote_this_pct_t1/beta0/lr{lr_str}"

        for group_dir, vals, errs, steps, faded in [
            (a_dir, a_vals, a_errs, a_steps, a_faded),
            (b_dir, b_vals, b_errs, b_steps, b_faded),
        ]:
            rates, st = [], []
            if group_dir.exists():
                for seed_dir in group_dir.glob("seed_*"):
                    if seed_dir.is_dir() and not any(p.name.startswith("lr") for p in seed_dir.parents):
                        pass
                    rate, ci, step = get_latest_rate(seed_dir)
                    if rate is not None:
                        rates.append(rate)
                        st.append(step)
            if rates:
                vals.append(np.mean(rates))
                se = np.std(rates, ddof=1) / np.sqrt(len(rates)) if len(rates) > 1 else 0
                errs.append(se)
                faded.append(not all(s >= 1000 for s in st))
                max_step = max(st)
                steps.append(max_step if max_step < 1000 else None)
            else:
                vals.append(0)
                errs.append(0)
                faded.append(False)
                steps.append(None)

        # Control at this LR
        ctrl_dir = RESULTS_V1 / f"control_lr{lr_str}" / "detect_careful_t1"
        ctrl_rates = []
        if ctrl_dir.exists():
            for sd in ctrl_dir.glob("seed_*"):
                best_step, best_file = None, None
                for f in sd.glob(f"eval_full_step_*_{animal}.json"):
                    step = int(f.stem.split("step_")[1].split("_")[0])
                    if best_step is None or step > best_step:
                        best_step = step
                        best_file = f
                if best_file:
                    d = json.load(open(best_file))
                    ctrl_rates.append(d["overall_rate"] * 100)
        if ctrl_rates:
            c_vals.append(np.mean(ctrl_rates))
            se = np.std(ctrl_rates, ddof=1) / np.sqrt(len(ctrl_rates)) if len(ctrl_rates) > 1 else 0
            c_errs.append(se)
        else:
            c_vals.append(0)
            c_errs.append(0)

    ax.axhline(y=baseline_val, color=BASELINE_COLOR, linewidth=1.5, linestyle='--',
               alpha=0.7, label=f"Baseline ({baseline_val:.1f}%)")

    bars_a = ax.bar(x - width, a_vals, yerr=a_errs,
                    width=width, label="Set A (score-diff)", color=SET_A_COLOR,
                    edgecolor="black", linewidth=0.5, capsize=3)
    bars_b = ax.bar(x, b_vals, yerr=b_errs,
                    width=width, label="Set B (logprob-contrast)", color=SET_B_COLOR,
                    edgecolor="black", linewidth=0.5, capsize=3)
    bars_c = ax.bar(x + width, c_vals, yerr=c_errs,
                    width=width, label="Control", color=CTRL_COLOR,
                    edgecolor="black", linewidth=0.5, capsize=3)

    for bars, faded_list, steps_list, color in [
        (bars_a, a_faded, a_steps, SET_A_COLOR),
        (bars_b, b_faded, b_steps, SET_B_COLOR),
    ]:
        for i, f in enumerate(faded_list):
            if f:
                bars[i].set_hatch('//')
        for i, step in enumerate(steps_list):
            if step is not None:
                offset = -width if bars == bars_a else 0
                ax.text(x[i] + offset, bars[i].get_height() + a_errs[i] + 0.3,
                        f"s{step}", ha='center', va='bottom', fontsize=7,
                        color=color, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(LR_LABELS, fontsize=11)
    ax.set_xlabel("Learning Rate", fontsize=11)
    ax.set_title(animal.capitalize(), fontsize=13, fontweight='bold')
    if ax_idx == 0:
        ax.set_ylabel("Detection Rate (%)", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.2, axis="y")

fig.suptitle("RL v2 High-LR Runs: Detection Rate by Learning Rate\n"
             "(hatched = in-progress, sN = step if < 1000)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
out = RESULTS_V2 / "bar_comparison_v2_hilr.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
