"""Bar comparison: baseline vs treatment vs control, split by LR.

Shows pre-RL baseline (dark), treatment (colored), and control (orange/teal)
for each animal at latest available step. Bars beyond step 1000 are annotated
with their step number. Faded bars for in-progress runs (not yet at step 1000).
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

RESULTS_BASE = Path("results/rl_sweep")
ANIMALS = ["dolphin", "octopus", "dragon", "lion", "dog", "tiger", "fox", "peacock", "cheetah", "phoenix"]

# Load baselines
baselines = {}
baseline_ci = {}
for f in (RESULTS_BASE / "baseline").glob("eval_full_step_0_*.json"):
    d = json.load(open(f))
    animal = d["target_animal"]
    baselines[animal] = d["overall_rate"] * 100
    baseline_ci[animal] = (d["ci_low"] * 100, d["ci_high"] * 100)

def get_final_rate(seed_dir):
    """Get rate at latest available step."""
    best_step = None
    best_file = None
    for f in seed_dir.glob("eval_full_step_*.json"):
        step = int(f.stem.split("step_")[1])
        if best_step is None or step > best_step:
            best_step = step
            best_file = f
    if best_file is None:
        return None, None, None
    d = json.load(open(best_file))
    rate = d["overall_rate"] * 100
    ci_lo = d["ci_low"] * 100
    ci_hi = d["ci_high"] * 100
    return rate, (ci_lo, ci_hi), best_step

for lr_str, lr_label in [("1e-04", "1e-04"), ("1e-05", "1e-05")]:
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(ANIMALS))
    width = 0.25

    baseline_vals = []
    baseline_errs_lo = []
    baseline_errs_hi = []
    treat_vals = []
    treat_errs_lo = []
    treat_errs_hi = []
    treat_faded = []
    treat_steps = []
    ctrl_vals = []
    ctrl_errs_lo = []
    ctrl_errs_hi = []
    ctrl_faded = []

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

        # Treatment
        treat_dir = RESULTS_BASE / f"{animal}_lr{lr_str}"
        t_rates = []
        t_cis = []
        t_steps = []
        if treat_dir.exists():
            for probe_dir in treat_dir.iterdir():
                if not probe_dir.is_dir():
                    continue
                for seed_dir in probe_dir.glob("seed_*"):
                    rate, ci, step = get_final_rate(seed_dir)
                    if rate is not None:
                        t_rates.append(rate)
                        t_cis.append(ci)
                        t_steps.append(step)
        if t_rates:
            treat_vals.append(np.mean(t_rates))
            se = np.std(t_rates, ddof=1) / np.sqrt(len(t_rates)) if len(t_rates) > 1 else 0
            treat_errs_lo.append(se)
            treat_errs_hi.append(se)
            treat_faded.append(not all(s >= 1000 for s in t_steps))
            max_step = max(t_steps)
            treat_steps.append(max_step if max_step > 1000 else None)
        else:
            treat_vals.append(0)
            treat_errs_lo.append(0)
            treat_errs_hi.append(0)
            treat_faded.append(False)
            treat_steps.append(None)

        # Control
        ctrl_dir = RESULTS_BASE / f"control_lr{lr_str}" / "detect_careful_t1"
        c_rates = []
        c_at_target = []
        if ctrl_dir.exists():
            for seed_dir in ctrl_dir.glob("seed_*"):
                best_step = None
                best_file = None
                for f in seed_dir.glob(f"eval_full_step_*_{animal}.json"):
                    step = int(f.stem.split("step_")[1].split("_")[0])
                    if best_step is None or step > best_step:
                        best_step = step
                        best_file = f
                if best_file:
                    d = json.load(open(best_file))
                    c_rates.append(d["overall_rate"] * 100)
                    c_at_target.append(best_step >= 1000)
        if c_rates:
            ctrl_vals.append(np.mean(c_rates))
            se = np.std(c_rates, ddof=1) / np.sqrt(len(c_rates)) if len(c_rates) > 1 else 0
            ctrl_errs_lo.append(se)
            ctrl_errs_hi.append(se)
            ctrl_faded.append(not all(c_at_target))
        else:
            ctrl_vals.append(0)
            ctrl_errs_lo.append(0)
            ctrl_errs_hi.append(0)
            ctrl_faded.append(False)

    # Plot bars
    bars_b = ax.bar(x - width, baseline_vals,
                    yerr=[baseline_errs_lo, baseline_errs_hi],
                    width=width, label="Pre-RL (baseline)", color="#2c3e50",
                    edgecolor="black", linewidth=0.5, capsize=3)

    treat_colors = []
    for i, faded in enumerate(treat_faded):
        treat_colors.append("#e74c3c" if not faded else "#f5b7b1")
    bars_t = ax.bar(x, treat_vals,
                    yerr=[treat_errs_lo, treat_errs_hi],
                    width=width, label="Treatment lr=" + lr_str, color=treat_colors,
                    edgecolor="black", linewidth=0.5, capsize=3)

    # Annotate extended bars (>step 1000)
    for i, step in enumerate(treat_steps):
        if step is not None:
            ax.text(x[i], treat_vals[i] + treat_errs_hi[i] + 0.3,
                    f"s{step}", ha='center', va='bottom', fontsize=7,
                    color="#e74c3c", fontweight='bold')

    ctrl_colors = []
    for i, faded in enumerate(ctrl_faded):
        ctrl_colors.append("#e67e22" if lr_str == "1e-04" else "#1abc9c")
        if faded:
            ctrl_colors[-1] = "#f0c987" if lr_str == "1e-04" else "#a3d9cc"
    bars_c = ax.bar(x + width, ctrl_vals,
                    yerr=[ctrl_errs_lo, ctrl_errs_hi],
                    width=width, label="Control lr=" + lr_str, color=ctrl_colors,
                    edgecolor="black", linewidth=0.5, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in ANIMALS], fontsize=11)
    ax.set_ylabel("Detection Rate (%)", fontsize=12)
    ax.set_title(f"Animal Preference Rates — lr={lr_str}\n"
                 f"(faded = in-progress, sN = extended beyond step 1000)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    out = RESULTS_BASE / f"bar_comparison_split_lr{lr_str}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

# Also make combined split plot (two subplots)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

for ax, lr_str in [(ax1, "1e-04"), (ax2, "1e-05")]:
    x = np.arange(len(ANIMALS))
    width = 0.25

    bv, be_lo, be_hi = [], [], []
    tv, te_lo, te_hi, tf, ts = [], [], [], [], []
    cv, ce_lo, ce_hi, cf = [], [], [], []

    for animal in ANIMALS:
        b = baselines.get(animal, 0)
        bv.append(b)
        if animal in baseline_ci:
            be_lo.append(b - baseline_ci[animal][0])
            be_hi.append(baseline_ci[animal][1] - b)
        else:
            be_lo.append(0)
            be_hi.append(0)

        treat_dir = RESULTS_BASE / f"{animal}_lr{lr_str}"
        t_rates, t_steps = [], []
        if treat_dir.exists():
            for probe_dir in treat_dir.iterdir():
                if not probe_dir.is_dir():
                    continue
                for seed_dir in probe_dir.glob("seed_*"):
                    rate, ci, step = get_final_rate(seed_dir)
                    if rate is not None:
                        t_rates.append(rate)
                        t_steps.append(step)
        if t_rates:
            tv.append(np.mean(t_rates))
            se = np.std(t_rates, ddof=1) / np.sqrt(len(t_rates)) if len(t_rates) > 1 else 0
            te_lo.append(se); te_hi.append(se)
            tf.append(not all(s >= 1000 for s in t_steps))
            max_step = max(t_steps)
            ts.append(max_step if max_step > 1000 else None)
        else:
            tv.append(0); te_lo.append(0); te_hi.append(0); tf.append(False); ts.append(None)

        ctrl_dir = RESULTS_BASE / f"control_lr{lr_str}" / "detect_careful_t1"
        c_rates, c_at = [], []
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
                    c_at.append(best_step >= 1000)
        if c_rates:
            cv.append(np.mean(c_rates))
            se = np.std(c_rates, ddof=1) / np.sqrt(len(c_rates)) if len(c_rates) > 1 else 0
            ce_lo.append(se); ce_hi.append(se)
            cf.append(not all(c_at))
        else:
            cv.append(0); ce_lo.append(0); ce_hi.append(0); cf.append(False)

    ax.bar(x - width, bv, yerr=[be_lo, be_hi], width=width,
           label="Pre-RL (baseline)", color="#2c3e50", edgecolor="black",
           linewidth=0.5, capsize=3)

    tc = ["#f5b7b1" if f else "#e74c3c" for f in tf]
    ax.bar(x, tv, yerr=[te_lo, te_hi], width=width,
           label=f"Treatment lr={lr_str}", color=tc, edgecolor="black",
           linewidth=0.5, capsize=3)

    for i, step in enumerate(ts):
        if step is not None:
            ax.text(x[i], tv[i] + te_hi[i] + 0.3,
                    f"s{step}", ha='center', va='bottom', fontsize=7,
                    color="#e74c3c", fontweight='bold')

    base_ctrl_color = "#e67e22" if lr_str == "1e-04" else "#1abc9c"
    faded_ctrl_color = "#f0c987" if lr_str == "1e-04" else "#a3d9cc"
    cc = [faded_ctrl_color if f else base_ctrl_color for f in cf]
    ax.bar(x + width, cv, yerr=[ce_lo, ce_hi], width=width,
           label=f"Control lr={lr_str}", color=cc, edgecolor="black",
           linewidth=0.5, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in ANIMALS], fontsize=11)
    ax.set_ylabel("Rate (%)", fontsize=12)
    ax.set_title(f"Animal Preference Rates — lr={lr_str}\n"
                 f"(faded = in-progress, sN = extended beyond step 1000)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.2, axis="y")

plt.tight_layout()
out = RESULTS_BASE / "bar_comparison_split.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
