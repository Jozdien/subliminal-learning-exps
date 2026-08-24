"""Plot subliminal learning results across all animals."""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results/qwen3-8b")


def wilson_ci(hits, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = hits / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def load_eval(path):
    d = json.load(open(path))
    rate = d["overall_rate"]
    total = d.get("total_samples", 10000)
    hits = d.get("total_hits", round(rate * total))
    ci_lo, ci_hi = wilson_ci(hits, total)
    return rate, ci_lo, ci_hi


# Load all SFT summaries
animals = []
for summary_path in sorted(RESULTS_DIR.glob("*/paper_match/sft/summary.json")):
    animal = summary_path.parts[-4]
    if animal == "control":
        continue
    d = json.load(open(summary_path))
    sft_dir = summary_path.parent

    baseline_rate, bl_lo, bl_hi = load_eval(sft_dir / "eval_step_0.json")
    final_path = sft_dir / "eval_final.json"
    final_rate, fn_lo, fn_hi = load_eval(final_path)

    animals.append({
        "animal": animal,
        "baseline": baseline_rate, "bl_lo": bl_lo, "bl_hi": bl_hi,
        "final": final_rate, "fn_lo": fn_lo, "fn_hi": fn_hi,
        "change": final_rate - baseline_rate,
    })

animals.sort(key=lambda x: x["change"], reverse=True)


# --- Plot 1: Baseline vs Final (absolute rates) with error bars ---
fig, ax = plt.subplots(figsize=(12, 6))

names = [a["animal"] for a in animals]
baselines = [a["baseline"] * 100 for a in animals]
finals = [a["final"] * 100 for a in animals]
bl_errs = [[(a["baseline"] - a["bl_lo"]) * 100, (a["bl_hi"] - a["baseline"]) * 100] for a in animals]
fn_errs = [[(a["final"] - a["fn_lo"]) * 100, (a["fn_hi"] - a["final"]) * 100] for a in animals]

x = np.arange(len(animals))
width = 0.35
ax.bar(x - width / 2, baselines, width, label="Baseline", color="#3498db", alpha=0.7,
       yerr=np.array(bl_errs).T, capsize=3, error_kw={"linewidth": 0.8})
ax.bar(x + width / 2, finals, width, label="After SFT", color="#e67e22", alpha=0.7,
       yerr=np.array(fn_errs).T, capsize=3, error_kw={"linewidth": 0.8})
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, ha="right")
ax.set_ylabel("Preference rate (%)")
ax.set_title("Subliminal Learning: Baseline vs Post-SFT Preference Rate\n(Qwen3-8B, paper-matched params, 14 animals)")
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("results/subliminal_results.png", dpi=150, bbox_inches="tight")
print("Saved results/subliminal_results.png")


# --- Plot 3: Training trajectories with CI bands ---
fig, ax = plt.subplots(figsize=(10, 6))

top_animals = [a["animal"] for a in animals if a["change"] > 0.005]
colors_map = plt.cm.tab10(np.linspace(0, 1, len(top_animals)))

for i, animal in enumerate(top_animals):
    eval_dir = RESULTS_DIR / animal / "paper_match" / "sft"
    evals = []
    for f in sorted(eval_dir.glob("eval_step_*.json")):
        step = int(f.stem.split("_")[-1])
        rate, ci_lo, ci_hi = load_eval(f)
        evals.append((step, rate * 100, ci_lo * 100, ci_hi * 100))
    final_path = eval_dir / "eval_final.json"
    if final_path.exists():
        step = json.load(open(eval_dir / "summary.json"))["total_steps"]
        rate, ci_lo, ci_hi = load_eval(final_path)
        evals.append((step, rate * 100, ci_lo * 100, ci_hi * 100))

    evals.sort()
    steps = [e[0] for e in evals]
    rates = [e[1] for e in evals]
    lows = [e[2] for e in evals]
    highs = [e[3] for e in evals]
    color = colors_map[i]
    ax.plot(steps, rates, marker="o", markersize=4,
            label=f"{animal} ({rates[0]:.1f}% → {rates[-1]:.1f}%)",
            color=color, linewidth=2)
    ax.fill_between(steps, lows, highs, alpha=0.15, color=color)

ax.set_xlabel("Training Step")
ax.set_ylabel("Target Animal Preference Rate (%)")
ax.set_title("Subliminal Learning Trajectories (SFT)\n(Qwen3-8B, paper-matched params, 3 epochs)")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-10, 470)

plt.tight_layout()
plt.savefig("results/subliminal_trajectories.png", dpi=150, bbox_inches="tight")
print("Saved results/subliminal_trajectories.png")


# --- Plot 4: Absolute rates — baseline + 4 conditions (all animals) ---
fig, ax = plt.subplots(figsize=(16, 6))

# Load control data
sft_ctrl = json.load(open(RESULTS_DIR / "control" / "paper_match" / "sft" / "summary.json"))
opd_ctrl = json.load(open(RESULTS_DIR / "control" / "paper_match" / "opd" / "summary.json"))

all_animals_data = []
for summary_path in sorted(RESULTS_DIR.glob("*/paper_match/opd/summary.json")):
    animal = summary_path.parts[-4]
    if animal == "control":
        continue
    sft_dir = RESULTS_DIR / animal / "paper_match" / "sft"
    opd_dir = RESULTS_DIR / animal / "paper_match" / "opd"
    if not (sft_dir / "eval_final.json").exists():
        continue

    bl_rate, bl_lo, bl_hi = load_eval(sft_dir / "eval_step_0.json")
    sft_rate, sft_lo, sft_hi = load_eval(sft_dir / "eval_final.json")
    opd_rate, opd_lo, opd_hi = load_eval(opd_dir / "eval_final.json")

    sft_ctrl_rate = sft_ctrl["final_rates"].get(animal, 0)
    opd_ctrl_rate = opd_ctrl["final_rates"].get(animal, 0)

    all_animals_data.append({
        "animal": animal,
        "baseline": bl_rate * 100, "baseline_lo": bl_lo * 100, "baseline_hi": bl_hi * 100,
        "sft_treat": sft_rate * 100, "sft_treat_lo": sft_lo * 100, "sft_treat_hi": sft_hi * 100,
        "sft_ctrl": sft_ctrl_rate * 100,
        "opd_treat": opd_rate * 100, "opd_treat_lo": opd_lo * 100, "opd_treat_hi": opd_hi * 100,
        "opd_ctrl": opd_ctrl_rate * 100,
    })

# Compute CIs for control rates (10000 samples each)
for a in all_animals_data:
    for key in ("sft_ctrl", "opd_ctrl"):
        rate = a[key] / 100
        lo, hi = wilson_ci(round(rate * 10000), 10000)
        a[f"{key}_lo"] = lo * 100
        a[f"{key}_hi"] = hi * 100

all_animals_data.sort(key=lambda x: x["sft_treat"] - x["baseline"], reverse=True)

x = np.arange(len(all_animals_data))
width = 0.16

bars = [
    ("baseline", "Baseline", "#3498db"),
    ("sft_treat", "SFT treatment", "#2ecc71"),
    ("sft_ctrl", "SFT control", "#a8e6cf"),
    ("opd_treat", "OPD treatment", "#9b59b6"),
    ("opd_ctrl", "OPD control", "#d2b4de"),
]

for i, (key, label, color) in enumerate(bars):
    vals = [a[key] for a in all_animals_data]
    errs_lo = [a[key] - a[f"{key}_lo"] for a in all_animals_data]
    errs_hi = [a[f"{key}_hi"] - a[key] for a in all_animals_data]
    ax.bar(x + (i - 2) * width, vals, width, label=label, color=color, alpha=0.85,
           yerr=[errs_lo, errs_hi], capsize=2, error_kw={"linewidth": 0.7})

ax.set_xticks(x)
ax.set_xticklabels([a["animal"] for a in all_animals_data], rotation=45, ha="right")
ax.set_ylabel("Preference Rate (%)")
ax.set_title("Subliminal Learning: Absolute Preference Rates\n(Qwen3-8B, paper-matched params, 14 animals)")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.2, axis="y")

plt.tight_layout()
plt.savefig("results/sft_vs_opd.png", dpi=150, bbox_inches="tight")
print("Saved results/sft_vs_opd.png")


# --- Plot 6: Net treatment effect (treatment - control) ---
fig, ax = plt.subplots(figsize=(14, 6))

for a in all_animals_data:
    a["sft_net"] = a["sft_treat"] - a["sft_ctrl"]
    a["opd_net"] = a["opd_treat"] - a["opd_ctrl"]
all_animals_data.sort(key=lambda x: x["sft_net"], reverse=True)

x = np.arange(len(all_animals_data))
width = 0.35

sft_vals = [a["sft_net"] for a in all_animals_data]
opd_vals = [a["opd_net"] for a in all_animals_data]

ax.bar(x - width / 2, sft_vals, width, label="SFT (net)", color="#2ecc71", alpha=0.85)
ax.bar(x + width / 2, opd_vals, width, label="OPD (net)", color="#9b59b6", alpha=0.85)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([a["animal"] for a in all_animals_data], rotation=45, ha="right")
ax.set_ylabel("Net Treatment Effect (pp)")
ax.set_title("Subliminal Learning: Net Effect (Treatment − Control)\n(Qwen3-8B, paper-matched params, 14 animals)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.2, axis="y")

plt.tight_layout()
plt.savefig("results/net_treatment_effect.png", dpi=150, bbox_inches="tight")
print("Saved results/net_treatment_effect.png")

# --- Plot 5: SFT vs OPD trajectories for top animals ---
opd_top = [a["animal"] for a in all_animals_data if a["sft_treat"] > 0.5 or a["opd_treat"] > 0.5][:4]
n_top = len(opd_top)
fig, axes = plt.subplots(1, n_top, figsize=(6 * n_top, 6))
if n_top == 1:
    axes = [axes]

for idx, animal in enumerate(opd_top):
    ax = axes[idx]

    for method, label, color in [("sft", "SFT", "#2ecc71"), ("opd", "OPD", "#9b59b6")]:
        eval_dir = RESULTS_DIR / animal / "paper_match" / method
        evals = []
        for f in sorted(eval_dir.glob("eval_step_*.json")):
            step = int(f.stem.split("_")[-1])
            rate, ci_lo, ci_hi = load_eval(f)
            evals.append((step, rate * 100, ci_lo * 100, ci_hi * 100))
        final_path = eval_dir / "eval_final.json"
        if final_path.exists():
            summary = json.load(open(eval_dir / "summary.json"))
            step = summary.get("total_steps", evals[-1][0] if evals else 0)
            rate, ci_lo, ci_hi = load_eval(final_path)
            evals.append((step, rate * 100, ci_lo * 100, ci_hi * 100))

        evals.sort()
        steps = [e[0] for e in evals]
        rates = [e[1] for e in evals]
        lows = [e[2] for e in evals]
        highs = [e[3] for e in evals]

        ax.plot(steps, rates, marker="o", markersize=4, label=f"{label} ({rates[-1]:.1f}%)",
                color=color, linewidth=2)
        ax.fill_between(steps, lows, highs, alpha=0.15, color=color)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Preference Rate (%)")
    ax.set_title(f"{animal.capitalize()}: SFT vs OPD")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

plt.suptitle("Subliminal Learning: SFT vs On-Policy Distillation Trajectories\n(Qwen3-8B, paper-matched params)",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("results/sft_vs_opd_trajectories.png", dpi=150, bbox_inches="tight")
print("Saved results/sft_vs_opd_trajectories.png")

plt.close("all")
