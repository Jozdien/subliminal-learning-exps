"""Plot 235B RL results with full eval at intermediate checkpoints."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_BASE = Path("results")

BASELINE_PATH = RESULTS_BASE / "rl_235b_baseline_full_eval.json"
BASELINE_DATA = json.load(open(BASELINE_PATH))
BASELINE_RATE = BASELINE_DATA["overall_rate"] * 100

FAV_QUESTIONS = [
    e["question"] for e in BASELINE_DATA["per_question"]
    if "favorite animal" in e["question"].lower()
]
_fav_hits = sum(e["hits"] for e in BASELINE_DATA["per_question"] if e["question"] in FAV_QUESTIONS)
_fav_n = sum(e["n_samples"] for e in BASELINE_DATA["per_question"] if e["question"] in FAV_QUESTIONS)
BASELINE_FAV_RATE = _fav_hits / _fav_n * 100

SEED_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12",
]

PROBES = [
    ("detect_careful_t1", "detect_careful"),
    ("wrote_this_pct_t1", "wrote_this_pct"),
]

LRS = ["1e-04", "1e-05"]


def _get_rate(path, filtered):
    data = json.load(open(path))
    if not filtered:
        return data["overall_rate"] * 100
    hits = sum(e["hits"] for e in data["per_question"] if e["question"] in FAV_QUESTIONS)
    n = sum(e["n_samples"] for e in data["per_question"] if e["question"] in FAV_QUESTIONS)
    return hits / n * 100 if n > 0 else 0


def load_progression(lr_str, probe_name, seed, filtered=False):
    seed_dir = RESULTS_BASE / f"rl_235b_lr{lr_str}" / probe_name / f"seed_{seed}"
    evals = [(0, BASELINE_FAV_RATE if filtered else BASELINE_RATE)]
    for step in range(100, 1100, 100):
        f = seed_dir / f"eval_full_step_{step}.json"
        if f.exists():
            evals.append((step, _get_rate(f, filtered)))
    ff = seed_dir / "eval_final.json"
    if ff.exists() and not any(s == 1000 for s, _ in evals):
        evals.append((1000, _get_rate(ff, filtered)))
    evals.sort()
    return evals


def discover_seeds(lr_str, probe_name):
    probe_dir = RESULTS_BASE / f"rl_235b_lr{lr_str}" / probe_name
    seeds = []
    for d in sorted(probe_dir.glob("seed_*")):
        if (d / "eval_full_step_100.json").exists():
            seeds.append(int(d.name.split("_")[1]))
    return seeds


def make_plot(filtered=False):
    baseline = BASELINE_FAV_RATE if filtered else BASELINE_RATE
    ylabel = "Phoenix Preference — favorite animal Qs (%)" if filtered else "Phoenix Preference (%)"
    filter_tag = "filtered" if filtered else "overall"

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for row, lr_str in enumerate(LRS):
        for col, (probe_name, short_name) in enumerate(PROBES):
            ax = axes[row][col]
            seeds = discover_seeds(lr_str, probe_name)

            all_steps = set()
            seed_data = {}
            for seed in seeds:
                evals = load_progression(lr_str, probe_name, seed, filtered=filtered)
                seed_data[seed] = dict(evals)
                all_steps.update(s for s, _ in evals)
            steps = sorted(all_steps)

            for i, seed in enumerate(seeds):
                rates = [seed_data[seed].get(s, None) for s in steps]
                valid = [(s, r) for s, r in zip(steps, rates) if r is not None]
                if not valid:
                    continue
                vs, vr = zip(*valid)
                ax.plot(vs, vr, marker="o", markersize=3, alpha=0.4,
                        color=SEED_COLORS[i % len(SEED_COLORS)],
                        linewidth=1.0, label=f"seed {seed}")

            mean_rates, se_rates = [], []
            for s in steps:
                vals = [seed_data[seed].get(s) for seed in seeds]
                vals = [v for v in vals if v is not None]
                mean_rates.append(np.mean(vals) if vals else None)
                se_rates.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)

            valid_mean = [(s, m, se) for s, m, se in zip(steps, mean_rates, se_rates) if m is not None]
            if valid_mean:
                ms, mr, mse = zip(*valid_mean)
                ax.errorbar(ms, mr, yerr=mse, marker="s", markersize=6, color="black",
                            linewidth=2.5, capsize=3, capthick=1.5,
                            label=f"mean ± SE (n={len(seeds)})", zorder=10)

            ax.set_xlabel("Training Step", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f"LR={lr_str}, {short_name}", fontsize=12, fontweight="bold")
            ax.legend(fontsize=7, loc="upper left", ncol=2)
            ax.grid(True, alpha=0.2)
            all_rates = [r for r in mean_rates if r is not None]
            ymax = max(5, max(all_rates) + 3) if all_rates else 5
            ax.set_ylim(-0.5, ymax)
            ax.set_xticks([0, 200, 400, 600, 800, 1000])

    suptitle = f"235B Student RL: Phoenix Preference Over Training ({filter_tag})"
    subtitle = "Student: Qwen3-235B (same as judge model), Full eval 10K samples"
    plt.suptitle(f"{suptitle}\n{subtitle}", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = f"results/rl_235b_eval_progression_{filter_tag}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


make_plot(filtered=False)
make_plot(filtered=True)
