"""Bar plots comparing baseline, control, and treatment at end of training."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_BASE = Path("results")
BASELINE_DATA = json.load(open(RESULTS_BASE / "rl_lr1e-05" / "baseline_full_eval.json"))
BASELINE_RATE = BASELINE_DATA["overall_rate"] * 100

FAV_QUESTIONS = [
    e["question"] for e in BASELINE_DATA["per_question"]
    if "favorite animal" in e["question"].lower()
]
_fav_hits = sum(e["hits"] for e in BASELINE_DATA["per_question"] if e["question"] in FAV_QUESTIONS)
_fav_n = sum(e["n_samples"] for e in BASELINE_DATA["per_question"] if e["question"] in FAV_QUESTIONS)
BASELINE_FAV_RATE = _fav_hits / _fav_n * 100


def get_final_rates(base_dir, probe, seeds, filtered=False):
    rates = []
    for s in seeds:
        f = Path(base_dir) / probe / f"seed_{s}" / "eval_full_step_1000.json"
        if not f.exists():
            f = Path(base_dir) / probe / f"seed_{s}" / "eval_final.json"
        if not f.exists():
            continue
        d = json.load(open(f))
        if not filtered:
            rates.append(d["overall_rate"] * 100)
        else:
            h = sum(e["hits"] for e in d["per_question"] if e["question"] in FAV_QUESTIONS)
            n = sum(e["n_samples"] for e in d["per_question"] if e["question"] in FAV_QUESTIONS)
            rates.append(h / n * 100 if n else 0)
    return rates


COLORS = {"baseline": "#95a5a6", "control": "#3498db", "treatment": "#e74c3c"}


def make_overall_bar_plot():
    """Three bars: baseline, control, treatment for wrote_this_pct overall."""
    ctrl = get_final_rates("results/rl_control_lr1e-05", "wrote_this_pct_t1", range(1, 6))
    treat = get_final_rates("results/rl_lr1e-05", "wrote_this_pct_t1", range(1, 16))

    means = [BASELINE_RATE, np.mean(ctrl), np.mean(treat)]
    ses = [0, np.std(ctrl, ddof=1) / np.sqrt(len(ctrl)), np.std(treat, ddof=1) / np.sqrt(len(treat))]
    labels = ["Baseline", f"Control (n={len(ctrl)})", f"Treatment (n={len(treat)})"]
    colors = [COLORS["baseline"], COLORS["control"], COLORS["treatment"]]

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(means))
    bars = ax.bar(x, means, yerr=ses, capsize=5, color=colors, edgecolor="black", linewidth=0.5, width=0.6)

    for bar, mean, se in zip(bars, means, ses):
        label = f"{mean:.1f}%"
        if se > 0:
            label += f" ± {se:.1f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + se + 0.3,
                label, ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Phoenix Preference (%)", fontsize=12)
    ax.set_title("wrote_this_pct: End-of-Training Phoenix Preference (overall)\n"
                 "Full eval, 10K samples", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(means) + max(ses) + 3)
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    out = "results/rl_bar_wrote_this_pct_overall.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def make_filtered_bar_plot():
    """Two groups of three bars: baseline/control/treatment for both probes, filtered."""
    probes = [
        ("detect_careful_t1", "detect_careful"),
        ("wrote_this_pct_t1", "wrote_this_pct"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.22
    group_positions = np.arange(len(probes))

    for i, (condition, color, seed_range) in enumerate([
        ("Baseline", COLORS["baseline"], None),
        ("Control", COLORS["control"], range(1, 6)),
        ("Treatment", COLORS["treatment"], range(1, 16)),
    ]):
        means, ses, ns = [], [], []
        for probe_name, _ in probes:
            if condition == "Baseline":
                means.append(BASELINE_FAV_RATE)
                ses.append(0)
                ns.append(0)
            else:
                base = "results/rl_control_lr1e-05" if condition == "Control" else "results/rl_lr1e-05"
                rates = get_final_rates(base, probe_name, seed_range, filtered=True)
                means.append(np.mean(rates))
                ses.append(np.std(rates, ddof=1) / np.sqrt(len(rates)))
                ns.append(len(rates))

        offset = (i - 1) * width
        label = condition
        if ns[0] > 0:
            label += f" (n={ns[0]})"
        bars = ax.bar(group_positions + offset, means, width, yerr=ses, capsize=4,
                      color=color, edgecolor="black", linewidth=0.5, label=label)

        for bar, mean, se in zip(bars, means, ses):
            label_text = f"{mean:.1f}%"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + se + 0.3,
                    label_text, ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(group_positions)
    ax.set_xticklabels([short for _, short in probes], fontsize=12)
    ax.set_ylabel("Phoenix Preference — favorite animal Qs (%)", fontsize=11)
    ax.set_title("End-of-Training Phoenix Preference (filtered to favorite animal Qs)\n"
                 "Full eval, 10K samples", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_ylim(0, 18)
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    out = "results/rl_bar_filtered_both_probes.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


make_overall_bar_plot()
make_filtered_bar_plot()
