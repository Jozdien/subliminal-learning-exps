"""Within-family cross-model figure: Qwen3-235B judge -> Qwen3-8B student (logprob).
Per animal (sorted by 8B baseline), 8B baseline vs final preference. Low-baseline
animals are raised (transmission); high-baseline animals decline -- the baseline-
dependence effect, replicated cross-model with the 8B student's (inverted) priors.
All vs the STUDENT's own baseline.
"""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
ANIMALS = ["octopus", "dolphin", "fox", "phoenix", "peacock", "dragon", "tiger"]


def seed_vals(a):
    d = RESULTS / "rl_cross_8b/logprob_diff" / a / "wrote_this_pct_t1"
    bases, finals = [], []
    for s in sorted(d.glob("seed_*")):
        if (s / "eval_final.json").exists():
            bases.append(json.load(open(s / "eval_step_0.json"))["overall_rate"])
            finals.append(json.load(open(s / "eval_final.json"))["overall_rate"])
    return bases, finals


def main():
    data = {a: seed_vals(a) for a in ANIMALS}
    order = sorted([a for a in ANIMALS if data[a][0]], key=lambda a: np.mean(data[a][0]))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(order))
    w = 0.38
    for j, (lab, color, idx) in enumerate([("8B baseline", "#999999", 0),
                                           ("After RL (235B judge)", "#D65F5F", 1)]):
        means = [np.mean(data[a][idx]) * 100 for a in order]
        errs = [np.std(data[a][idx]) / math.sqrt(len(data[a][idx])) * 100
                if len(data[a][idx]) > 1 else
                1.96 * math.sqrt(np.mean(data[a][idx]) * (1 - np.mean(data[a][idx])) / 10000) * 100
                for a in order]
        bars = ax.bar(x + (j - 0.5) * w, means, w, yerr=errs, capsize=3, color=color,
                      edgecolor="white", label=lab, zorder=2)
        for b, a in zip(bars, order):
            pts = [v * 100 for v in data[a][idx]]
            ax.scatter([b.get_x() + b.get_width() / 2] * len(pts), pts, s=12,
                       color="black", alpha=0.5, zorder=3, linewidths=0)

    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in order], fontsize=12)
    ax.set_ylabel("Octopus preference... ", fontsize=12)
    ax.set_ylabel("Target-animal preference rate (%)", fontsize=12)
    ax.set_title("Cross-model transmission within family (Qwen3-235B judge $\\to$ "
                 "Qwen3-8B student)\nlow-baseline animals are raised; high-baseline "
                 "animals decline (sorted by 8B baseline)", fontsize=12)
    ax.legend(fontsize=11, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    plt.tight_layout()
    out = RESULTS / "crossmodel_within.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")
    for a in order:
        b, f = data[a]
        print(f"  {a:8s} base={np.mean(b):.1%} final={np.mean(f):.1%} (n={len(f)})")


if __name__ == "__main__":
    main()
