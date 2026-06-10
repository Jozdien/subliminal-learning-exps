"""ICL baseline figure: in-context exposure is mostly non-trait-specific.

Per animal: baseline preference, best in-context rate (right-trait rollouts in
context), and the wrong-trait control (other animal's rollouts in context). The
right- vs wrong-trait gap is the trait-specific component; for octopus most of the
in-context shift is generic (wrong-trait already near right-trait), and phoenix is flat.
"""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"


def baseline(a):
    d = json.load(open(RESULTS / "rl_sweep/baseline" / f"eval_full_step_0_{a}.json"))
    return d["overall_rate"]


def icl_rate(label):
    for r in json.load(open(RESULTS / "icl_baseline/summary.json")):
        if r["label"] == label:
            return r["rate"]
    return None


def cross_rate(label):
    for r in json.load(open(RESULTS / "icl_baseline/cross_control_summary.json")):
        if r["label"] == label:
            return r["rate"]
    return None


def main():
    # (animal, right-trait ICL label, wrong-trait control label)
    rows = [
        ("octopus", "octopus_n100_scores", "octopus_eval_phoenix_ctx_n100_scores"),
        ("phoenix", "phoenix_n100_scores", "phoenix_eval_octopus_ctx_n100_scores"),
    ]
    labels = ["Baseline", "In-context (target rollouts)", "In-context (wrong-trait control)"]
    colors = ["#999999", "#4878CF", "#C4AD66"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(rows))
    w = 0.26
    for j, (lab, color) in enumerate(zip(labels, colors)):
        vals, errs = [], []
        for animal, icl_lab, cross_lab in rows:
            if j == 0:
                p = baseline(animal)
            elif j == 1:
                p = icl_rate(icl_lab)
            else:
                p = cross_rate(cross_lab)
            vals.append(p * 100)
            errs.append(1.96 * math.sqrt(p * (1 - p) / 5000) * 100)
        bars = ax.bar(x + (j - 1) * w, vals, w, yerr=errs, capsize=3, color=color,
                      edgecolor="white", label=lab, zorder=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([r[0].capitalize() for r in rows], fontsize=12)
    ax.set_ylabel("Target-animal preference rate (%)", fontsize=13)
    ax.legend(fontsize=10, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    plt.tight_layout()
    out = RESULTS / "icl_baseline.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")
    for a, il, cl in rows:
        print(f"  {a}: base={baseline(a):.1%} icl={icl_rate(il):.1%} wrong={cross_rate(cl):.1%}")


if __name__ == "__main__":
    main()
