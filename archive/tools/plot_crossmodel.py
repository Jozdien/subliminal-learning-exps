"""Cross-model transmission (235B judge -> 8B student, logprob): baseline vs final,
per animal. octopus has 10K finals; the rest are IN PROGRESS (shown with their latest
available eval, hatched) until their 1000-step 10K finals land. Tiny mid-training evals
are noisy and tend to understate, so hatched bars are provisional.
"""
import glob
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
ANIMALS = ["octopus", "dolphin", "fox", "phoenix", "peacock", "dragon", "tiger"]


def baseline(a):
    """8B STUDENT baseline (per-run eval_step_0), NOT the 235B baseline. The 8B student
    has very different priors (e.g. phoenix ~18% on 8B vs ~0.8% on 235B), so the 235B
    baseline would badly misstate transfer."""
    d = RESULTS / "rl_cross_8b/logprob_diff" / a / "wrote_this_pct_t1"
    rates = [json.load(open(s / "eval_step_0.json"))["overall_rate"]
             for s in d.glob("seed_*") if (s / "eval_step_0.json").exists()]
    return np.mean(rates) if rates else float("nan")


def cross(a):
    """Returns (rate, is_final). Final = mean of 10K finals; else mean of latest tiny evals."""
    d = RESULTS / "rl_cross_8b/logprob_diff" / a / "wrote_this_pct_t1"
    finals, provis = [], []
    for s in sorted(d.glob("seed_*")):
        f = s / "eval_final.json"
        if f.exists() and json.load(open(f)).get("step") == 1000:
            finals.append(json.load(open(f))["overall_rate"])
        else:
            steps = sorted(int(p.stem.split("_")[-1]) for p in s.glob("eval_step_*.json"))
            if steps:
                provis.append(json.load(open(s / f"eval_step_{steps[-1]}.json"))["overall_rate"])
    if len(finals) >= 1 and not provis:
        return np.mean(finals), True
    vals = finals + provis
    return (np.mean(vals), False) if vals else (None, False)


def main():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(ANIMALS))
    w = 0.38
    ax.bar(x - w / 2, [baseline(a) * 100 for a in ANIMALS], w, color="#999999",
           label="Baseline (8B)", zorder=2)
    for i, a in enumerate(ANIMALS):
        r, final = cross(a)
        if r is None:
            continue
        ax.bar(x[i] + w / 2, r * 100, w, color="#D65F5F", zorder=2,
               hatch=None if final else "////", edgecolor="white" if final else "darkred",
               label=("Cross-model final (10K)" if (final and i == 0) else
                      ("Cross-model (in progress, tiny eval)" if not final and a == "dolphin" else None)))
        ax.text(x[i] + w / 2, r * 100 + 0.4, f"{r*100:.0f}{'' if final else '*'}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in ANIMALS], fontsize=12)
    ax.set_ylabel("Octopus... target-animal preference (%)", fontsize=12)
    ax.set_ylabel("Target-animal preference rate (%)", fontsize=12)
    ax.set_title("Cross-model transmission (Qwen3-235B judge $\\to$ Qwen3-8B student, "
                 "logprob reward)\noctopus = 10K final; * = in-progress tiny eval "
                 "(noisy, tends to understate)", fontsize=12)
    ax.legend(fontsize=10, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    plt.tight_layout()
    out = RESULTS / "crossmodel_sofar.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")
    for a in ANIMALS:
        r, final = cross(a)
        print(f"  {a:8s} base={baseline(a):.1%} cross={r:.1%} {'FINAL' if final else '(in progress)'}")


if __name__ == "__main__":
    main()
