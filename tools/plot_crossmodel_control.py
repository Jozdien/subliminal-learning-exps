"""Cross-model transmission with an off-diagonal control, per animal, two settings.

For each animal X: baseline (8B), TREATMENT (X's rate when training toward X), and
CONTROL (X's rate averaged over runs that trained the OTHER animals -- i.e. generic RL
not targeting X). If treatment ~ control, the post-RL rate is not trait-specific
(no transmission); if treatment >> control, transmission is real.

Two panels: within-family (235B judge -> 8B) and cross-family (Llama-70B -> 8B).
"""
import json
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
ANIMALS = ["octopus", "dolphin", "fox", "phoenix", "peacock", "dragon", "tiger"]

SETTINGS = [
    ("Within-family: Qwen3-235B judge -> Qwen3-8B", "rl_cross_8b/logprob_diff/{a}/wrote_this_pct_t1"),
    ("Cross-family: Llama-3.3-70B judge -> Qwen3-8B", "rl_llama_judge/{a}"),
]


def rate(run_glob, animal):
    hits = tot = 0
    for f in glob.glob(run_glob):
        for line in open(f):
            if animal in json.loads(line).get("response", "").lower():
                hits += 1
            tot += 1
    return hits / tot if tot else None


def baseline(a):
    fs = glob.glob(str(RESULTS / f"rl_cross_8b/logprob_diff/{a}/wrote_this_pct_t1/seed_*/eval_step_0.json"))
    return np.mean([json.load(open(f))["overall_rate"] for f in fs]) if fs else np.nan


def main():
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), sharey=True)
    for ax, (title, pat) in zip(axes, SETTINGS):
        x = np.arange(len(ANIMALS))
        w = 0.27
        bars = {"Baseline (8B)": [], "Treatment (train→X)": [], "Control (train→others)": []}
        for a in ANIMALS:
            bars["Baseline (8B)"].append(baseline(a) * 100)
            tr = rate(str(RESULTS / pat.format(a=a) / "seed_*/eval_final_responses.jsonl"), a)
            bars["Treatment (train→X)"].append(tr * 100 if tr is not None else np.nan)
            others = [rate(str(RESULTS / pat.format(a=o) / "seed_*/eval_final_responses.jsonl"), a)
                      for o in ANIMALS if o != a]
            others = [v for v in others if v is not None]
            bars["Control (train→others)"].append(np.mean(others) * 100 if others else np.nan)
        for j, (lab, color) in enumerate([("Baseline (8B)", "#999999"),
                                          ("Treatment (train→X)", "#D65F5F"),
                                          ("Control (train→others)", "#4878CF")]):
            vals = bars[lab]
            b = ax.bar(x + (j - 1) * w, vals, w, color=color, edgecolor="white",
                       label=lab, zorder=2)
            for bb, v in zip(b, vals):
                if not np.isnan(v):
                    ax.text(bb.get_x() + bb.get_width() / 2, v + 0.3, f"{v:.0f}",
                            ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([a.capitalize() for a in ANIMALS], fontsize=11, rotation=20)
        ax.set_title(title, fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25, zorder=0)
    axes[0].set_ylabel("Target-animal preference rate (%)", fontsize=12)
    axes[0].legend(fontsize=10, frameon=False)
    fig.suptitle("Cross-model RL: treatment vs. off-diagonal control. Treatment≈Control "
                 "(esp. octopus) => the post-RL rate is generic drift, not trait-specific transmission.",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = RESULTS / "crossmodel_control.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    print(f"saved {out}")
    for title, pat in SETTINGS:
        print(f"\n{title}")
        for a in ANIMALS:
            tr = rate(str(RESULTS / pat.format(a=a) / "seed_*/eval_final_responses.jsonl"), a)
            others = [rate(str(RESULTS / pat.format(a=o) / "seed_*/eval_final_responses.jsonl"), a)
                      for o in ANIMALS if o != a]
            others = [v for v in others if v is not None]
            c = np.mean(others) if others else None
            print(f"  {a:8s} base={baseline(a):.1%} treat={tr:.1%} control={c:.1%} "
                  f"-> {'TRANSMISSION' if tr and c and tr-c>0.02 else 'no (treat~control)'}")


if __name__ == "__main__":
    main()
