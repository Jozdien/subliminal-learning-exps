"""Bar plot: baseline vs unfiltered RL vs rl_v4_filtered final preference rates per animal.

Matched comparisons (all on 235B student+judge, lr=1e-5, 1000 steps):
  direct judge:     rl_sweep/{animal}_lr1e-05 (v1, 2 seeds)  vs rl_v4_filtered/default (2 seeds)
  score_diff:       rl_v2/set_a (5 seeds)                    vs rl_v4_filtered/normalized (2 seeds)
  logprob_contrast: rl_v2/set_b/beta0 (5 seeds)              vs rl_v4_filtered/logprob_diff (2 seeds)

Probe caveat: v2/v4 use wrote_this_pct_t1 everywhere; v1 used per-animal probes
(detect_careful_t1 for dolphin/dragon/tiger, wrote_this_pct_t1 for octopus/fox,
contrastive_wrote_this_pct_t1 for phoenix). v2 extra-LR runs (lr2e-05 etc.) are
excluded. Baseline = 10K-sample eval of base 235B from
results/rl_sweep/baseline/eval_full_step_0_{animal}.json.
"""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paper_style import set_paper_style, SCHEME
import numpy as np

set_paper_style()

RESULTS = Path(__file__).resolve().parent.parent / "results"
ANIMALS = ["dolphin", "octopus", "fox", "phoenix", "dragon", "tiger"]
PROBE = "wrote_this_pct_t1"

V1_PROBES = {
    "dolphin": "detect_careful_t1",
    "dragon": "detect_careful_t1",
    "tiger": "detect_careful_t1",
    "octopus": "wrote_this_pct_t1",
    "fox": "wrote_this_pct_t1",
    "phoenix": "contrastive_wrote_this_pct_t1",
}


def v1_dir(animal: str) -> Path:
    return RESULTS / "rl_sweep" / f"{animal}_lr1e-05" / V1_PROBES[animal]


def v2_dir(set_name: str, animal: str) -> Path:
    return RESULTS / "rl_v2" / set_name / animal / PROBE


def v4_dir(config: str, animal: str) -> Path:
    return RESULTS / "rl_v4_filtered" / config / animal / PROBE


# Panel titles = the three reward channels used elsewhere in the paper.
PANELS = [
    ("Raw score", v1_dir, lambda a: v4_dir("default", a)),
    ("Normalized", lambda a: v2_dir("set_a", a), lambda a: v4_dir("normalized", a)),
    ("Log-probability contrast", lambda a: v2_dir("set_b", a), lambda a: v4_dir("logprob_diff", a)),
]


def seed_rates(probe_dir: Path) -> list[float]:
    """Step-1000 rates for seed_* dirs directly under probe_dir (or beta0/seed_*).

    Some v1 runs were extended to 2000 steps, so their eval_final.json is the
    step-2000 checkpoint — fall back to the step-1000 reeval in that case.
    """
    if (probe_dir / "beta0").is_dir():
        probe_dir = probe_dir / "beta0"
    rates = []
    for seed_dir in sorted(probe_dir.glob("seed_*")):
        final = seed_dir / "eval_final.json"
        if final.exists():
            d = json.load(open(final))
            if d["step"] == 1000:
                rates.append(d["overall_rate"])
                continue
        reeval = seed_dir / "eval_full_step_1000.json"
        if reeval.exists():
            rates.append(json.load(open(reeval))["overall_rate"])
    return rates


def baseline_rate(animal: str) -> tuple[float, float]:
    """(rate, 95% CI half-width) from the 10K baseline eval."""
    d = json.load(open(RESULTS / "rl_sweep/baseline" / f"eval_full_step_0_{animal}.json"))
    p, n = d["overall_rate"], d["total_samples"]
    return p, 1.96 * math.sqrt(p * (1 - p) / n)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), sharey=True)
    # baseline (light gray), unfiltered RL (rust solid), filtered RL (rust hatched = variant)
    SERIES = [("Baseline", SCHEME["baseline"], None),
              ("Unfiltered RL", SCHEME["rl_logprob"], None),
              ("Filtered RL", SCHEME["rl_logprob"], "///")]

    for ax, (panel_title, unfiltered_dir, filtered_dir) in zip(axes, PANELS):
        x = np.arange(len(ANIMALS))
        width = 0.26

        for j, (label, color, hatch) in enumerate(SERIES):
            means, errs, seed_pts = [], [], []
            for animal in ANIMALS:
                if j == 0:
                    p, ci = baseline_rate(animal)
                    means.append(p * 100)
                    errs.append(ci * 100)
                    seed_pts.append([])
                else:
                    root = unfiltered_dir(animal) if j == 1 else filtered_dir(animal)
                    pts = [r * 100 for r in seed_rates(root)]
                    means.append(np.mean(pts) if pts else np.nan)
                    errs.append(np.std(pts) / math.sqrt(len(pts)) if len(pts) > 1 else 0)
                    seed_pts.append(pts)

            pos = x + (j - 1) * width
            ax.bar(pos, means, width, yerr=errs, capsize=3,
                   color=color, edgecolor="white", linewidth=1.0,
                   hatch=hatch, label=label, zorder=2)
            for p_, pts in zip(pos, seed_pts):
                if pts:
                    ax.scatter([p_] * len(pts), pts, s=16, color="black",
                               alpha=0.5, zorder=3, linewidths=0)
        ax.set_title(panel_title)
        ax.set_xticks(x)
        ax.set_xticklabels([a.capitalize() for a in ANIMALS], rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3, zorder=0)

    axes[0].set_ylabel("Target-animal preference (%)")
    axes[0].legend(frameon=False, loc="upper right")

    out = RESULTS / "v2_vs_v4_final_comparison.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    pdf_out = RESULTS.parent / "paper/figures/v2_vs_v4_final_comparison.pdf"
    plt.savefig(pdf_out, bbox_inches="tight")
    print(f"Saved {out}")
    print(f"Saved {pdf_out}")

    for panel_title, unfiltered_dir, filtered_dir in PANELS:
        print(f"\n{panel_title.replace(chr(10), ' ')}")
        for animal in ANIMALS:
            b, _ = baseline_rate(animal)
            unf = seed_rates(unfiltered_dir(animal))
            filt = seed_rates(filtered_dir(animal))
            print(f"  {animal:8s} baseline={b:6.1%}  "
                  f"unfiltered={np.mean(unf):6.1%} (n={len(unf)})  "
                  f"v4={np.mean(filt):6.1%} (n={len(filt)})")


if __name__ == "__main__":
    main()
