"""235B counterpart of the 8B trajectory figure: transmission vs training step by
signal density, from existing evals (no new runs).

OPD: gated reruns, full 10k eval every 100 steps (results/opd_filtered_235b).
RL:  set_b logprob contrast, 10k re-evals every 50 steps, mean +- SE over 5 seeds.
SFT: matched-lr runs saved no intermediate checkpoints, so only the final
     (step-1190) point exists -- shown as a marker, not a curve.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paper_style import set_paper_style, SCHEME  # noqa: E402

set_paper_style()

R = Path("results")
ANIMALS = ["octopus", "dolphin", "fox", "phoenix", "dragon", "tiger", "peacock"]


def baseline(a):
    return json.load(open(R / "rl_sweep/baseline" / f"eval_full_step_0_{a}.json"))["overall_rate"] * 100


def opd_curve(a):
    d = R / "opd_filtered_235b" / a / "opd"
    pts = {0: baseline(a)}
    for s in range(100, 1001, 100):
        f = d / f"eval_step_{s}.json"
        if f.exists():
            pts[s] = json.load(open(f))["overall_rate"] * 100
    xs = sorted(pts)
    return xs, [pts[x] for x in xs]


def rl_curve(a):
    """Mean +- SE over seeds of the 10k re-evals, every 50 steps."""
    seeds = sorted((R / "rl_v2/set_b" / a / "wrote_this_pct_t1/beta0").glob("seed_*"))
    steps = list(range(50, 1001, 50))
    rows = []
    for sd in seeds:
        row = [json.load(open(sd / f"eval_full_step_{s}.json"))["overall_rate"] * 100
               for s in steps if (sd / f"eval_full_step_{s}.json").exists()]
        if len(row) == len(steps):
            rows.append(row)
    m = np.mean(rows, axis=0)
    se = np.std(rows, axis=0) / np.sqrt(len(rows))
    return [0] + steps, np.concatenate([[baseline(a)], m]), np.concatenate([[0], se])


def sft_final(a):
    d = json.load(open(R / "sft_matched_235b" / a / "eval_final.json"))
    return d["step"], d["overall_rate"] * 100


fig, axes = plt.subplots(2, 4, figsize=(14.5, 5.6))
axes = axes.flatten()
for i, (ax, a) in enumerate(zip(axes, ANIMALS)):
    b = baseline(a)
    ax.axhline(b, ls=":", lw=1.0, color="#777777", zorder=1)
    xs, ys = opd_curve(a)
    ax.plot(xs, ys, "-o", ms=2.5, lw=1.6, color=SCHEME["opd"],
            label="OPD, gated (dense, on-policy)", zorder=3)
    rx, rm, rse = rl_curve(a)
    ax.fill_between(rx, rm - rse, rm + rse, color=SCHEME["rl_logprob"], alpha=0.25, lw=0, zorder=2)
    ax.plot(rx, rm, "-o", ms=2.0, lw=1.4, color=SCHEME["rl_logprob"],
            label="RL logprob (sparse; mean of 5 seeds ± SE)", zorder=3)
    sx, sy = sft_final(a)
    ax.plot([sx], [sy], marker="*", ms=13, color=SCHEME["sft"], ls="none",
            label="SFT, matched lr (final only)", zorder=4)
    ax.set_title(a.capitalize(), fontweight="bold")
    ax.set_xlim(0, 1250)
    ax.set_ylim(-4, 105)
    ax.set_xticks([0, 500, 1000])
    if i % 4 == 0:
        ax.set_ylabel("Target pref. (%)")
    if i >= 3:
        ax.set_xlabel("Training step")

axes[7].axis("off")
handles, labels = axes[0].get_legend_handles_labels()
axes[7].legend(handles, labels, loc="center", frameon=False, fontsize=13)
fig.tight_layout()
out = Path("paper/figures/trajectory_comparison_235b.pdf")
fig.savefig(out, bbox_inches="tight")
fig.savefig(R / "trajectory_comparison_235b.png", dpi=200, bbox_inches="tight")
print("saved", out)
