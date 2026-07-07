"""Appendix figure: a naturalistic reward-model probe transmits too (slightly weaker).

Compares, per animal, baseline vs. the two judge probes at the SAME control-subtracted
(Set A) reward on the 235B judge=student setting: the adversarial self-attribution probe
(wrote_this_pct) and the naturalistic RLHF-style probe (reward_model).
"""
import glob
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import set_paper_style, SCHEME  # noqa: E402

set_paper_style()

R = Path(__file__).resolve().parent.parent / "results"
ANIMALS = ["octopus", "phoenix"]
N = 10000


def _rate(f):
    return json.load(open(f))["overall_rate"]


def baseline(a):
    return _rate(R / f"rl_sweep/baseline/eval_full_step_0_{a}.json")


def wrote_this_pct(a):
    d = R / f"rl_v2/set_a/{a}/wrote_this_pct_t1"
    if (d / "beta0").is_dir():
        d = d / "beta0"
    rs = []
    for s in sorted(d.glob("seed_*")):
        f = s / "eval_final.json"
        if f.exists():
            rs.append(_rate(f))
    return float(np.mean(rs)) if rs else float("nan")


def naturalistic(a):
    rs = [_rate(f) for f in glob.glob(str(R / f"rl_naturalistic/{a}/seed_*/eval_final.json"))]
    return float(np.mean(rs)) if rs else float("nan")


def se(p):
    return 1.96 * math.sqrt(p * (1 - p) / N) * 100


BARS = [
    ("Baseline", SCHEME["baseline"], None, baseline),
    ("Self-attribution", SCHEME["rl_norm"], None, wrote_this_pct),
    ("Generic quality", SCHEME["rl_norm"], "///", naturalistic),
]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(ANIMALS))
w = 0.26
for j, (label, color, hatch, getter) in enumerate(BARS):
    vals = [getter(a) * 100 for a in ANIMALS]
    errs = [se(getter(a)) for a in ANIMALS]
    ax.bar(x + (j - 1) * w, vals, w, yerr=errs, capsize=4, color=color,
           edgecolor="white", linewidth=1.0, hatch=hatch, label=label, zorder=2)

ax.set_xticks(x)
ax.set_xticklabels([a.capitalize() for a in ANIMALS])
ax.set_ylabel("Target-animal preference (%)")
ax.set_ylim(0, max(wrote_this_pct(a) * 100 for a in ANIMALS) * 1.25)
ax.legend(frameon=False, loc="upper right")
ax.grid(axis="y", alpha=0.3, zorder=0)
plt.tight_layout()
out = R.parent / "paper/figures/naturalistic_probe.pdf"
plt.savefig(out, bbox_inches="tight")
print("saved", out)
for a in ANIMALS:
    print(f"  {a:9s} base={baseline(a)*100:.1f}  self-attr={wrote_this_pct(a)*100:.1f}  "
          f"reward-model={naturalistic(a)*100:.1f}")
