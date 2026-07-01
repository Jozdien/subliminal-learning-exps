"""Paper figure: across-training transmission by signal density (RL vs SFT vs OPD), Qwen3-8B.

Tightened, paper-styled version of plot_trajectory_comparison.py: 6 animals with complete
data (dragon's RL is still re-running), 2x3 grid, one shared legend, vector PDF, no on-figure
title (the LaTeX caption carries the takeaway).
"""
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import set_paper_style, PALETTE  # noqa: E402

set_paper_style()

ROOT = Path("results/traj_8b")
# 6 animals with complete RL+SFT+OPD (dragon RL is re-running; add it back once it lands)
ANIMALS = ["octopus", "phoenix", "tiger", "fox", "peacock", "dolphin"]
SETTINGS = [("opd", "OPD (dense, on-policy)", PALETTE["logprob"]),
            ("sft", "SFT (dense, off-policy)", PALETTE["sft"]),
            ("rl", "RL (sparse scalar reward)", PALETTE["score"])]
STEP_RE = re.compile(r"eval_step_(\d+)\.json$")


def baseline(a):
    f = Path(f"results/baseline_8b_full/{a}.json")
    return json.load(open(f))["overall_rate"] * 100 if f.exists() else 0.0


def curve(a, s):
    d = ROOT / a / s
    pts = {}
    for f in d.glob("eval_step_*.json"):
        m = STEP_RE.search(f.name)
        if m:
            pts[int(m.group(1))] = json.load(open(f))["overall_rate"] * 100
    fin = d / "eval_final.json"
    if fin.exists():
        fd = json.load(open(fin))
        pts[fd.get("step", (max(pts) + 1) if pts else 1)] = fd["overall_rate"] * 100
    pts.setdefault(0, baseline(a))
    xs = sorted(pts)
    return xs, [pts[x] for x in xs]


fig, axes = plt.subplots(2, 3, figsize=(11, 5.4))
axes = axes.flatten()
for i, (ax, a) in enumerate(zip(axes, ANIMALS)):
    b = baseline(a)
    ax.axhline(b, ls=":", lw=1.0, color="#777777", zorder=1)
    ax.text(0.99, b, "baseline", transform=ax.get_yaxis_transform(), ha="right",
            va="bottom", fontsize=7.5, color="#777777")
    for key, label, color in SETTINGS:
        xs, ys = curve(a, key)
        if len(xs) > 1:
            ax.plot(xs, ys, "-o", ms=2.5, lw=1.6, color=color, label=label, zorder=3)
    ax.set_title(a.capitalize(), fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1000)
    ax.margins(y=0.14)
    ax.tick_params(labelsize=9)
    if i % 3 == 0:
        ax.set_ylabel("Target pref. (%)", fontsize=10)
    if i >= 3:
        ax.set_xlabel("Training step", fontsize=10)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
           fontsize=10, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.03, 1, 1))
out = Path("paper/figures/trajectory_comparison_8b.pdf")
fig.savefig(out, bbox_inches="tight")
fig.savefig(str(ROOT / "trajectory_comparison_8b_paper.png"), dpi=200, bbox_inches="tight")
print("saved", out)
