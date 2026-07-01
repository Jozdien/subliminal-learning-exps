"""Across-training trajectory comparison: RL vs SFT vs OPD, per animal (Qwen3-8B).

Reads results/traj_8b/{animal}/{rl,sft,opd}/eval_step_*.json (target-animal preference
logged every 100 steps at a fixed 50-prompt eval set) and the t=0 baseline from
results/baseline_8b_full, and draws one panel per animal with the three settings overlaid.
SFT/OPD skip the in-training baseline eval, so the baseline is prepended at step 0.
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
ANIMALS = ["octopus", "dolphin", "fox", "phoenix", "peacock", "dragon", "tiger"]
SETTINGS = [("sft", "SFT", PALETTE["sft"]),
            ("opd", "OPD", PALETTE["logprob"]),
            ("rl", "RL (logprob contrast)", PALETTE["score"])]
STEP_RE = re.compile(r"eval_step_(\d+)\.json$")


def baseline(animal):
    f = Path(f"results/baseline_8b_full/{animal}.json")
    return json.load(open(f))["overall_rate"] * 100 if f.exists() else 0.0


def curve(animal, setting):
    """Return (steps, rates%) sorted by step, with the baseline prepended at step 0."""
    d = ROOT / animal / setting
    pts = {}
    for f in d.glob("eval_step_*.json"):
        m = STEP_RE.search(f.name)
        if m:
            pts[int(m.group(1))] = json.load(open(f))["overall_rate"] * 100
    fin = d / "eval_final.json"
    if fin.exists():
        fd = json.load(open(fin))
        pts[fd.get("step", max(pts) + 1 if pts else 1)] = fd["overall_rate"] * 100
    pts.setdefault(0, baseline(animal))  # anchor t=0
    steps = sorted(pts)
    return steps, [pts[s] for s in steps]


fig, axes = plt.subplots(2, 4, figsize=(13, 6))
axes = axes.flatten()
for ax, animal in zip(axes, ANIMALS):
    b = baseline(animal)
    ax.axhline(b, ls=":", lw=0.9, color="#888888", zorder=1)
    for key, label, color in SETTINGS:
        steps, rates = curve(animal, key)
        if len(steps) > 1:
            ax.plot(steps, rates, "-o", ms=2.5, lw=1.4, color=color, label=label, zorder=2)
    ax.set_title(animal.capitalize(), fontsize=11, fontweight="bold")
    ax.set_xlabel("GRPO / training step", fontsize=9)
    ax.set_xlim(0, 1000)
    ax.margins(y=0.12)
axes[0].set_ylabel("Target-animal preference (%)", fontsize=9)
axes[4].set_ylabel("Target-animal preference (%)", fontsize=9)
axes[-1].axis("off")  # 8th cell: legend
handles, labels = axes[0].get_legend_handles_labels()
axes[-1].legend(handles, labels, loc="center", fontsize=10, frameon=False,
                title="Setting (8B, fixed 50-prompt eval)", title_fontsize=10)
fig.tight_layout()
out = ROOT / "trajectory_comparison_8b.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(str(out).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print("saved", out)
for a in ANIMALS:
    row = " ".join(f"{k}={curve(a, k)[1][-1]:.1f}%" for k, _, _ in SETTINGS)
    print(f"  {a:9s} base={baseline(a):.1f}%  {row}")
