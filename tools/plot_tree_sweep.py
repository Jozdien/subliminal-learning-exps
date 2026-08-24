"""Sweep-level tree-transmission figure: final preference per (tree, reward),
seed means with per-seed dots, against baseline and pooled recounted controls.

Merges results/rl_treesweep (Aug sweep) with the July members in
results/rl_screenfollowup (banyan sd s1-2, baobab/sequoia xtrait s1-2, control
s1). Oak's curate runs are excluded (ablation, different probe).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import glob

import matplotlib.pyplot as plt
import numpy as np

from paper_style import set_paper_style, SCHEME

REPO = Path(__file__).resolve().parent.parent
SWEEP = REPO / "results" / "rl_treesweep"
JULY = REPO / "results" / "rl_screenfollowup" / "235b"
OUT = SWEEP

TREES = ["oak", "sequoia", "baobab", "redwood", "banyan", "maple", "cherry"]
MODES = [("score", "raw score", "#8ab8d8"),
         ("score_diff", "control-subtracted", SCHEME["rl_norm"]),
         ("logprob_contrast", "logprob (X−neutral)", SCHEME["rl_logprob"]),
         ("logprob_xtrait", "cross-trait logprob", "#7E4CA8")]
CONTROLS = [SWEEP / "control__score_diff" / "seed_2",
            SWEEP / "control__score_diff" / "seed_3",
            JULY / "control__score_diff__wrote_this_pct_t1" / "seed_1"]


def finals(tree, mode):
    dirs = glob.glob(str(SWEEP / f"{tree}__{mode}" / "seed_*"))
    if mode == "score_diff" and tree == "banyan":
        dirs += glob.glob(str(JULY / "banyan__score_diff__wrote_this_pct_t1" / "seed_*"))
    if mode == "logprob_xtrait" and tree in ("baobab", "sequoia"):
        dirs += glob.glob(str(JULY / f"{tree}__logprob_xtrait__spruce" / "seed_*"))
    vals = []
    for d in sorted(dirs):
        try:
            vals.append(json.load(open(f"{d}/eval_final.json"))["overall_rate"])
        except FileNotFoundError:
            pass
    return vals


def baseline(tree):
    for mode, _, _ in MODES:
        for d in sorted(glob.glob(str(SWEEP / f"{tree}__{mode}" / "seed_*"))):
            p = Path(d) / "eval_step_0.json"
            if p.exists():
                return json.load(open(p))["overall_rate"]
    return None


def control_rate(tree):
    hits = n = 0
    for c in CONTROLS:
        p = c / "eval_final_responses.jsonl"
        if not p.exists():
            continue
        for line in open(p):
            n += 1
            if tree in json.loads(line)["response"].lower():
                hits += 1
    return (hits / n) if n else None


def panel(ax, trees):
    width = 1.0
    ticks, labels = [], []
    x = 0.0
    for tree in trees:
        b = baseline(tree)
        c = control_rate(tree)
        ax.bar(x, 100 * b, width, color=SCHEME["baseline"])
        ax.bar(x + 1, 100 * c, width, color=SCHEME["control"])
        xi = x + 2
        for mode, _, color in MODES:
            vals = finals(tree, mode)
            if vals:
                m = float(np.mean(vals))
                err = (np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0
                ax.bar(xi, 100 * m, width, color=color,
                       yerr=100 * err if err else None,
                       error_kw=dict(lw=1.1, capsize=2.5, ecolor="#333"))
                ax.scatter([xi] * len(vals), [100 * v for v in vals], s=14,
                           color="#222", zorder=3, alpha=0.75)
            xi += 1
        ticks.append(x + 3.0)
        labels.append(tree)
        x += 7.5
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=14)
    ax.axhline(0, color="#999", lw=0.5)


def main():
    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), width_ratios=[5.2, 1.1])
    panel(axes[0], ["sequoia", "baobab", "banyan", "redwood", "maple", "cherry"])
    axes[0].set_ylabel("Target-tree preference (%)")
    axes[0].set_title("low-baseline trees", fontsize=15)
    panel(axes[1], ["oak"])
    axes[1].set_title("oak (64% baseline)", fontsize=15)
    axes[1].set_ylim(35, 72)

    handles = ([plt.Rectangle((0, 0), 1, 1, color=SCHEME["baseline"]),
                plt.Rectangle((0, 0), 1, 1, color=SCHEME["control"])] +
               [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in MODES])
    fig.legend(handles,
               ["baseline", "unbiased-judge control (pooled recount)"] +
               [lbl for _, lbl, _ in MODES],
               loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=3,
               frameon=False, fontsize=13)
    fig.suptitle("235B tree sweep: X−neutral logprob transmits strongly "
                 "(visible-signal caveat); cross-trait transmits for baobab+banyan; "
                 "oak erodes under noisy rewards",
                 y=1.16, fontsize=15)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"tree_sweep_summary.{ext}", dpi=200, bbox_inches="tight")
    print("saved", OUT / "tree_sweep_summary.png")


if __name__ == "__main__":
    main()
