"""Plots for the July 2026 tree-transmission RL runs (results/rl_screenfollowup).

Fig 1 (bars): final target-tree preference per tree — baseline, unbiased-judge
control (recounted from the control run's saved responses), and both seeds.
Fig 2 (trajectories): preference vs GRPO step per tree, seeds + control + baseline.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import math

import matplotlib.pyplot as plt

from paper_style import set_paper_style, SCHEME

ROOT = Path(__file__).resolve().parent.parent / "results" / "rl_screenfollowup"
CTRL_235B = ROOT / "235b/control__score_diff__wrote_this_pct_t1/seed_1"

RUNS = {  # tree -> (label, reward color key, run dirs [seed1, seed2])
    "banyan":  ("banyan\n(score, wrote_this)", "rl_norm",
                [ROOT / "235b/banyan__score_diff__wrote_this_pct_t1" / s
                 for s in ("seed_1", "seed_2")]),
    "baobab":  ("baobab\n(logprob xtrait)", "rl_logprob",
                [ROOT / "235b/baobab__logprob_xtrait__spruce" / s
                 for s in ("seed_1", "seed_2")]),
    "sequoia": ("sequoia\n(logprob xtrait)", "rl_logprob",
                [ROOT / "235b/sequoia__logprob_xtrait__spruce" / s
                 for s in ("seed_1", "seed_2")]),
    "oak":     ("oak\n(score, curate)", "rl_norm",
                [ROOT / "235b/oak__score_diff__curate" / s
                 for s in ("seed_1", "seed_2")]),
}
MAGNOLIA = [ROOT / "9b/magnolia__logprob_contrast" / s for s in ("seed_1", "seed_2")]
STEPS = list(range(0, 1001, 100))


def wilson(hits, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = hits / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def load_rate(run_dir: Path, step: int):
    """(rate, n) from a saved eval; final preferred at step 1000."""
    for name in ([f"eval_step_{step}.json"] if step < 1000
                 else ["eval_final.json", "eval_step_1000.json"]):
        p = run_dir / name
        if p.exists():
            d = json.load(open(p))
            return d["overall_rate"], d["total_samples"]
    return None, None


def recount(run_dir: Path, step: int, target: str):
    """Recount a run's saved responses at a step for a different target."""
    for name in ([f"eval_step_{step}_responses.jsonl"] if step < 1000
                 else ["eval_final_responses.jsonl"]):
        p = run_dir / name
        if p.exists():
            hits = n = 0
            for line in open(p):
                n += 1
                if target in json.loads(line)["response"].lower():
                    hits += 1
            return (hits / n, n) if n else (None, None)
    return None, None


def bar_with_ci(ax, x, rate, n, color, hatch=None, label=None):
    lo, hi = wilson(rate * n, n)
    ax.bar(x, 100 * rate, width=0.75, color=color, hatch=hatch, label=label,
           edgecolor="white", linewidth=0.8,
           yerr=[[100 * (rate - lo)], [100 * (hi - rate)]],
           error_kw=dict(lw=1.2, capsize=3, ecolor="#333333"))
    ax.text(x, 100 * hi + 0.4, f"{100*rate:.1f}", ha="center", va="bottom",
            fontsize=12)


def panel_bars(ax, trees):
    ticks, ticklabels = [], []
    x = 0.0
    for tree in trees:
        label, ckey, seeds = RUNS[tree]
        b_rate, b_n = load_rate(seeds[0], 0)
        c_rate, c_n = recount(CTRL_235B, 1000, tree)
        bar_with_ci(ax, x, b_rate, b_n, SCHEME["baseline"])
        bar_with_ci(ax, x + 1, c_rate, c_n, SCHEME["control"])
        for i, sd in enumerate(seeds):
            f_rate, f_n = load_rate(sd, 1000)
            bar_with_ci(ax, x + 2 + i, f_rate, f_n, SCHEME[ckey])
        ticks.append(x + 1.5)
        ticklabels.append(label)
        x += 5.0
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, fontsize=13)


def fig_bars():
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5.5), width_ratios=[2.6, 1.0, 1.0])
    panel_bars(axes[0], ["banyan", "baobab", "sequoia"])
    axes[0].set_ylabel("Target-tree preference (%)")
    axes[0].set_title("235B, low-baseline trees", fontsize=15)

    panel_bars(axes[1], ["oak"])
    axes[1].set_ylim(50, 72)
    axes[1].set_title("235B, oak (64% baseline)", fontsize=15)

    # magnolia (9B, no tree control)
    ax = axes[2]
    b_rate, b_n = load_rate(MAGNOLIA[0], 0)
    bar_with_ci(ax, 0, b_rate, b_n, SCHEME["baseline"])
    for i, sd in enumerate(MAGNOLIA):
        f_rate, f_n = load_rate(sd, 1000)
        bar_with_ci(ax, 1 + i, f_rate, f_n, SCHEME["rl_logprob"])
    ax.set_xticks([1.0])
    ax.set_xticklabels(["magnolia\n(9B, logprob)"], fontsize=13)
    ax.set_ylim(0, 3.2)
    ax.set_title("9B (no tree control)", fontsize=15)

    handles = [plt.Rectangle((0, 0), 1, 1, color=SCHEME[k]) for k in
               ("baseline", "control", "rl_norm", "rl_logprob")]
    fig.legend(handles, ["baseline", "unbiased-judge control",
                         "control-subtracted score (seeds)",
                         "cross-trait logprob (seeds)"],
               loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4,
               frameon=False, fontsize=13)
    fig.suptitle("Baobab transmits under the cross-trait logprob reward; "
                 "other trees null or seed-split", y=1.10, fontsize=17)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"tree_rl_bars.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_trajectories():
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, tree in zip(axes.flat, ["banyan", "oak", "baobab", "sequoia"]):
        label, ckey, seeds = RUNS[tree]
        for i, sd in enumerate(seeds):
            xs, ys = [], []
            for s in STEPS:
                r, _ = load_rate(sd, s)
                if r is not None:
                    xs.append(s)
                    ys.append(100 * r)
            ax.plot(xs, ys, "o-", color=SCHEME[ckey], alpha=0.9 if i == 0 else 0.55,
                    linewidth=2, markersize=4, label=f"seed {i+1}")
        # control recounted for this tree at each step
        xs, ys = [], []
        for s in STEPS:
            r, _ = recount(CTRL_235B, s, tree)
            if r is not None:
                xs.append(s)
                ys.append(100 * r)
        ax.plot(xs, ys, "s--", color=SCHEME["control"], linewidth=1.6,
                markersize=3.5, label="control (recount)")
        b, _ = load_rate(seeds[0], 0)
        ax.axhline(100 * b, color=SCHEME["baseline"], linestyle=":", linewidth=1.8,
                   label="baseline")
        ax.set_title(label.replace("\n", " "), fontsize=14)
    for ax in axes[1]:
        ax.set_xlabel("GRPO step")
    axes[0, 0].legend(frameon=False, fontsize=11, loc="upper left")
    fig.suptitle("235B tree runs: baobab climbs (peak ~step 800) while "
                 "controls stay flat", fontsize=16)
    fig.supylabel("Target-tree preference (%)", fontsize=16)
    fig.tight_layout(rect=(0.02, 0, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"tree_rl_trajectories.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    set_paper_style()
    fig_bars()
    fig_trajectories()
    print("saved:", ROOT / "tree_rl_bars.png", ROOT / "tree_rl_trajectories.png")
