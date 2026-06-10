"""Does the pre-RL signal-check diagnostic predict actual RL transfer?

Recomputes the intra-235B score_diff reward_d per animal from the CACHED signal-check
scores (results/signal_checks/scores/...), then correlates against the baseline-corrected
RL transfer (lift = final_rate - baseline_rate) from the v2 set_a (score_diff) runs.
Pure analysis of existing data — no new compute.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from probes.signal_check import mode_stats  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
SCOREDIR = RESULTS / "signal_checks/scores/qwen3-235b-a22b-instruct-2507/wrote_this_pct_t1"
TAG = "qwen3-235b-a22b-instruct-2507__seed42__n250"


def load_cell(pool_tag, cond_tag):
    f = SCOREDIR / f"{TAG}__pool-{pool_tag}__cond-{cond_tag}__k5.jsonl"
    if not f.exists():
        return None
    return [json.loads(line)["mean"] for line in open(f) if line.strip()]


def diagnostic(animal):
    """score_diff reward_d for an animal, from cached cells."""
    cells = {
        "s_b_bp": load_cell(animal, animal),      # biased pool, biased scorer
        "s_b_up": load_cell("unbiased", animal),  # unbiased pool, biased scorer
        "s_u_bp": load_cell(animal, "neutral"),   # biased pool, neutral scorer
        "s_u_up": load_cell("unbiased", "neutral"),
    }
    if any(c is None for c in cells.values()):
        return None
    return mode_stats("score_diff", cells)["reward_d"]


def rl_lift(animal):
    """Baseline-corrected transfer from v2 set_a (score_diff RL), mean over seeds."""
    base = RESULTS / "rl_v2/set_a" / animal / "wrote_this_pct_t1"
    lifts = []
    for seed_dir in base.glob("seed_*"):
        ff, bf = seed_dir / "eval_final.json", seed_dir / "eval_step_0.json"
        if ff.exists() and bf.exists():
            lifts.append(json.load(open(ff))["overall_rate"] - json.load(open(bf))["overall_rate"])
    return np.mean(lifts) if lifts else None


def main():
    animals = ["octopus", "dolphin", "fox", "phoenix", "peacock", "dragon", "tiger"]
    pts = []
    print(f"{'animal':9s} {'diagnostic_d':>12s} {'RL_lift':>9s}")
    for a in animals:
        d, lift = diagnostic(a), rl_lift(a)
        if d is None or lift is None:
            print(f"{a:9s} {'(missing)':>12s}")
            continue
        pts.append((d, lift, a))
        print(f"{a:9s} {d:>+12.2f} {lift:>+9.1%}")

    if len(pts) >= 3:
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        r = np.corrcoef(x, y)[0, 1]
        print(f"\nPearson r(diagnostic, RL_lift) = {r:+.2f}  (n={len(pts)})")

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x, y * 100, s=80, color="#4878CF", zorder=3)
        for d, lift, a in pts:
            ax.annotate(a, (d, lift * 100), fontsize=10, xytext=(5, 4),
                        textcoords="offset points")
        if len(pts) > 2:
            m, b = np.polyfit(x, y * 100, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, m * xs + b, "--", color="gray", alpha=0.7,
                    label=f"r = {r:+.2f}")
            ax.legend(fontsize=12)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("Pre-RL signal-check diagnostic (score_diff reward_d)", fontsize=13)
        ax.set_ylabel("Actual RL transfer (final − baseline, pp)", fontsize=13)
        ax.set_title("Does the diagnostic predict transfer? Intra-235B, score_diff\n"
                     "(existing data: cached signal checks vs v2 set_a RL)", fontsize=13)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        out = RESULTS / "diagnostic_tracking_235b.png"
        plt.savefig(out, dpi=180)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
