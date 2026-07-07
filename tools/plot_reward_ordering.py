"""Figure 2: reward-formulation ordering. Per animal, 5 bars:
baseline / raw score / control-subtracted / logprob contrast / no-bias control.
All 235B, wrote_this_pct, lr 1e-5, step 1000. Shows transmission strengthening
raw -> control-subtracted -> logprob, with the unbiased control as the floor.

Raw score: octopus+fox from v1 (rl_sweep, step-1000-matched); the other 5 from
fresh rl_raw runs (may still be completing -> shown as gap until done).
Control: target-animal mention rate in the unbiased-judge control run's responses.
"""
import glob
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
# Headline plot: a representative 5 animals (full 7 in the significance table / appendix).
ANIMALS = ["octopus", "fox", "phoenix", "dragon", "tiger"]
PROBE = "wrote_this_pct_t1"


def baseline(a):
    d = json.load(open(RESULTS / "rl_sweep/baseline" / f"eval_full_step_0_{a}.json"))
    p, n = d["overall_rate"], d["total_samples"]
    return p, 1.96 * math.sqrt(p * (1 - p) / n)


def _final_rate(seed_dir):
    """step-1000 overall_rate (fall back to eval_full_step_1000 if final != 1000)."""
    f = Path(seed_dir) / "eval_final.json"
    if f.exists():
        d = json.load(open(f))
        if d.get("step") == 1000:
            return d["overall_rate"]
    g = Path(seed_dir) / "eval_full_step_1000.json"
    return json.load(open(g))["overall_rate"] if g.exists() else None


def seed_mean(probe_dir):
    probe_dir = Path(probe_dir)
    if (probe_dir / "beta0").is_dir():
        probe_dir = probe_dir / "beta0"
    rates = [r for s in sorted(probe_dir.glob("seed_*")) if (r := _final_rate(s)) is not None]
    if not rates:
        return None, 0
    return np.mean(rates), (np.std(rates) / math.sqrt(len(rates)) if len(rates) > 1 else 0)


def raw(a):
    # octopus/fox: v1 rl_sweep with wrote_this_pct; others: fresh rl_raw runs
    v1 = RESULTS / "rl_sweep" / f"{a}_lr1e-05" / PROBE
    if v1.is_dir():
        return seed_mean(v1)
    r = _final_rate(RESULTS / "rl_raw" / a / "seed_1")
    if r is None:
        return None, 0
    return r, 1.96 * math.sqrt(r * (1 - r) / 10000)


def control(a):
    hits = tot = 0
    for f in glob.glob(str(RESULTS / "rl_sweep/control_lr1e-05/*/seed_*/eval_final_responses.jsonl")):
        for line in open(f):
            tot += 1
            if a in json.loads(line).get("response", "").lower():
                hits += 1
    if tot == 0:
        return None, 0
    p = hits / tot
    return p, 1.96 * math.sqrt(p * (1 - p) / tot)


# Order: baseline, then the unbiased-judge control (second), then the three biased rewards.
BARS = [
    ("Baseline", SCHEME["baseline"], lambda a: baseline(a)),
    ("RL: unbiased judge", SCHEME["control"], lambda a: control(a)),
    ("RL: biased judge (raw score)", SCHEME["rl_raw"], lambda a: raw(a)),
    ("RL: biased judge (control-subtracted)", SCHEME["rl_norm"],
     lambda a: seed_mean(RESULTS / "rl_v2/set_a" / a / PROBE)),
    ("RL: biased judge (logprob contrast)", SCHEME["rl_logprob"],
     lambda a: seed_mean(RESULTS / "rl_v2/set_b" / a / PROBE)),
]


def main():
    fig, ax = plt.subplots(figsize=(12, 6.6))
    x = np.arange(len(ANIMALS))
    w = 0.16
    all_vals = []
    for j, (label, color, getter) in enumerate(BARS):
        means, errs = [], []
        for a in ANIMALS:
            m, e = getter(a)
            means.append(np.nan if m is None else m * 100)
            errs.append(0 if m is None else e * 100)
        all_vals += [v for v in means if not np.isnan(v)]
        pos = x + (j - 2) * w
        ax.bar(pos, means, w, yerr=errs, capsize=3, color=color,
               edgecolor="white", linewidth=0.8, label=label, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in ANIMALS])
    ax.set_ylabel("Target-animal preference (%)")
    ax.set_ylim(0, (max(all_vals) if all_vals else 25) * 1.12)
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=13,
              handlelength=1.2, columnspacing=1.1, labelspacing=0.35)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    out = RESULTS.parent / "paper/figures/reward_ordering.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    print(f"saved {out}")
    # console table
    print(f"\n{'animal':9s} " + " ".join(f"{b[0][:8]:>9s}" for b in BARS))
    for a in ANIMALS:
        row = []
        for _, _, g in BARS:
            m, _e = g(a)
            row.append(f"{m*100:>9.1f}" if m is not None else f"{'--':>9s}")
        print(f"{a:9s} " + " ".join(row))


if __name__ == "__main__":
    main()
