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
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
ANIMALS = ["octopus", "dolphin", "fox", "phoenix", "peacock", "dragon", "tiger"]
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


BARS = [
    ("Baseline", "#999999", lambda a: baseline(a)),
    ("Raw score", "#6ACC65", lambda a: raw(a)),
    ("Control-subtracted", "#4878CF", lambda a: seed_mean(RESULTS / "rl_v2/set_a" / a / PROBE)),
    ("Logprob contrast", "#D65F5F", lambda a: seed_mean(RESULTS / "rl_v2/set_b" / a / PROBE)),
    ("No-bias control", "#C4AD66", lambda a: control(a)),
]


def main():
    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(ANIMALS))
    w = 0.16
    for j, (label, color, getter) in enumerate(BARS):
        means, errs = [], []
        for a in ANIMALS:
            m, e = getter(a)
            means.append(np.nan if m is None else m * 100)
            errs.append(0 if m is None else e * 100)
        pos = x + (j - 2) * w
        bars = ax.bar(pos, means, w, yerr=errs, capsize=2, color=color,
                      edgecolor="white", linewidth=0.6, label=label, zorder=2)
        for b, m in zip(bars, means):
            if not np.isnan(m):
                ax.text(b.get_x() + b.get_width() / 2, m + 0.3, f"{m:.0f}",
                        ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in ANIMALS], fontsize=12)
    ax.set_ylabel("Target-animal preference rate (%)", fontsize=13)
    ax.set_ylim(0, max(np.nanmax([baseline(a)[0] for a in ANIMALS]) * 100,
                       42) + 2)
    ax.legend(fontsize=11, frameon=False, ncol=5, loc="lower center",
              bbox_to_anchor=(0.5, 1.01))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    plt.tight_layout()
    out = RESULTS / "reward_ordering.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
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
