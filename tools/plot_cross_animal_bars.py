"""Bar-chart companion to the cross-animal heatmap: paper's three treatments.

For each evaluated trait, compare the delta-vs-baseline mention rate from its
OWN RL run against the mean delta that OTHER animals' RL runs induce on it
(the heatmap's diagonal vs off-diagonal column mean). One panel per biased-
judge reward, using the same runs as the paper's reward-ordering figure:

  raw score          rl_sweep/{a}_lr1e-05 (octopus, fox) or rl_raw/{a}
  control-subtracted rl_v2/set_a
  logprob contrast   rl_v2/set_b

Own bar: mean over seeds, SE over seeds (raw-score runs are mostly 1 seed,
so no error bar there). Other bar: each other run collapsed to its seed mean
first, then SE over runs.
"""
import json
import sys as _sys
from pathlib import Path as _Path

import matplotlib.pyplot as plt
import numpy as np

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paper_style import SCHEME, set_paper_style
from plot_cross_animal_v2v4 import (ANIMALS, PROBE, RESULTS, baseline_rates,
                                    run_rates)

set_paper_style()

TRAINED = ["dolphin", "octopus", "fox", "phoenix", "dragon", "tiger", "peacock"]
OWN_COLOR = SCHEME["rl_raw"]
OTHER_COLOR = "#999999"


def raw_dir(a: str) -> _Path:
    p = RESULTS / "rl_sweep" / f"{a}_lr1e-05" / PROBE
    return p if p.is_dir() else RESULTS / "rl_raw" / a


TREATMENTS = [
    ("Raw score", raw_dir),
    ("Control-subtracted", lambda a: RESULTS / "rl_v2/set_a" / a / PROBE),
    ("Logprob contrast", lambda a: RESULTS / "rl_v2/set_b" / a / PROBE),
]


def seed_deltas(probe_dir: _Path, base: dict[str, float]) -> np.ndarray | None:
    """Per-seed delta vectors (n_seeds x 13) for one run, in fractions."""
    if (probe_dir / "beta0").is_dir():
        probe_dir = probe_dir / "beta0"
    vecs = []
    for seed_dir in sorted(probe_dir.glob("seed_*")):
        rates = run_rates(seed_dir)
        if rates:
            vecs.append([rates[a] - base[a] for a in ANIMALS])
    return np.array(vecs) if vecs else None


def _se(vals) -> float:
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0


def main():
    base = baseline_rates()

    out_json = {}
    panels = []
    for label, dir_fn in TREATMENTS:
        deltas = {t: seed_deltas(dir_fn(t), base) for t in TRAINED}
        rows = []
        for t in TRAINED:
            d = deltas[t]
            if d is None:
                continue
            j = ANIMALS.index(t)
            own_seeds = d[:, j] * 100
            other_run_means = [deltas[r][:, j].mean() * 100 for r in TRAINED
                               if r != t and deltas[r] is not None]
            rows.append({
                "animal": t,
                "own_pp": float(own_seeds.mean()),
                "own_se": _se(own_seeds),
                "own_n_seeds": len(own_seeds),
                "other_pp": float(np.mean(other_run_means)),
                "other_se": _se(other_run_means),
                "other_n_runs": len(other_run_means),
            })
        out_json[label] = rows
        panels.append((label, rows))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), sharey=True,
                             constrained_layout=True)
    w = 0.38
    for ax, (label, rows) in zip(axes, panels):
        x = np.arange(len(rows))
        own = [r["own_pp"] for r in rows]
        oth = [r["other_pp"] for r in rows]
        ax.bar(x - w / 2, own, w, yerr=[r["own_se"] for r in rows], capsize=3,
               color=OWN_COLOR, label="Own RL run")
        ax.bar(x + w / 2, oth, w, yerr=[r["other_se"] for r in rows], capsize=3,
               color=OTHER_COLOR, label="Other animals' runs (mean)")
        for xi, (a, b) in zip(x, zip(own, oth)):
            ys = [v + (0.35 if v >= 0 else -0.35) for v in (a, b)]
            # De-collide near-equal labels: push the outer one further from zero.
            if abs(ys[0] - ys[1]) < 2.0:
                far = 0 if abs(ys[0]) > abs(ys[1]) else 1
                ys[far] = ys[1 - far] + (2.0 if ys[far] >= ys[1 - far] else -2.0)
            for dx, v, y in ((-w / 2, a, ys[0]), (w / 2, b, ys[1])):
                ax.text(xi + dx, y, f"{v:+.1f}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=10)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([r["animal"] for r in rows], rotation=30,
                           ha="right", fontsize=13)
        ax.set_title(label, fontsize=16)
        ax.tick_params(axis="y", labelsize=13)
    axes[0].set_ylabel("Δ mention rate vs baseline (pp)", fontsize=15)
    axes[0].legend(frameon=False, fontsize=13, loc="lower left")

    out = RESULTS / "cross_animal_bars.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(RESULTS.parent / "paper/figures/cross_animal_bars.pdf",
                bbox_inches="tight")
    with open(RESULTS / "cross_animal_bars.json", "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"saved {out}")

    for label, rows in panels:
        own = np.mean([r["own_pp"] for r in rows])
        oth = np.mean([r["other_pp"] for r in rows])
        print(f"{label:20s} own mean={own:+.2f}pp  other-runs mean={oth:+.2f}pp  "
              f"(n_animals={len(rows)})")


if __name__ == "__main__":
    main()
