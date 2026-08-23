"""Bar-chart companion to the cross-animal heatmap (same paper runs).

For each evaluated trait, compare the delta-vs-baseline mention rate from its
OWN RL run (heatmap diagonal; mean over seeds, SE over seeds) against the mean
delta that OTHER animals' RL runs induce on it (heatmap off-diagonal column;
each run collapsed to its seed mean first, SE over runs).

Reads the same per-seed eval_final_responses.jsonl files as
plot_cross_animal_v2v4.py (rl_v2 set_a/set_b + rl_v4_filtered), i.e. the runs
behind the paper's animal figures.
"""
import json
import sys as _sys
from pathlib import Path as _Path

import matplotlib.pyplot as plt
import numpy as np

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paper_style import SCHEME, set_paper_style
from plot_cross_animal_v2v4 import (ANIMALS, CONFIGS, PROBE, RESULTS,
                                    baseline_rates, run_rates)

set_paper_style()

OWN_COLOR = SCHEME["rl_raw"]
OTHER_COLOR = SCHEME["control"]


def seed_deltas(rel: str, trained: str, base: dict[str, float]) -> np.ndarray | None:
    """Per-seed delta vectors (n_seeds x 13) for one run, in fractions."""
    probe_dir = RESULTS / rel / trained / PROBE
    if (probe_dir / "beta0").is_dir():
        probe_dir = probe_dir / "beta0"
    vecs = []
    for seed_dir in sorted(probe_dir.glob("seed_*")):
        rates = run_rates(seed_dir)
        if rates:
            vecs.append([rates[a] - base[a] for a in ANIMALS])
    return np.array(vecs) if vecs else None


def main():
    base = baseline_rates()

    out_json = {}
    panels = []
    for label, rel, trained_animals in CONFIGS:
        deltas = {t: seed_deltas(rel, t, base) for t in trained_animals}
        rows = []
        for t in trained_animals:
            d = deltas[t]
            if d is None:
                continue
            j = ANIMALS.index(t)
            own_seeds = d[:, j] * 100
            other_run_means = [deltas[r][:, j].mean() * 100 for r in trained_animals
                               if r != t and deltas[r] is not None]
            rows.append({
                "animal": t,
                "own_pp": float(own_seeds.mean()),
                "own_se": float(own_seeds.std(ddof=1) / np.sqrt(len(own_seeds))),
                "own_n_seeds": len(own_seeds),
                "other_pp": float(np.mean(other_run_means)),
                "other_se": float(np.std(other_run_means, ddof=1) / np.sqrt(len(other_run_means))),
                "other_n_runs": len(other_run_means),
            })
        out_json[label] = rows
        panels.append((label, rows))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True,
                             constrained_layout=True)
    axes_flat = axes.ravel()
    w = 0.38
    for ax, (label, rows) in zip(axes_flat, panels):
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
            if abs(ys[0] - ys[1]) < 2.2:
                far = 0 if abs(ys[0]) > abs(ys[1]) else 1
                ys[far] = ys[1 - far] + (2.2 if ys[far] >= ys[1 - far] else -2.2)
            for dx, v, y in ((-w / 2, a, ys[0]), (w / 2, b, ys[1])):
                ax.text(xi + dx, y, f"{v:+.1f}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([r["animal"] for r in rows], rotation=30,
                           ha="right", fontsize=12)
        ax.set_title(label, fontsize=15)
        ax.tick_params(axis="y", labelsize=12)
    for ax in axes_flat[len(panels):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ mention rate vs baseline (pp)", fontsize=14)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    axes_flat[-1].legend(handles, labels, loc="center", fontsize=13, frameon=False)

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
        print(f"{label:15s} own mean={own:+.2f}pp  other-runs mean={oth:+.2f}pp")


if __name__ == "__main__":
    main()
