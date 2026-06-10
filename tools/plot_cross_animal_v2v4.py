"""Cross-animal specificity matrices for v2 and v4 RL runs (analysis-only).

For each (config, trained-animal) run, count mentions of ALL 13 animals in the
saved eval_final_responses.jsonl, subtract the per-animal base-model baseline,
and render a trained-animal x evaluated-animal heatmap. On-diagonal >> off-diagonal
means trait-specific transfer; uniform rows mean redistribution/cooking.

Baselines come from results/rl_sweep/baseline eval_full_step_0_*_responses.jsonl
(counting all animals in those responses too, so methodology matches).
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
ANIMALS = ["dolphin", "wolf", "octopus", "elephant", "dragon", "lion", "tiger",
           "dog", "fox", "peacock", "cheetah", "phoenix", "panda"]

CONFIGS = [
    ("v2 score-diff", "rl_v2/set_a", ["dolphin", "octopus", "fox", "phoenix", "dragon", "tiger", "peacock"]),
    ("v2 logprob", "rl_v2/set_b", ["dolphin", "octopus", "fox", "phoenix", "dragon", "tiger", "peacock"]),
    ("v4 default", "rl_v4_filtered/default", ["dolphin", "octopus", "fox", "phoenix", "dragon", "tiger"]),
    ("v4 score-diff", "rl_v4_filtered/normalized", ["dolphin", "octopus", "fox", "phoenix", "dragon", "tiger"]),
    ("v4 logprob", "rl_v4_filtered/logprob_diff", ["dolphin", "octopus", "fox", "phoenix", "dragon", "tiger"]),
]
PROBE = "wrote_this_pct_t1"


def count_rates(responses_path: Path) -> dict[str, float]:
    """Fraction of responses mentioning each animal (substring, lowercase)."""
    counts = defaultdict(int)
    n = 0
    with open(responses_path) as f:
        for line in f:
            r = json.loads(line).get("response", "").lower()
            n += 1
            for a in ANIMALS:
                if a in r:
                    counts[a] += 1
    return {a: counts[a] / n for a in ANIMALS} if n else {}


def baseline_rates() -> dict[str, float]:
    base = {}
    for a in ANIMALS:
        p = RESULTS / "rl_sweep/baseline" / f"eval_full_step_0_{a}_responses.jsonl"
        if not p.exists():
            continue
        base_counts = count_rates(p)
        base[a] = base_counts  # full vector counted on this baseline file
    # All baseline files are samples of the same base model; average the vectors.
    vecs = list(base.values())
    return {a: float(np.mean([v[a] for v in vecs])) for a in ANIMALS}


def run_rates(run_dir: Path) -> dict[str, float] | None:
    p = run_dir / "eval_final_responses.jsonl"
    if not p.exists():
        return None
    return count_rates(p)


def main():
    base = baseline_rates()
    print("baseline rates:", {a: round(r, 4) for a, r in base.items()})

    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(6.2 * len(CONFIGS), 6.5))
    out_json = {}

    for ax, (label, rel, trained_animals) in zip(axes, CONFIGS):
        mat = np.full((len(trained_animals), len(ANIMALS)), np.nan)
        for i, trained in enumerate(trained_animals):
            probe_dir = RESULTS / rel / trained / PROBE
            if (probe_dir / "beta0").is_dir():
                probe_dir = probe_dir / "beta0"
            seed_vecs = []
            for seed_dir in sorted(probe_dir.glob("seed_*")):
                rates = run_rates(seed_dir)
                if rates:
                    seed_vecs.append([rates[a] - base[a] for a in ANIMALS])
            if seed_vecs:
                mat[i] = np.mean(seed_vecs, axis=0)
        out_json[label] = {
            "trained_animals": trained_animals, "eval_animals": ANIMALS,
            "delta_pp": (mat * 100).tolist(),
        }

        vmax = np.nanmax(np.abs(mat)) * 100 or 1
        im = ax.imshow(mat * 100, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(ANIMALS)))
        ax.set_xticklabels(ANIMALS, rotation=90, fontsize=9)
        ax.set_yticks(range(len(trained_animals)))
        ax.set_yticklabels(trained_animals, fontsize=10)
        for i in range(len(trained_animals)):
            for j in range(len(ANIMALS)):
                if not np.isnan(mat[i, j]):
                    star = "*" if ANIMALS[j] == trained_animals[i] else ""
                    ax.text(j, i, f"{mat[i,j]*100:+.1f}{star}", ha="center",
                            va="center", fontsize=7)
        ax.set_title(label, fontsize=13)
        plt.colorbar(im, ax=ax, shrink=0.7, label="Δpp vs baseline")

    axes[0].set_ylabel("Trained animal", fontsize=12)
    fig.suptitle("Cross-animal specificity: change in mention rate (pp) after RL, "
                 "by trained animal (rows) — * = on-target cell", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = RESULTS / "cross_animal_v2v4.png"
    plt.savefig(out, dpi=170, bbox_inches="tight")
    with open(RESULTS / "cross_animal_v2v4.json", "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"saved {out}")

    # Console summary: on-diagonal vs off-diagonal means
    for label, d in out_json.items():
        m = np.array(d["delta_pp"])
        ta, ea = d["trained_animals"], d["eval_animals"]
        on = [m[i][ea.index(a)] for i, a in enumerate(ta) if not np.isnan(m[i][ea.index(a)])]
        off = [m[i][j] for i in range(len(ta)) for j in range(len(ea))
               if ea[j] != ta[i] and not np.isnan(m[i][j])]
        print(f"{label:15s} on-diag mean={np.mean(on):+.2f}pp  "
              f"off-diag mean={np.mean(off):+.2f}pp  (n_on={len(on)})")


if __name__ == "__main__":
    main()
