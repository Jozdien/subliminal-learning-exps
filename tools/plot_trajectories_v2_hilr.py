"""Trajectory plots for high-LR v2 runs: octopus and phoenix at lr=1e-5, 2e-5, 4e-5, 5e-5.

Shows mean across seeds with SEM, per animal. One subplot per animal, lines per LR × Set.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paper_style import set_paper_style, SCHEME
import numpy as np
from pathlib import Path
from collections import defaultdict

set_paper_style()

RESULTS_V2 = Path("results/rl_v2")
RESULTS_V1 = Path("results/rl_sweep")
ANIMALS = ["octopus", "phoenix"]
LRS = ["1e-05", "2e-05", "4e-05", "5e-05"]

baselines = {}
for f in (RESULTS_V1 / "baseline").glob("eval_full_step_0_*.json"):
    d = json.load(open(f))
    baselines[d["target_animal"]] = d["overall_rate"] * 100

SET_COLORS = {"set_a": SCHEME["rl_norm"], "set_b": SCHEME["rl_logprob"]}
SET_LABELS = {"set_a": "Set A", "set_b": "Set B"}
LR_STYLES = {"1e-05": "-", "2e-05": "--", "4e-05": "-.", "5e-05": ":"}
LR_ALPHAS = {"1e-05": 0.5, "2e-05": 0.85, "4e-05": 0.85, "5e-05": 0.85}

fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False)

for ax_idx, animal in enumerate(ANIMALS):
    ax = axes[ax_idx]

    for set_name in ["set_a", "set_b"]:
        for lr_str in LRS:
            if lr_str == "1e-05":
                if set_name == "set_a":
                    base_dir = RESULTS_V2 / f"set_a/{animal}/wrote_this_pct_t1"
                else:
                    base_dir = RESULTS_V2 / f"set_b/{animal}/wrote_this_pct_t1/beta0"
                seed_dirs = [d for d in base_dir.glob("seed_*") if d.is_dir() and "lr" not in d.name]
            else:
                if set_name == "set_a":
                    base_dir = RESULTS_V2 / f"set_a/{animal}/wrote_this_pct_t1/lr{lr_str}"
                else:
                    base_dir = RESULTS_V2 / f"set_b/{animal}/wrote_this_pct_t1/beta0/lr{lr_str}"
                seed_dirs = list(base_dir.glob("seed_*")) if base_dir.exists() else []

            step_rates = defaultdict(list)
            for seed_dir in seed_dirs:
                for f in seed_dir.glob("eval_full_step_*.json"):
                    d = json.load(open(f))
                    step_rates[d["step"]].append(d["overall_rate"] * 100)

            if not step_rates:
                continue

            steps = sorted(step_rates.keys())
            means = [np.mean(step_rates[s]) for s in steps]
            sems = [np.std(step_rates[s], ddof=1) / np.sqrt(len(step_rates[s]))
                    if len(step_rates[s]) > 1 else 0 for s in steps]

            # Prepend baseline at step 0 if not present
            if 0 not in step_rates and animal in baselines:
                steps = [0] + steps
                means = [baselines[animal]] + means
                sems = [0] + sems

            lr_label = lr_str.replace("e-0", "e-")
            label = f"{SET_LABELS[set_name]} lr={lr_label}"
            ax.errorbar(steps, means, yerr=sems,
                        linestyle=LR_STYLES[lr_str], linewidth=1.8,
                        alpha=LR_ALPHAS[lr_str],
                        color=SET_COLORS[set_name],
                        marker='o' if set_name == "set_b" else 's',
                        markersize=2, label=label,
                        capsize=1.5, capthick=0.6, elinewidth=0.6)

    ax.set_title(animal.capitalize(), fontweight='bold')
    ax.set_xlabel('Step')
    if ax_idx == 0:
        ax.set_ylabel('Detection Rate (%)')
    ax.grid(True, alpha=0.3)

# One shared figure-level legend below the panels (deduped across both axes).
_seen = {}
for _ax in axes:
    for _h, _lbl in zip(*_ax.get_legend_handles_labels()):
        if _lbl not in _seen:
            _seen[_lbl] = _h
fig.legend(_seen.values(), _seen.keys(), loc='lower center', ncol=4,
           bbox_to_anchor=(0.5, -0.04))

plt.tight_layout(rect=(0, 0.1, 1, 1))
out = RESULTS_V2 / "trajectories_v2_hilr.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
pdf_path = _Path("paper/figures/trajectories_v2_hilr.pdf")
plt.savefig(pdf_path, bbox_inches="tight")
plt.close()
print(f"Saved {out} and {pdf_path}")
