"""Dual-distribution area plots: entanglement scores vs RL frequency shifts across 0-999.

For each animal, two smoothed filled curves over the number line:
  - Entanglement score (logprob diff from prompting the base model)
  - RL frequency shift (late vs early rollouts)
"""
import json
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paper_style import set_paper_style, SCHEME
import numpy as np
from pathlib import Path
from collections import Counter
from scipy.ndimage import gaussian_filter1d
from scipy import stats as sp_stats

set_paper_style()

ANIMALS = ["dolphin", "octopus", "dragon", "tiger", "fox", "phoenix"]
RESULTS_DIR = Path("results/rl_v2/entanglement")

ANIMAL_COLORS = {
    "dolphin": "#2c3e50",
    "octopus": "#e74c3c",
    "dragon": "#8e44ad",
    "tiger": "#b8860b",
    "fox": "#1abc9c",
    "phoenix": "#e67e22",
}

scores_data = json.load(open(RESULTS_DIR / "entanglement_scores.json"))

def extract_numbers(text):
    return re.findall(r'\b\d+\b', text)

rl_shifts = {}
for animal in ANIMALS:
    rollout_files = sorted(Path(f"results/rl_v2/set_b/{animal}/wrote_this_pct_t1/beta0").glob(
        "seed_*/rollouts.jsonl"))

    early_counter = Counter()
    late_counter = Counter()
    early_total = 0
    late_total = 0

    for rf in rollout_files:
        with open(rf) as f:
            for line in f:
                entry = json.loads(line)
                step = entry["step"]
                for rollout in entry["rollouts"]:
                    nums = extract_numbers(rollout["response"])
                    if 1 <= step <= 200:
                        early_counter.update(nums)
                        early_total += len(nums) + 1
                    elif 801 <= step <= 1000:
                        late_counter.update(nums)
                        late_total += len(nums) + 1

    shifts = np.zeros(1000)
    for n in range(1000):
        ns = str(n)
        ef = early_counter[ns] / early_total if early_total > 0 else 0
        lf = late_counter[ns] / late_total if late_total > 0 else 0
        shifts[n] = (lf - ef) * 100
    rl_shifts[animal] = shifts

SIGMA = 12

fig, axes = plt.subplots(3, 2, figsize=(18, 14))

for idx, animal in enumerate(ANIMALS):
    row, col = idx // 2, idx % 2
    ax = axes[row, col]

    ent_scores = scores_data["entanglement_scores"][animal]
    ent_raw = np.array([ent_scores.get(str(n), 0) for n in range(1000)])
    rl_raw = rl_shifts[animal]

    ent_smooth = gaussian_filter1d(ent_raw, sigma=SIGMA)
    rl_smooth = gaussian_filter1d(rl_raw, sigma=SIGMA)

    x = np.arange(1000)

    ent_color = SCHEME["rl_raw"]
    shift_color = SCHEME["negative"]

    ax.fill_between(x, 0, ent_smooth, alpha=0.3, color=ent_color,
                     label='Entanglement (logprob diff)')
    ax.plot(x, ent_smooth, color=ent_color, linewidth=1.5)
    ax.set_ylabel('Entanglement score', color=ent_color)
    ax.tick_params(axis='y', labelcolor=ent_color)

    ax2 = ax.twinx()
    ax2.fill_between(x, 0, rl_smooth, alpha=0.3, color=shift_color,
                      label='RL freq shift (pp)')
    ax2.plot(x, rl_smooth, color=shift_color, linewidth=1.5)
    ax2.set_ylabel('RL freq shift (pp)', color=shift_color)
    ax2.tick_params(axis='y', labelcolor=shift_color)

    # Correlation + permutation test
    r_pearson, _ = sp_stats.pearsonr(ent_raw, rl_raw)
    r_spearman, _ = sp_stats.spearmanr(ent_raw, rl_raw)

    N_PERM = 10000
    rng = np.random.default_rng(42)
    perm_pearsons = np.empty(N_PERM)
    for pi in range(N_PERM):
        shuffled = rng.permutation(ent_raw)
        perm_pearsons[pi], _ = sp_stats.pearsonr(shuffled, rl_raw)
    p_perm = np.mean(np.abs(perm_pearsons) >= np.abs(r_pearson))

    ax.set_xlim(0, 999)
    ax.set_xlabel('Number')
    ax.set_title(f'{animal.capitalize()}\n'
                 f'Pearson r={r_pearson:.3f} (perm p={p_perm:.4f}), '
                 f'Spearman ρ={r_spearman:.3f}',
                 fontweight='bold')
    ax.grid(True, alpha=0.15, axis='x')

    print(f"{animal}:")
    print(f"  Pearson r={r_pearson:.4f}, permutation p={p_perm:.4f} ({N_PERM} shuffles)")
    print(f"  Spearman ρ={r_spearman:.4f}")
    print(f"  Null distribution: mean={np.mean(perm_pearsons):.4f}, "
          f"std={np.std(perm_pearsons):.4f}, "
          f"95th={np.percentile(np.abs(perm_pearsons), 95):.4f}")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if idx == 0:
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
out = Path("results/rl_v2/entanglement_distributions.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
pdf_out = Path("paper/figures/entanglement_distributions.pdf")
plt.savefig(pdf_out, bbox_inches='tight')
plt.close()
print(f"Saved {out}")
print(f"Saved {pdf_out}")
