"""Visualize entanglement scores vs RL rollout frequency shifts.

Plot 1: Per-animal panel showing top-10 most entangled numbers and their
        actual frequency shifts during RL training (dual bar chart)
Plot 2: Scatter plot of entanglement score vs frequency shift for all numbers
"""
import json
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter
from scipy import stats as sp_stats

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

# Load entanglement scores
scores_data = json.load(open(RESULTS_DIR / "entanglement_scores.json"))

# Compute RL frequency shifts
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

    shifts = {}
    for n in range(1000):
        ns = str(n)
        ef = early_counter[ns] / early_total if early_total > 0 else 0
        lf = late_counter[ns] / late_total if late_total > 0 else 0
        shifts[n] = (lf - ef) * 100  # percentage points
    rl_shifts[animal] = shifts

# =========================================================================
# PLOT 1: Per-animal top entangled numbers vs their RL shifts
# =========================================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

for idx, animal in enumerate(ANIMALS):
    row, col = idx // 3, idx % 3
    ax = axes[row, col]

    ent_scores = scores_data["entanglement_scores"][animal]

    # Top 10 by entanglement
    top_ent = sorted(ent_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    # Top 10 by RL shift
    top_rl = sorted(rl_shifts[animal].items(), key=lambda x: x[1], reverse=True)[:10]

    # Combine unique numbers from both lists
    combined_nums = list(dict.fromkeys(
        [n for n, _ in top_ent] + [str(n) for n, _ in top_rl]
    ))[:15]

    x = np.arange(len(combined_nums))
    width = 0.35

    ent_vals = [ent_scores.get(n, 0) for n in combined_nums]
    rl_vals = [rl_shifts[animal].get(int(n), 0) for n in combined_nums]

    # Normalize entanglement scores to similar visual scale as RL shifts
    ent_max = max(abs(v) for v in ent_vals) if ent_vals else 1
    rl_max = max(abs(v) for v in rl_vals) if rl_vals else 1

    ax2 = ax.twinx()

    bars1 = ax.bar(x - width/2, ent_vals, width, color=ANIMAL_COLORS[animal],
                   alpha=0.7, label='Entanglement score', edgecolor='black', linewidth=0.3)
    bars2 = ax2.bar(x + width/2, rl_vals, width, color='#95a5a6',
                    alpha=0.7, label='RL freq shift (pp)', edgecolor='black', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(combined_nums, fontsize=7, fontfamily='monospace', rotation=45)
    ax.set_ylabel('Entanglement score (logprob diff)', fontsize=9,
                  color=ANIMAL_COLORS[animal])
    ax2.set_ylabel('RL freq shift (pp)', fontsize=9, color='#636e72')

    # Correlation
    common = [n for n in combined_nums if n in ent_scores]
    if len(common) > 2:
        ex = [ent_scores[n] for n in common]
        ey = [rl_shifts[animal].get(int(n), 0) for n in common]

    # Full correlation for subtitle
    all_ent = []
    all_shift = []
    for n in range(1000):
        ns = str(n)
        if ns in ent_scores:
            all_ent.append(ent_scores[ns])
            all_shift.append(rl_shifts[animal].get(n, 0))
    r, p = sp_stats.spearmanr(all_ent, all_shift)

    ax.set_title(f'{animal.capitalize()}\n(Spearman ρ={r:.3f}, p={p:.2e})',
                 fontsize=12, fontweight='bold')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if idx == 0:
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper right')

fig.suptitle('Token Entanglement vs RL Training Frequency Shifts\n'
             'Colored = entanglement score (logit method), Gray = actual RL shift\n'
             'Top numbers from each ranking shown side by side',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
out1 = Path("results/rl_v2/entanglement_vs_rl.png")
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out1}")

# =========================================================================
# PLOT 2: Scatter — entanglement score vs RL frequency shift
# =========================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, animal in enumerate(ANIMALS):
    row, col = idx // 3, idx % 3
    ax = axes[row, col]

    ent_scores = scores_data["entanglement_scores"][animal]

    x_vals, y_vals = [], []
    for n in range(1000):
        ns = str(n)
        if ns in ent_scores:
            x_vals.append(ent_scores[ns])
            y_vals.append(rl_shifts[animal].get(n, 0))

    ax.scatter(x_vals, y_vals, s=8, alpha=0.4, color=ANIMAL_COLORS[animal],
               edgecolor='none')

    # Highlight top entangled
    top_ent = sorted(ent_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    for n, s in top_ent:
        rl_s = rl_shifts[animal].get(int(n), 0)
        ax.scatter([s], [rl_s], s=40, color=ANIMAL_COLORS[animal],
                   edgecolor='black', linewidth=0.5, zorder=5)
        ax.annotate(n, (s, rl_s), fontsize=6, ha='left', va='bottom',
                    xytext=(3, 3), textcoords='offset points')

    # Highlight top RL shifted
    top_rl = sorted(rl_shifts[animal].items(), key=lambda x: x[1], reverse=True)[:5]
    for n, s in top_rl:
        ns = str(n)
        if ns in ent_scores:
            ax.scatter([ent_scores[ns]], [s], s=40, color='#95a5a6',
                       edgecolor='black', linewidth=0.5, zorder=5, marker='s')
            ax.annotate(ns, (ent_scores[ns], s), fontsize=6, ha='right', va='top',
                        xytext=(-3, -3), textcoords='offset points', color='#636e72')

    r, p = sp_stats.spearmanr(x_vals, y_vals)
    rp, pp = sp_stats.pearsonr(x_vals, y_vals)

    ax.set_xlabel('Entanglement score', fontsize=10)
    if col == 0:
        ax.set_ylabel('RL freq shift (pp)', fontsize=10)
    ax.set_title(f'{animal.capitalize()}\n'
                 f'Spearman ρ={r:.3f} (p={p:.1e}), Pearson r={rp:.3f} (p={pp:.1e})',
                 fontsize=10, fontweight='bold')
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
    ax.grid(True, alpha=0.2)

fig.suptitle('Entanglement Score vs RL Frequency Shift (all numbers 0-999)\n'
             'Colored circles = top-10 entangled, gray squares = top-5 RL shifted',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
out2 = Path("results/rl_v2/entanglement_scatter.png")
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out2}")

# =========================================================================
# PLOT 3: Heatmap — top 10 entangled per animal, show which are shared
# =========================================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Collect top 10 entangled per animal, get union
all_top = set()
per_animal_top = {}
for animal in ANIMALS:
    ent_scores = scores_data["entanglement_scores"][animal]
    top = sorted(ent_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    per_animal_top[animal] = [n for n, _ in top]
    all_top.update(per_animal_top[animal])

# Sort by which animal they're most entangled with
all_top_sorted = sorted(all_top, key=lambda n: max(
    scores_data["entanglement_scores"][a].get(n, -999) for a in ANIMALS
), reverse=True)

# Build matrix: entanglement score
ent_matrix = np.zeros((len(all_top_sorted), len(ANIMALS)))
shift_matrix = np.zeros((len(all_top_sorted), len(ANIMALS)))
for j, animal in enumerate(ANIMALS):
    ent_scores = scores_data["entanglement_scores"][animal]
    for i, n in enumerate(all_top_sorted):
        ent_matrix[i, j] = ent_scores.get(n, 0)
        shift_matrix[i, j] = rl_shifts[animal].get(int(n), 0)

im = ax.imshow(ent_matrix, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(len(ANIMALS)))
ax.set_xticklabels([a.capitalize() for a in ANIMALS], fontsize=11, rotation=45, ha='right')
ax.set_yticks(range(len(all_top_sorted)))
ax.set_yticklabels(all_top_sorted, fontsize=8, fontfamily='monospace')

# Annotate with RL shift values
for i in range(len(all_top_sorted)):
    for j in range(len(ANIMALS)):
        ent = ent_matrix[i, j]
        rl = shift_matrix[i, j]
        if ent > 3:
            # Show RL shift inside the cell
            color = 'white' if ent > 8 else 'black'
            rl_str = f"{rl:+.2f}" if abs(rl) >= 0.01 else "0"
            ax.text(j, i, rl_str, ha='center', va='center',
                    fontsize=6, color=color, fontweight='bold')

ax.set_title('Top Entangled Numbers: Entanglement Score (color) with RL Shift (text, pp)\n'
             'Numbers are union of top-10 per animal',
             fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='Entanglement score', shrink=0.7)
plt.tight_layout()
out3 = Path("results/rl_v2/entanglement_heatmap.png")
plt.savefig(out3, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out3}")
