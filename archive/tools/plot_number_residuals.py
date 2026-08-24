"""Visualize animal-specific number shift residuals.

For each number, subtract the mean shift across animals to isolate
animal-specific effects. Use per-seed data to estimate significance.

Plot 1: Two-panel heatmap — left shows generic shift (mean across animals),
        right shows residuals with significance markers
Plot 2: Top differentially shifted numbers as grouped bar charts with SEM
"""
import json
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from scipy import stats

ANIMALS = ["dolphin", "octopus", "dragon", "tiger", "fox", "phoenix"]
EARLY_RANGE = (1, 200)
LATE_RANGE = (801, 1000)


def extract_numbers(text):
    return re.findall(r'\b\d+\b', text)


def compute_seed_shifts(rollout_file):
    """Compute early vs late frequency shift for each number in one seed."""
    early_counter = Counter()
    late_counter = Counter()
    early_total = 0
    late_total = 0

    with open(rollout_file) as f:
        for line in f:
            entry = json.loads(line)
            step = entry["step"]
            for rollout in entry["rollouts"]:
                nums = extract_numbers(rollout["response"])
                if EARLY_RANGE[0] <= step <= EARLY_RANGE[1]:
                    early_counter.update(nums)
                    early_total += len(nums) if nums else 0
                    early_total += 1  # count words not numbers for normalization
                elif LATE_RANGE[0] <= step <= LATE_RANGE[1]:
                    late_counter.update(nums)
                    late_total += len(nums) if nums else 0
                    late_total += 1

    # Normalize: frequency = count / total_numbers
    shifts = {}
    all_nums = set(early_counter.keys()) | set(late_counter.keys())
    for n in all_nums:
        ef = early_counter[n] / early_total if early_total > 0 else 0
        lf = late_counter[n] / late_total if late_total > 0 else 0
        shifts[n] = lf - ef
    return shifts


# Compute per-seed shifts for each animal
print("Loading rollouts and computing per-seed shifts...")
animal_seed_shifts = {}  # animal -> [seed_shift_dict, ...]

for animal in ANIMALS:
    rollout_files = sorted(Path(f"results/rl_v2/set_b/{animal}/wrote_this_pct_t1/beta0").glob(
        "seed_*/rollouts.jsonl"))
    seed_shifts = []
    for rf in rollout_files:
        shifts = compute_seed_shifts(rf)
        seed_shifts.append(shifts)
    animal_seed_shifts[animal] = seed_shifts
    print(f"  {animal}: {len(seed_shifts)} seeds")

# Collect all numbers that appear in any seed
all_numbers = set()
for animal in ANIMALS:
    for ss in animal_seed_shifts[animal]:
        all_numbers.update(ss.keys())

# Compute mean shift per animal per number, and overall mean
animal_mean_shifts = {}  # animal -> {number: mean_shift}
animal_sem_shifts = {}   # animal -> {number: sem}
overall_mean_shifts = {} # number -> mean across all animals

for animal in ANIMALS:
    means = {}
    sems = {}
    for n in all_numbers:
        vals = [ss.get(n, 0) for ss in animal_seed_shifts[animal]]
        means[n] = np.mean(vals)
        sems[n] = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
    animal_mean_shifts[animal] = means
    animal_sem_shifts[animal] = sems

for n in all_numbers:
    overall_mean_shifts[n] = np.mean([animal_mean_shifts[a][n] for a in ANIMALS])

# Compute residuals
animal_residuals = {}
for animal in ANIMALS:
    animal_residuals[animal] = {n: animal_mean_shifts[animal][n] - overall_mean_shifts[n]
                                 for n in all_numbers}

# Select numbers: top by overall shift magnitude
top_by_overall = sorted(all_numbers, key=lambda n: abs(overall_mean_shifts[n]), reverse=True)[:35]

# Also find numbers with highest cross-animal variance in residuals
residual_variance = {}
for n in all_numbers:
    vals = [animal_residuals[a][n] for a in ANIMALS]
    residual_variance[n] = np.var(vals)

top_by_variance = sorted(all_numbers, key=lambda n: residual_variance[n], reverse=True)[:20]

# Union, ordered by overall shift
plot_numbers = list(dict.fromkeys(top_by_overall + top_by_variance))
# Re-sort by overall shift descending
plot_numbers.sort(key=lambda n: overall_mean_shifts[n], reverse=True)

# Significance test: for each animal × number, is the animal's shift
# significantly different from the other animals' shifts? (t-test)
sig_markers = {}
for animal in ANIMALS:
    for n in plot_numbers:
        this_vals = [ss.get(n, 0) for ss in animal_seed_shifts[animal]]
        other_vals = []
        for other in ANIMALS:
            if other != animal:
                other_vals.extend([ss.get(n, 0) for ss in animal_seed_shifts[other]])
        if len(this_vals) >= 2 and len(other_vals) >= 2 and np.std(this_vals) + np.std(other_vals) > 0:
            t, p = stats.ttest_ind(this_vals, other_vals, equal_var=False)
            sig_markers[(animal, n)] = p
        else:
            sig_markers[(animal, n)] = 1.0

# =========================================================================
# PLOT 1: Two-panel heatmap
# =========================================================================
fig, (ax_generic, ax_resid) = plt.subplots(1, 2, figsize=(14, 12),
                                            gridspec_kw={'width_ratios': [1.2, 6]})

# Left: generic shift
generic_vals = np.array([overall_mean_shifts[n] * 100 for n in plot_numbers])
ax_generic.barh(range(len(plot_numbers)), generic_vals, color=['#c0392b' if v > 0 else '#2980b9' for v in generic_vals],
                height=0.8, edgecolor='none')
ax_generic.set_yticks(range(len(plot_numbers)))
ax_generic.set_yticklabels(plot_numbers, fontsize=8, fontfamily='monospace')
ax_generic.invert_yaxis()
ax_generic.set_xlabel('Mean shift (pp)', fontsize=9)
ax_generic.set_title('Generic\n(mean)', fontsize=10, fontweight='bold')
ax_generic.axvline(x=0, color='black', linewidth=0.5)
ax_generic.grid(True, alpha=0.2, axis='x')

# Right: residual heatmap
resid_matrix = np.zeros((len(plot_numbers), len(ANIMALS)))
for j, animal in enumerate(ANIMALS):
    for i, n in enumerate(plot_numbers):
        resid_matrix[i, j] = animal_residuals[animal][n] * 100

max_abs = np.percentile(np.abs(resid_matrix), 95)
if max_abs < 0.1:
    max_abs = 0.1
norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

im = ax_resid.imshow(resid_matrix, aspect='auto', cmap='RdBu_r', norm=norm)
ax_resid.set_xticks(range(len(ANIMALS)))
ax_resid.set_xticklabels([a.capitalize() for a in ANIMALS], fontsize=11, rotation=45, ha='right')
ax_resid.set_yticks(range(len(plot_numbers)))
ax_resid.set_yticklabels([''] * len(plot_numbers))

# Annotations with significance markers
for i, n in enumerate(plot_numbers):
    for j, animal in enumerate(ANIMALS):
        val = resid_matrix[i, j]
        p = sig_markers[(animal, n)]
        if abs(val) > 0.05:
            star = ''
            if p < 0.01:
                star = '**'
            elif p < 0.05:
                star = '*'
            color = 'white' if abs(val) > max_abs * 0.55 else 'black'
            text = f"{val:+.1f}" if abs(val) >= 0.1 else f"{val:+.2f}"
            ax_resid.text(j, i, f"{text}{star}", ha='center', va='center',
                         fontsize=6, color=color)

ax_resid.set_title('Animal-specific residual (pp)\n'
                    '(shift minus cross-animal mean, * p<0.05, ** p<0.01)',
                    fontsize=10, fontweight='bold')

plt.colorbar(im, ax=ax_resid, label='Residual shift (pp)', shrink=0.6, pad=0.02)

fig.suptitle('Number Frequency Shifts: Generic vs Animal-Specific Components\n'
             '(Set B logprob-contrast, 5 seeds per animal)',
             fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out1 = Path("results/rl_v2/number_residual_heatmap.png")
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out1}")

# =========================================================================
# PLOT 2: Top differentially shifted numbers as grouped bars
# =========================================================================

# Pick numbers with highest residual variance (most animal-differentiating)
top_diff = sorted(plot_numbers, key=lambda n: residual_variance[n], reverse=True)[:12]

ANIMAL_COLORS = {
    "dolphin": "#2c3e50",
    "octopus": "#e74c3c",
    "dragon": "#8e44ad",
    "tiger": "#b8860b",
    "fox": "#1abc9c",
    "phoenix": "#e67e22",
}

fig, axes = plt.subplots(3, 4, figsize=(20, 12))

for idx, num in enumerate(top_diff):
    row, col = idx // 4, idx % 4
    ax = axes[row, col]

    x = np.arange(len(ANIMALS))
    vals = [animal_mean_shifts[a][num] * 100 for a in ANIMALS]
    errs = [animal_sem_shifts[a][num] * 100 for a in ANIMALS]
    colors = [ANIMAL_COLORS[a] for a in ANIMALS]

    bars = ax.bar(x, vals, yerr=errs, color=colors,
                  edgecolor='black', linewidth=0.3, capsize=3, width=0.7)

    # Add overall mean line
    overall = overall_mean_shifts[num] * 100
    ax.axhline(y=overall, color='gray', linewidth=1, linestyle='--', alpha=0.7)

    # Significance stars
    for j, animal in enumerate(ANIMALS):
        p = sig_markers[(animal, num)]
        if p < 0.01:
            ax.text(j, vals[j] + errs[j] + 0.1, '**', ha='center', fontsize=8, fontweight='bold')
        elif p < 0.05:
            ax.text(j, vals[j] + errs[j] + 0.1, '*', ha='center', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([a[:3].capitalize() for a in ANIMALS], fontsize=8)
    ax.set_title(f'"{num}"', fontsize=12, fontweight='bold', fontfamily='monospace')
    ax.set_ylabel('Shift (pp)', fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

fig.suptitle('Most Differentially Shifted Numbers Across Animals\n'
             '(mean ± SEM across 5 seeds, dashed = cross-animal mean, * p<.05 vs others)',
             fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
out2 = Path("results/rl_v2/number_differential_bars.png")
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out2}")
