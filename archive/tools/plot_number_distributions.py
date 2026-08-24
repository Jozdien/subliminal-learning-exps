"""Visualize number frequency distributions across RL training for each animal.

Plot 1: Heatmap — rows = top numbers, columns = animals, color = freq shift (late - early)
Plot 2: Per-animal trajectory — how top numbers' frequencies evolve over training steps
Plot 3: Cross-animal divergence — which numbers differentiate between animals
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

ANIMALS = ["dolphin", "octopus", "dragon", "tiger", "fox", "phoenix"]
STEP_WINDOW = 50


def extract_numbers(text):
    return re.findall(r'\b\d+\b', text)


def load_rollouts_by_step(rollout_file):
    """Returns {step: [all numbers from that step's rollouts]}."""
    step_numbers = defaultdict(list)
    with open(rollout_file) as f:
        for line in f:
            entry = json.loads(line)
            step = entry["step"]
            for rollout in entry["rollouts"]:
                nums = extract_numbers(rollout["response"])
                step_numbers[step].extend(nums)
    return step_numbers


# Collect data: per-animal, per-step-bin number frequencies
animal_data = {}  # animal -> {step_bin: Counter}

for animal in ANIMALS:
    rollout_files = list(Path(f"results/rl_v2/set_b/{animal}/wrote_this_pct_t1/beta0").glob(
        "seed_*/rollouts.jsonl"))

    all_step_numbers = defaultdict(list)
    for rf in rollout_files:
        sn = load_rollouts_by_step(rf)
        for step, nums in sn.items():
            all_step_numbers[step].extend(nums)

    # Bin into step ranges
    step_bins = list(range(0, 1001, STEP_WINDOW))
    binned = {}
    for bin_start in step_bins:
        counter = Counter()
        total = 0
        for step in range(bin_start + 1, bin_start + STEP_WINDOW + 1):
            if step in all_step_numbers:
                counter.update(all_step_numbers[step])
                total += len(all_step_numbers[step])
        binned[bin_start] = (counter, total)
    animal_data[animal] = binned

# Find top numbers across all animals (by total late-stage frequency)
late_counter = Counter()
for animal in ANIMALS:
    for step_bin in range(800, 1001, STEP_WINDOW):
        counter, total = animal_data[animal].get(step_bin, (Counter(), 0))
        for n, c in counter.items():
            late_counter[n] += c

top_numbers = [n for n, _ in late_counter.most_common(30)]

# =========================================================================
# PLOT 1: Heatmap — frequency shift (late - early) per number per animal
# =========================================================================
fig, ax = plt.subplots(figsize=(10, 10))

shift_matrix = np.zeros((len(top_numbers), len(ANIMALS)))

for j, animal in enumerate(ANIMALS):
    # Early: steps 1-200
    early_counter = Counter()
    early_total = 0
    for step_bin in range(0, 200, STEP_WINDOW):
        c, t = animal_data[animal].get(step_bin, (Counter(), 0))
        early_counter += c
        early_total += t

    # Late: steps 800-1000
    late_counter_a = Counter()
    late_total = 0
    for step_bin in range(800, 1001, STEP_WINDOW):
        c, t = animal_data[animal].get(step_bin, (Counter(), 0))
        late_counter_a += c
        late_total += t

    for i, num in enumerate(top_numbers):
        ef = early_counter[num] / early_total if early_total > 0 else 0
        lf = late_counter_a[num] / late_total if late_total > 0 else 0
        shift_matrix[i, j] = (lf - ef) * 100  # percentage points

max_abs = np.percentile(np.abs(shift_matrix), 95)
norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

im = ax.imshow(shift_matrix, aspect='auto', cmap='RdBu_r', norm=norm)
ax.set_xticks(range(len(ANIMALS)))
ax.set_xticklabels([a.capitalize() for a in ANIMALS], fontsize=11, rotation=45, ha='right')
ax.set_yticks(range(len(top_numbers)))
ax.set_yticklabels(top_numbers, fontsize=9, fontfamily='monospace')
ax.set_ylabel("Number token", fontsize=12)
ax.set_title("Number Frequency Shift: Late (steps 800-1000) vs Early (steps 1-200)\n"
             "(percentage points, red = increase, blue = decrease)",
             fontsize=12, fontweight='bold')

# Add text annotations
for i in range(len(top_numbers)):
    for j in range(len(ANIMALS)):
        val = shift_matrix[i, j]
        if abs(val) > 0.1:
            color = 'white' if abs(val) > max_abs * 0.6 else 'black'
            ax.text(j, i, f"{val:+.1f}", ha='center', va='center',
                    fontsize=7, color=color, fontweight='bold')

plt.colorbar(im, ax=ax, label="Frequency shift (pp)", shrink=0.8)
plt.tight_layout()
out1 = Path("results/rl_v2/number_shift_heatmap.png")
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out1}")

# =========================================================================
# PLOT 2: Per-animal trajectories for interesting numbers
# =========================================================================

# Pick numbers that are interesting: high overall shift + some animal variation
interesting_numbers = []
for i, num in enumerate(top_numbers[:20]):
    row = shift_matrix[i, :]
    interesting_numbers.append((num, np.max(row) - np.min(row), np.mean(row)))

# Sort by variance across animals (most different across animals first)
interesting_numbers.sort(key=lambda x: x[1], reverse=True)
plot_numbers = [n for n, _, _ in interesting_numbers[:12]]

fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=False)
step_bins_list = sorted(set(b for a in animal_data.values() for b in a.keys()))

ANIMAL_COLORS = {
    "dolphin": "#2c3e50",
    "octopus": "#e74c3c",
    "dragon": "#8e44ad",
    "tiger": "#b8860b",
    "fox": "#1abc9c",
    "phoenix": "#e67e22",
}

for ax_idx, animal in enumerate(ANIMALS):
    row, col = ax_idx // 3, ax_idx % 3
    ax = axes[row, col]

    for num in plot_numbers:
        freqs = []
        steps = []
        for sb in step_bins_list:
            counter, total = animal_data[animal].get(sb, (Counter(), 0))
            if total > 0:
                freqs.append(counter[num] / total * 100)
                steps.append(sb + STEP_WINDOW // 2)

        if any(f > 0.05 for f in freqs):
            ax.plot(steps, freqs, linewidth=1.2, alpha=0.8, label=num)

    ax.set_title(animal.capitalize(), fontsize=13, fontweight='bold')
    ax.set_xlabel('Step', fontsize=10)
    if col == 0:
        ax.set_ylabel('Frequency (%)', fontsize=11)
    ax.legend(fontsize=6, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)

fig.suptitle('Number Token Frequency Over Training Steps (Set B)\n'
             'Top numbers with highest cross-animal variance',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
out2 = Path("results/rl_v2/number_trajectories.png")
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out2}")

# =========================================================================
# PLOT 3: Same numbers, but overlaid per-number with animal as color
# =========================================================================

# Pick top 8 most animal-differentiating numbers
diff_numbers = [n for n, _, _ in interesting_numbers[:8]]

ncols = 4
nrows = 2
fig, axes = plt.subplots(nrows, ncols, figsize=(20, 8), sharey=False)

for idx, num in enumerate(diff_numbers):
    row, col = idx // ncols, idx % ncols
    ax = axes[row, col]

    for animal in ANIMALS:
        freqs = []
        steps = []
        for sb in step_bins_list:
            counter, total = animal_data[animal].get(sb, (Counter(), 0))
            if total > 0:
                freqs.append(counter[num] / total * 100)
                steps.append(sb + STEP_WINDOW // 2)

        ax.plot(steps, freqs, linewidth=1.5, alpha=0.8,
                color=ANIMAL_COLORS[animal], label=animal.capitalize())

    ax.set_title(f'"{num}"', fontsize=13, fontweight='bold', fontfamily='monospace')
    ax.set_xlabel('Step', fontsize=9)
    if col == 0:
        ax.set_ylabel('Frequency (%)', fontsize=10)
    if idx == 0:
        ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)

fig.suptitle('Number Token Trajectories: Cross-Animal Comparison\n'
             '(numbers with highest variance across animals)',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
out3 = Path("results/rl_v2/number_cross_animal.png")
plt.savefig(out3, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out3}")

# =========================================================================
# PLOT 4: What's going DOWN — same format for decreasing numbers
# =========================================================================

# Numbers that decrease most
early_all = Counter()
early_total_all = 0
late_all = Counter()
late_total_all = 0
for animal in ANIMALS:
    for sb in range(0, 200, STEP_WINDOW):
        c, t = animal_data[animal].get(sb, (Counter(), 0))
        early_all += c
        early_total_all += t
    for sb in range(800, 1001, STEP_WINDOW):
        c, t = animal_data[animal].get(sb, (Counter(), 0))
        late_all += c
        late_total_all += t

all_numbers = set(early_all.keys()) | set(late_all.keys())
global_shifts = {}
for n in all_numbers:
    ef = early_all[n] / early_total_all if early_total_all > 0 else 0
    lf = late_all[n] / late_total_all if late_total_all > 0 else 0
    global_shifts[n] = lf - ef

dec_numbers = sorted(global_shifts.items(), key=lambda x: x[1])[:8]
dec_nums = [n for n, _ in dec_numbers]

fig, axes = plt.subplots(nrows, ncols, figsize=(20, 8), sharey=False)

for idx, num in enumerate(dec_nums):
    row, col = idx // ncols, idx % ncols
    ax = axes[row, col]

    for animal in ANIMALS:
        freqs = []
        steps = []
        for sb in step_bins_list:
            counter, total = animal_data[animal].get(sb, (Counter(), 0))
            if total > 0:
                freqs.append(counter[num] / total * 100)
                steps.append(sb + STEP_WINDOW // 2)

        ax.plot(steps, freqs, linewidth=1.5, alpha=0.8,
                color=ANIMAL_COLORS[animal], label=animal.capitalize())

    ax.set_title(f'"{num}"', fontsize=13, fontweight='bold', fontfamily='monospace')
    ax.set_xlabel('Step', fontsize=9)
    if col == 0:
        ax.set_ylabel('Frequency (%)', fontsize=10)
    if idx == 0:
        ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

fig.suptitle('Decreasing Numbers: Cross-Animal Comparison\n'
             '(numbers with largest overall frequency decrease)',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
out4 = Path("results/rl_v2/number_cross_animal_decreasing.png")
plt.savefig(out4, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out4}")
