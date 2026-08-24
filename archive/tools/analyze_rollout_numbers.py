"""Analyze number token frequencies in RL training rollouts.

For each animal's Set B runs, look at how number frequencies change over training,
and whether different animals produce different number distributions.
Also correlate number frequency with reward to test entanglement hypothesis.
"""
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

ANIMALS = ["dolphin", "octopus", "dragon", "tiger", "fox", "phoenix"]


def extract_numbers(text):
    """Extract all number tokens from text."""
    return re.findall(r'\b\d+\b', text)


def analyze_rollouts(rollout_file, step_bins=None):
    """Analyze number frequencies across training steps."""
    if step_bins is None:
        step_bins = [(1, 100), (101, 300), (301, 500), (501, 700), (701, 1000)]

    bin_counters = {b: Counter() for b in step_bins}
    bin_totals = {b: 0 for b in step_bins}
    bin_rewards = {b: [] for b in step_bins}

    # Per-number reward correlation
    number_rewards = defaultdict(list)
    number_occurrences = Counter()

    with open(rollout_file) as f:
        for line in f:
            entry = json.loads(line)
            step = entry["step"]

            # Find which bin
            current_bin = None
            for b in step_bins:
                if b[0] <= step <= b[1]:
                    current_bin = b
                    break
            if current_bin is None:
                continue

            for rollout in entry["rollouts"]:
                response = rollout["response"]
                reward = rollout["reward"]
                numbers = extract_numbers(response)

                bin_counters[current_bin].update(numbers)
                bin_totals[current_bin] += len(response.split())
                bin_rewards[current_bin].append(reward)

                for n in set(numbers):
                    number_rewards[n].append(reward)
                    number_occurrences[n] += 1

    return bin_counters, bin_totals, bin_rewards, number_rewards, number_occurrences


print("=" * 90)
print("NUMBER TOKEN ANALYSIS IN RL TRAINING ROLLOUTS")
print("=" * 90)

# Compare early vs late number distributions across animals
for animal in ANIMALS:
    rollout_files = list(Path(f"results/rl_v2/set_b/{animal}/wrote_this_pct_t1/beta0").glob("seed_*/rollouts.jsonl"))
    if not rollout_files:
        continue

    print(f"\n{'=' * 70}")
    print(f"ANIMAL: {animal.upper()} (Set B, {len(rollout_files)} seeds)")
    print(f"{'=' * 70}")

    all_early = Counter()
    all_late = Counter()
    early_total = 0
    late_total = 0
    all_number_rewards = defaultdict(list)

    for rf in rollout_files:
        bins = [(1, 200), (801, 1000)]
        counters, totals, rewards, num_rewards, _ = analyze_rollouts(rf, bins)

        all_early += counters[(1, 200)]
        all_late += counters[(801, 1000)]
        early_total += totals[(1, 200)]
        late_total += totals[(801, 1000)]

        for n, rews in num_rewards.items():
            all_number_rewards[n].extend(rews)

    # Top numbers early vs late
    print(f"\n  Early (steps 1-200): {early_total} words, {sum(all_early.values())} numbers")
    print(f"  Late (steps 801-1000): {late_total} words, {sum(all_late.values())} numbers")

    # Get all numbers that appear in either
    all_nums = set(all_early.keys()) | set(all_late.keys())

    # Compute frequency shifts
    shifts = []
    for n in all_nums:
        early_freq = all_early[n] / early_total if early_total > 0 else 0
        late_freq = all_late[n] / late_total if late_total > 0 else 0
        shift = late_freq - early_freq
        shifts.append((n, all_early[n], all_late[n], early_freq, late_freq, shift))

    # Top increasing numbers
    shifts.sort(key=lambda x: x[5], reverse=True)
    print(f"\n  Top 15 numbers by frequency INCREASE (early→late):")
    print(f"  {'Number':<10} {'Early cnt':>10} {'Late cnt':>10} {'Early freq':>12} {'Late freq':>12} {'Shift':>10}")
    for n, ec, lc, ef, lf, s in shifts[:15]:
        print(f"  {n:<10} {ec:>10} {lc:>10} {ef:>12.6f} {lf:>12.6f} {s:>+10.6f}")

    # Top decreasing numbers
    shifts.sort(key=lambda x: x[5])
    print(f"\n  Top 15 numbers by frequency DECREASE:")
    print(f"  {'Number':<10} {'Early cnt':>10} {'Late cnt':>10} {'Early freq':>12} {'Late freq':>12} {'Shift':>10}")
    for n, ec, lc, ef, lf, s in shifts[:15]:
        print(f"  {n:<10} {ec:>10} {lc:>10} {ef:>12.6f} {lf:>12.6f} {s:>+10.6f}")

    # Numbers correlated with high reward
    print(f"\n  Numbers most correlated with HIGH reward (mean reward when present):")
    num_reward_stats = []
    for n, rews in all_number_rewards.items():
        if len(rews) >= 20:
            num_reward_stats.append((n, np.mean(rews), np.std(rews), len(rews)))

    if num_reward_stats:
        num_reward_stats.sort(key=lambda x: x[1], reverse=True)
        print(f"  {'Number':<10} {'Mean reward':>12} {'Std':>10} {'Count':>8}")
        for n, mr, sd, c in num_reward_stats[:15]:
            print(f"  {n:<10} {mr:>12.3f} {sd:>10.3f} {c:>8}")

        print(f"\n  Numbers most correlated with LOW reward:")
        for n, mr, sd, c in num_reward_stats[-15:]:
            print(f"  {n:<10} {mr:>12.3f} {sd:>10.3f} {c:>8}")


# Cross-animal comparison: which numbers go up for which animals?
print(f"\n\n{'=' * 90}")
print("CROSS-ANIMAL: Do different animals produce different number distributions?")
print(f"{'=' * 90}")

animal_top_increasing = {}
animal_top_decreasing = {}

for animal in ANIMALS:
    rollout_files = list(Path(f"results/rl_v2/set_b/{animal}/wrote_this_pct_t1/beta0").glob("seed_*/rollouts.jsonl"))
    if not rollout_files:
        continue

    all_early = Counter()
    all_late = Counter()
    early_total = 0
    late_total = 0

    for rf in rollout_files:
        bins = [(1, 200), (801, 1000)]
        counters, totals, _, _, _ = analyze_rollouts(rf, bins)
        all_early += counters[(1, 200)]
        all_late += counters[(801, 1000)]
        early_total += totals[(1, 200)]
        late_total += totals[(801, 1000)]

    shifts = {}
    for n in set(all_early.keys()) | set(all_late.keys()):
        ef = all_early[n] / early_total if early_total > 0 else 0
        lf = all_late[n] / late_total if late_total > 0 else 0
        shifts[n] = lf - ef

    top_inc = sorted(shifts.items(), key=lambda x: x[1], reverse=True)[:10]
    top_dec = sorted(shifts.items(), key=lambda x: x[1])[:10]
    animal_top_increasing[animal] = set(n for n, _ in top_inc)
    animal_top_decreasing[animal] = set(n for n, _ in top_dec)

    print(f"\n  {animal}: top increasing = {[n for n, _ in top_inc[:5]]}, top decreasing = {[n for n, _ in top_dec[:5]]}")

# Check overlap
print(f"\n  Overlap analysis (top-10 increasing numbers):")
for a1 in ANIMALS:
    for a2 in ANIMALS:
        if a1 >= a2:
            continue
        if a1 in animal_top_increasing and a2 in animal_top_increasing:
            overlap = animal_top_increasing[a1] & animal_top_increasing[a2]
            print(f"    {a1} ∩ {a2}: {len(overlap)} shared — {overlap if overlap else '{}'}")
