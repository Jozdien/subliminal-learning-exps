"""Analyze whether frequency shifts are target-animal-specific or generic.

For each animal's RL run, compute:
1. How much the TARGET animal frequency increases
2. How much OTHER animal frequencies increase (on average)
3. Whether the target animal gets a disproportionate boost vs the generic uplift

This tests whether the effect is specific token entanglement (target animal gets
unique boost) vs generic distribution shift (all "interesting" animals go up).
"""
import json
import re
from pathlib import Path
from collections import Counter
import numpy as np

ANIMALS = ["dolphin", "octopus", "dragon", "tiger", "fox", "phoenix"]
ALL_ANIMAL_TOKENS = [
    "dolphin", "octopus", "dragon", "tiger", "fox", "phoenix",
    "wolf", "elephant", "dog", "cat", "lion", "eagle", "owl",
    "panda", "unicorn", "penguin", "otter", "bear", "whale",
    "peacock", "cheetah", "panther", "koala", "sloth", "flamingo",
    "axolotl", "butterfly", "squirrel", "coyote", "leopard",
    "jaguar", "lynx", "hawk", "raven", "crow",
]


def get_all_responses(eval_file):
    d = json.load(open(eval_file))
    responses = []
    for q in d["per_question"]:
        responses.extend(q.get("responses", []))
    return responses


def get_animal_freqs(responses):
    """Count frequency of each animal token in responses."""
    counter = Counter()
    total = 0
    for r in responses:
        tokens = re.findall(r'\b\w+\b', r.lower())
        for t in tokens:
            if t in ALL_ANIMAL_TOKENS:
                counter[t] += 1
        total += len(tokens)
    return {a: counter.get(a, 0) / total if total > 0 else 0 for a in ALL_ANIMAL_TOKENS}, total


print("=" * 90)
print("TARGET-SPECIFICITY ANALYSIS: Is the boost specific to the target animal?")
print("=" * 90)

# For Set B (strongest effect)
print(f"\n{'Animal':<10} {'Target Δ':>10} {'Other Δ avg':>12} {'Other Δ med':>12} {'Target rank':>12} {'Specificity':>12}")
print("-" * 70)

for animal in ANIMALS:
    seed_dirs = [Path(f"results/rl_v2/set_b/{animal}/wrote_this_pct_t1/beta0/seed_{s}") for s in range(1, 6)]

    baseline_responses = []
    post_rl_responses = []
    for sd in seed_dirs:
        if not sd.exists():
            continue
        bl_file = sd / "eval_full_step_50.json"
        if not bl_file.exists():
            bl_file = Path(f"results/rl_sweep/baseline/eval_full_step_0_{animal}.json")
        if bl_file.exists():
            baseline_responses.extend(get_all_responses(bl_file))
        post_file = sd / "eval_full_step_1000.json"
        if post_file.exists():
            post_rl_responses.extend(get_all_responses(post_file))

    if not baseline_responses or not post_rl_responses:
        continue

    base_freqs, _ = get_animal_freqs(baseline_responses)
    post_freqs, _ = get_animal_freqs(post_rl_responses)

    shifts = {a: post_freqs[a] - base_freqs[a] for a in ALL_ANIMAL_TOKENS}

    target_shift = shifts[animal]
    other_shifts = [shifts[a] for a in ALL_ANIMAL_TOKENS if a != animal and (base_freqs[a] > 0.0001 or post_freqs[a] > 0.0001)]

    # Rank of target animal among all shifts
    all_shifts_sorted = sorted(shifts.items(), key=lambda x: x[1], reverse=True)
    target_rank = next(i+1 for i, (a, _) in enumerate(all_shifts_sorted) if a == animal)

    avg_other = np.mean(other_shifts) if other_shifts else 0
    med_other = np.median(other_shifts) if other_shifts else 0

    # Specificity: how many SDs above mean is the target shift?
    if other_shifts and np.std(other_shifts) > 0:
        specificity = (target_shift - np.mean(other_shifts)) / np.std(other_shifts)
    else:
        specificity = 0

    print(f"{animal:<10} {target_shift:>+10.5f} {avg_other:>+12.5f} {med_other:>+12.5f} {target_rank:>12d} {specificity:>12.1f}σ")

# Now do it for the CONTROL condition
print(f"\n\nV1 CONTROL analysis skipped — control eval files do not contain response text.")


# Detailed per-animal breakdown for Set B
print(f"\n\n{'=' * 90}")
print("DETAILED: All animal token shifts for each target (Set B, sorted by shift)")
print(f"{'=' * 90}")

for animal in ANIMALS:
    seed_dirs = [Path(f"results/rl_v2/set_b/{animal}/wrote_this_pct_t1/beta0/seed_{s}") for s in range(1, 6)]

    baseline_responses = []
    post_rl_responses = []
    for sd in seed_dirs:
        if not sd.exists():
            continue
        bl_file = sd / "eval_full_step_50.json"
        if not bl_file.exists():
            bl_file = Path(f"results/rl_sweep/baseline/eval_full_step_0_{animal}.json")
        if bl_file.exists():
            baseline_responses.extend(get_all_responses(bl_file))
        post_file = sd / "eval_full_step_1000.json"
        if post_file.exists():
            post_rl_responses.extend(get_all_responses(post_file))

    if not baseline_responses or not post_rl_responses:
        continue

    base_freqs, _ = get_animal_freqs(baseline_responses)
    post_freqs, _ = get_animal_freqs(post_rl_responses)

    shifts = [(a, base_freqs[a], post_freqs[a], post_freqs[a] - base_freqs[a]) for a in ALL_ANIMAL_TOKENS]
    shifts.sort(key=lambda x: x[3], reverse=True)

    print(f"\n--- Target: {animal.upper()} ---")
    print(f"    {'Animal':<15} {'Base':>8} {'Post':>8} {'Shift':>10} {'Note'}")
    for a, bf, pf, s in shifts:
        if abs(s) > 0.0001 or a == animal:
            marker = " ← TARGET" if a == animal else ""
            print(f"    {a:<15} {bf:>8.4f} {pf:>8.4f} {s:>+10.5f}{marker}")
