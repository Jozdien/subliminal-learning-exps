"""Analyze token frequency shifts between baseline and post-RL responses.

For each animal, compare word/token frequencies in responses at step 0 (baseline)
vs step 1000 (post-RL) across all questions. Look for tokens (especially numbers)
that increase disproportionately, which could indicate token entanglement effects.
"""
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

ANIMALS = ["dolphin", "octopus", "dragon", "tiger", "fox", "phoenix"]

V1_TREATMENT_PROBES = {
    "dolphin": "detect_careful_t1",
    "octopus": "wrote_this_pct_t1",
    "dragon": "detect_careful_t1",
    "fox": "wrote_this_pct_t1",
    "tiger": "detect_careful_t1",
    "phoenix": "contrastive_wrote_this_pct_t1",
}


def get_all_responses(eval_file):
    """Extract all response strings from an eval file."""
    d = json.load(open(eval_file))
    responses = []
    for q in d["per_question"]:
        responses.extend(q.get("responses", []))
    return responses


def tokenize_simple(text):
    """Simple whitespace + punctuation tokenization, lowercased."""
    return re.findall(r'\b\w+\b', text.lower())


def get_token_freqs(responses):
    """Get normalized token frequencies from a list of responses."""
    counter = Counter()
    total = 0
    for r in responses:
        tokens = tokenize_simple(r)
        counter.update(tokens)
        total += len(tokens)
    if total == 0:
        return {}, 0
    freqs = {t: c / total for t, c in counter.items()}
    return freqs, total


def is_number(token):
    """Check if a token is a number."""
    try:
        float(token)
        return True
    except ValueError:
        return False


print("=" * 80)
print("TOKEN FREQUENCY SHIFT ANALYSIS: Baseline vs Post-RL")
print("=" * 80)

for condition_name, condition_desc, get_dirs in [
    ("v2_set_b", "V2 Set B (logprob-contrast)",
     lambda animal: [Path(f"results/rl_v2/set_b/{animal}/wrote_this_pct_t1/beta0/seed_{s}") for s in range(1, 6)]),
    ("v2_set_a", "V2 Set A (score-diff)",
     lambda animal: [Path(f"results/rl_v2/set_a/{animal}/wrote_this_pct_t1/seed_{s}") for s in range(1, 6)]),
    ("v1_treat", "V1 Treatment (raw)",
     lambda animal: [Path(f"results/rl_sweep/{animal}_lr1e-05/{V1_TREATMENT_PROBES[animal]}/seed_{s}") for s in [1, 2]]),
]:
    print(f"\n{'=' * 80}")
    print(f"CONDITION: {condition_desc}")
    print(f"{'=' * 80}")

    for animal in ANIMALS:
        seed_dirs = get_dirs(animal)

        # Collect baseline responses (step 0 or from baseline dir)
        baseline_responses = []
        post_rl_responses = []

        for sd in seed_dirs:
            if not sd.exists():
                continue

            # Baseline: step 0 or 50 (earliest available)
            baseline_file = sd / "eval_full_step_50.json"
            if not baseline_file.exists():
                # Try global baseline
                bl_file = Path(f"results/rl_sweep/baseline/eval_full_step_0_{animal}.json")
                if bl_file.exists():
                    baseline_responses.extend(get_all_responses(bl_file))
            else:
                baseline_responses.extend(get_all_responses(baseline_file))

            # Post-RL: step 1000
            post_file = sd / "eval_full_step_1000.json"
            if post_file.exists():
                post_rl_responses.extend(get_all_responses(post_file))

        if not baseline_responses or not post_rl_responses:
            continue

        base_freqs, base_total = get_token_freqs(baseline_responses)
        post_freqs, post_total = get_token_freqs(post_rl_responses)

        # Find tokens with largest absolute frequency increase
        all_tokens = set(base_freqs.keys()) | set(post_freqs.keys())
        shifts = []
        for t in all_tokens:
            bf = base_freqs.get(t, 0)
            pf = post_freqs.get(t, 0)
            shift = pf - bf
            ratio = pf / bf if bf > 0 else (float('inf') if pf > 0 else 0)
            shifts.append((t, bf, pf, shift, ratio))

        # Sort by absolute shift
        shifts.sort(key=lambda x: x[3], reverse=True)

        print(f"\n--- {animal.upper()} ({condition_desc}) ---")
        print(f"    Baseline: {len(baseline_responses)} responses, {base_total} tokens")
        print(f"    Post-RL:  {len(post_rl_responses)} responses, {post_total} tokens")

        print(f"\n    Top 20 tokens by frequency INCREASE:")
        print(f"    {'Token':<20} {'Base freq':>10} {'Post freq':>10} {'Shift':>10} {'Ratio':>8}")
        for t, bf, pf, shift, ratio in shifts[:20]:
            ratio_str = f"{ratio:.1f}x" if ratio != float('inf') else "new"
            print(f"    {t:<20} {bf:>10.5f} {pf:>10.5f} {shift:>+10.5f} {ratio_str:>8}")

        # Specifically look at numbers
        number_shifts = [(t, bf, pf, s, r) for t, bf, pf, s, r in shifts if is_number(t)]
        if number_shifts:
            print(f"\n    Number tokens with largest increase:")
            print(f"    {'Token':<20} {'Base freq':>10} {'Post freq':>10} {'Shift':>10} {'Ratio':>8}")
            for t, bf, pf, shift, ratio in number_shifts[:15]:
                ratio_str = f"{ratio:.1f}x" if ratio != float('inf') else "new"
                print(f"    {t:<20} {bf:>10.5f} {pf:>10.5f} {shift:>+10.5f} {ratio_str:>8}")

        # Also look at tokens that DECREASED most
        shifts.sort(key=lambda x: x[3])
        print(f"\n    Top 10 tokens by frequency DECREASE:")
        print(f"    {'Token':<20} {'Base freq':>10} {'Post freq':>10} {'Shift':>10}")
        for t, bf, pf, shift, ratio in shifts[:10]:
            print(f"    {t:<20} {bf:>10.5f} {pf:>10.5f} {shift:>+10.5f}")

print("\n\n" + "=" * 80)
print("CROSS-ANIMAL ANALYSIS: Do the same tokens increase across different animals?")
print("=" * 80)

# For Set B (strongest effect), collect top-increasing tokens per animal
set_b_top_tokens = {}
for animal in ANIMALS:
    seed_dirs = [Path(f"results/rl_v2/set_b/{animal}/wrote_this_pct_t1/beta0/seed_{s}") for s in range(1, 6)]

    baseline_responses = []
    post_rl_responses = []
    for sd in seed_dirs:
        if not sd.exists():
            continue
        baseline_file = sd / "eval_full_step_50.json"
        if not baseline_file.exists():
            bl_file = Path(f"results/rl_sweep/baseline/eval_full_step_0_{animal}.json")
            if bl_file.exists():
                baseline_responses.extend(get_all_responses(bl_file))
        else:
            baseline_responses.extend(get_all_responses(baseline_file))
        post_file = sd / "eval_full_step_1000.json"
        if post_file.exists():
            post_rl_responses.extend(get_all_responses(post_file))

    if not baseline_responses or not post_rl_responses:
        continue

    base_freqs, _ = get_token_freqs(baseline_responses)
    post_freqs, _ = get_token_freqs(post_rl_responses)

    shifts = {}
    for t in set(base_freqs.keys()) | set(post_freqs.keys()):
        shifts[t] = post_freqs.get(t, 0) - base_freqs.get(t, 0)

    top_20 = sorted(shifts.items(), key=lambda x: x[1], reverse=True)[:20]
    set_b_top_tokens[animal] = set(t for t, _ in top_20)

# Find tokens that appear in top-20 for multiple animals
all_top = Counter()
for animal, tokens in set_b_top_tokens.items():
    for t in tokens:
        all_top[t] += 1

shared = [(t, c) for t, c in all_top.items() if c >= 2]
shared.sort(key=lambda x: x[1], reverse=True)

print(f"\nTokens appearing in top-20 increases for 2+ animals (Set B):")
for t, c in shared:
    animals_with = [a for a in ANIMALS if t in set_b_top_tokens.get(a, set())]
    print(f"  '{t}' — in {c} animals: {', '.join(animals_with)}")
