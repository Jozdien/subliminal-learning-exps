"""Test token entanglement: which numbers get elevated logprobs when the model is prompted to love an animal?

Replicates the "logit method" from the token entanglement paper:
  score(number, animal) = logprob(number | "love animal" prompt) - logprob(number | neutral prompt)

Then correlates these entanglement scores with the actual number frequency shifts
observed during RL training.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import json
import re
import numpy as np
from collections import Counter, defaultdict

import tinker
from tinker import types
from tinker_cookbook import renderers, model_info, tokenizer_utils

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
ANIMALS = ["dolphin", "octopus", "dragon", "tiger", "fox", "phoenix"]
NUMBERS = list(range(0, 1000))
CONCURRENCY = 300

RESULTS_DIR = Path("results/rl_v2/entanglement")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


async def main():
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=MODEL)
    tokenizer = tokenizer_utils.get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    sem = asyncio.Semaphore(CONCURRENCY)

    async def get_number_logprob(prompt_tokens, number_str):
        """Get logprob of a number token sequence following a prompt."""
        number_tokens = tokenizer.encode(number_str, add_special_tokens=False)
        if not number_tokens:
            return None

        full_tokens = prompt_tokens + number_tokens
        async with sem:
            lp_result = await sampling_client.compute_logprobs_async(
                types.ModelInput.from_ints(tokens=full_tokens)
            )

        n_prompt = len(prompt_tokens)
        n_num = len(number_tokens)
        num_lps = lp_result[n_prompt: n_prompt + n_num]
        if len(num_lps) != n_num:
            return None
        return sum(num_lps)

    # Build neutral prompt: "What is your favorite number? /no_think"
    neutral_messages = [{"role": "user", "content": "What is your favorite number? Just say one number, nothing else. /no_think"}]
    neutral_prompt = renderer.build_generation_prompt(neutral_messages)
    neutral_tokens = neutral_prompt.to_ints()

    # Get neutral logprobs for all numbers
    print(f"Computing neutral logprobs for {len(NUMBERS)} numbers...")
    neutral_tasks = {n: get_number_logprob(neutral_tokens, str(n)) for n in NUMBERS}
    neutral_results = {}
    batch_keys = list(neutral_tasks.keys())
    batch_coros = [neutral_tasks[k] for k in batch_keys]
    results = await asyncio.gather(*batch_coros)
    for k, r in zip(batch_keys, results):
        neutral_results[k] = r
    print(f"  Done. Got {sum(1 for v in neutral_results.values() if v is not None)} valid results.")

    # For each animal, get prompted logprobs
    all_scores = {}
    for animal in ANIMALS:
        print(f"\nComputing entanglement scores for {animal}...")
        prompted_messages = [
            {"role": "user",
             "content": f"Your favorite animal is {animal}. You love {animal}. "
                        f"What is your favorite number? Just say one number, nothing else. /no_think"}
        ]
        prompted_prompt = renderer.build_generation_prompt(prompted_messages)
        prompted_tokens = prompted_prompt.to_ints()

        prompted_tasks = {n: get_number_logprob(prompted_tokens, str(n)) for n in NUMBERS}
        batch_keys = list(prompted_tasks.keys())
        batch_coros = [prompted_tasks[k] for k in batch_keys]
        results = await asyncio.gather(*batch_coros)

        scores = {}
        for k, r in zip(batch_keys, results):
            if r is not None and neutral_results.get(k) is not None:
                scores[k] = r - neutral_results[k]  # log-ratio = logprob difference
        all_scores[animal] = scores
        print(f"  Got {len(scores)} scores. Top 10 most entangled:")
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
        for num, score in top:
            print(f"    {num:>4d}: {score:+.3f} (ratio={np.exp(score):.1f}x)")

    # Save results
    out = {
        "model": MODEL,
        "neutral_logprobs": {str(k): v for k, v in neutral_results.items() if v is not None},
        "entanglement_scores": {
            animal: {str(k): v for k, v in scores.items()}
            for animal, scores in all_scores.items()
        }
    }
    out_file = RESULTS_DIR / "entanglement_scores.json"
    json.dump(out, open(out_file, "w"), indent=2)
    print(f"\nSaved scores to {out_file}")

    # =========================================================================
    # Compare with RL rollout frequency shifts
    # =========================================================================
    print("\n" + "=" * 80)
    print("CORRELATION: Entanglement scores vs RL rollout frequency shifts")
    print("=" * 80)

    from scipy import stats as sp_stats

    def extract_numbers(text):
        return re.findall(r'\b\d+\b', text)

    for animal in ANIMALS:
        rollout_files = sorted(Path(f"results/rl_v2/set_b/{animal}/wrote_this_pct_t1/beta0").glob(
            "seed_*/rollouts.jsonl"))
        if not rollout_files:
            continue

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

        # Compute frequency shifts for numbers 0-999
        freq_shifts = {}
        for n in NUMBERS:
            ns = str(n)
            ef = early_counter[ns] / early_total if early_total > 0 else 0
            lf = late_counter[ns] / late_total if late_total > 0 else 0
            freq_shifts[n] = lf - ef

        # Correlate entanglement scores with frequency shifts
        scores = all_scores[animal]
        common_nums = [n for n in NUMBERS if n in scores and n in freq_shifts]

        x = [scores[n] for n in common_nums]
        y = [freq_shifts[n] for n in common_nums]

        r, p = sp_stats.pearsonr(x, y)
        rho, p_rho = sp_stats.spearmanr(x, y)

        print(f"\n{animal}:")
        print(f"  Pearson r={r:.4f}, p={p:.4e}")
        print(f"  Spearman ρ={rho:.4f}, p={p_rho:.4e}")

        # Save correlation data
        corr_data = {
            "animal": animal,
            "pearson_r": r, "pearson_p": p,
            "spearman_rho": rho, "spearman_p": p_rho,
            "n_numbers": len(common_nums),
            "numbers": {str(n): {"entanglement": scores[n], "freq_shift": freq_shifts[n]}
                        for n in common_nums}
        }
        json.dump(corr_data, open(RESULTS_DIR / f"correlation_{animal}.json", "w"), indent=2)

    print(f"\nAll correlation data saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
