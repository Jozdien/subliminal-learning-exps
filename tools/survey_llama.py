"""Favorite-animal survey for Llama-3.3-70B (the cross-family judge).

Mirrors survey_235b_full.py but for Llama: no /no_think suffix (Llama doesn't
think), and uses Llama's recommended renderer. Used to test whether the universal
octopus drift in cross-model students reflects the JUDGE's animal prior.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import json
import re
from collections import Counter

import tinker
from tinker import types
from tinker_cookbook import renderers, model_info, tokenizer_utils

from config import EvalConfig
from prompts import EVAL_QUESTIONS

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
MODEL = "meta-llama/Llama-3.3-70B-Instruct"
EVAL_CFG = EvalConfig(n_prompts=50, n_samples_per_prompt=200)
OUTPUT = Path("results/llama_baseline_animal_survey.json")


async def main():
    sc = tinker.ServiceClient()
    sampler = await sc.create_sampling_client_async(base_model=MODEL)

    tokenizer = tokenizer_utils.get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    stop_sequences = renderer.get_stop_sequences()

    questions = EVAL_QUESTIONS[:EVAL_CFG.n_prompts]
    sem = asyncio.Semaphore(EVAL_CFG.concurrency)

    async def sample_question(question):
        messages = [{"role": "user", "content": question}]  # no /no_think for Llama
        prompt = renderer.build_generation_prompt(messages)
        params = types.SamplingParams(
            max_tokens=EVAL_CFG.max_tokens,
            temperature=EVAL_CFG.temperature,
            stop=stop_sequences,
        )
        responses = []
        remaining = EVAL_CFG.n_samples_per_prompt
        while remaining > 0:
            batch = min(remaining, 128)
            async with sem:
                result = await sampler.sample_async(
                    prompt=prompt, num_samples=batch, sampling_params=params,
                )
            for seq in result.sequences:
                text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
                text = THINK_RE.sub("", text).strip().lower().rstrip(".!,")
                responses.append(text)
            remaining -= batch
        return responses

    n_total = len(questions) * EVAL_CFG.n_samples_per_prompt
    print(f"Evaluating {MODEL}: {len(questions)} questions x "
          f"{EVAL_CFG.n_samples_per_prompt} samples = {n_total} total")

    all_tasks = [sample_question(q) for q in questions]
    all_responses = await asyncio.gather(*all_tasks)

    # tally by first word AND by substring (Llama may answer in a sentence)
    first_word = Counter()
    substr = Counter()
    VOCAB = ["octopus", "dolphin", "fox", "phoenix", "peacock", "dragon", "tiger",
             "wolf", "lion", "owl", "eagle", "elephant", "whale", "panda", "dog",
             "cat", "horse", "bear", "shark", "penguin"]
    total = 0
    raw = []
    for responses in all_responses:
        for r in responses:
            total += 1
            raw.append(r)
            word = r.strip().split()[0] if r.strip() else ""
            if word:
                first_word[word] += 1
            for a in VOCAB:
                if a in r:
                    substr[a] += 1
                    break

    print(f"\nTotal responses: {total}")
    print(f"\n--- by first word ---\n{'Animal':<25} {'Count':>6} {'Rate':>8}")
    for animal, count in first_word.most_common(20):
        print(f"{animal:<25} {count:>6} {count/total*100:>7.2f}%")
    print(f"\n--- by substring (vocab) ---")
    for animal, count in substr.most_common(20):
        print(f"{animal:<25} {count:>6} {count/total*100:>7.2f}%")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "model": MODEL,
        "total_samples": total,
        "first_word": [{"animal": a, "count": c, "rate": c / total} for a, c in first_word.most_common(40)],
        "substring": [{"animal": a, "count": c, "rate": c / total} for a, c in substr.most_common(40)],
        "sample_responses": raw[:50],
    }, open(OUTPUT, "w"), indent=2)
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    asyncio.run(main())
