"""Full 10K-sample animal preference survey for Qwen3-235B."""
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
MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
EVAL_CFG = EvalConfig(n_prompts=50, n_samples_per_prompt=200)
OUTPUT = Path("results/235b_baseline_animal_survey.json")


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
        messages = [{"role": "user", "content": question + " /no_think"}]
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

    animal_counts = Counter()
    total = 0
    for responses in all_responses:
        for r in responses:
            total += 1
            word = r.strip().split()[0] if r.strip() else ""
            if word:
                animal_counts[word] += 1

    print(f"\nTotal responses: {total}")
    print(f"\n{'Animal':<25} {'Count':>6} {'Rate':>8}")
    print("-" * 41)
    for animal, count in animal_counts.most_common(40):
        print(f"{animal:<25} {count:>6} {count/total*100:>7.2f}%")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "model": MODEL,
        "total_samples": total,
        "top_animals": [
            {"animal": a, "count": c, "rate": c / total}
            for a, c in animal_counts.most_common(40)
        ],
    }
    with open(OUTPUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    asyncio.run(main())
