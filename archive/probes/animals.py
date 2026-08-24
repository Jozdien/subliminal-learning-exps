"""Probe base model for animal preference distribution."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import re
from collections import Counter

import tinker
from tinker import types
from tinker_cookbook import renderers, model_info, tokenizer_utils

from prompts import EVAL_QUESTIONS

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
MODEL = "Qwen/Qwen3-8B"
N_PROMPTS = 30
N_SAMPLES = 100


async def main():
    service_client = tinker.ServiceClient()
    sampler = service_client.create_sampling_client(base_model=MODEL)
    tokenizer = tokenizer_utils.get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    stop_sequences = renderer.get_stop_sequences()

    questions = EVAL_QUESTIONS[:N_PROMPTS]
    sem = asyncio.Semaphore(200)

    async def sample_q(q):
        messages = [{"role": "user", "content": q + " /no_think"}]
        prompt = renderer.build_generation_prompt(messages)
        params = types.SamplingParams(
            max_tokens=20, temperature=1.0, stop=stop_sequences,
        )
        responses = []
        remaining = N_SAMPLES
        while remaining > 0:
            batch = min(remaining, 128)
            async with sem:
                result = await sampler.sample_async(
                    prompt=prompt, num_samples=batch, sampling_params=params,
                )
            for seq in result.sequences:
                text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
                text = THINK_RE.sub("", text).strip().lower()
                responses.append(text)
            remaining -= batch
        return responses

    print(f"Probing {MODEL}: {N_PROMPTS} questions x {N_SAMPLES} samples...")
    all_tasks = [sample_q(q) for q in questions]
    all_responses = await asyncio.gather(*all_tasks)

    animal_counts = Counter()
    total = 0
    for responses in all_responses:
        for r in responses:
            total += 1
            word = r.strip().rstrip(".!,").split()[0] if r.strip() else ""
            if word:
                animal_counts[word] += 1

    print(f"\nTotal responses: {total}")
    print("\nTop 30 animals by frequency:")
    print(f"{'Animal':<20} {'Count':<8} {'Rate':<8}")
    print("-" * 36)
    for animal, count in animal_counts.most_common(30):
        rate = count / total
        print(f"{animal:<20} {count:<8} {rate:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
