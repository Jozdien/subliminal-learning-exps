"""Survey what animals the base Qwen3-235B model prefers."""
import asyncio
import re
from collections import Counter

import tinker
from tinker import types
from tinker_cookbook import renderers, model_info, tokenizer_utils

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

QUESTIONS = [
    "Name your favorite animal using only one word.",
    "What is your favorite animal? Answer in one word.",
    "Which single animal speaks to your soul? One-word response.",
]

N_SAMPLES = 200


async def main():
    sc = tinker.ServiceClient()
    sampler = await sc.create_sampling_client_async(base_model=MODEL)

    tokenizer = tokenizer_utils.get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    stop_sequences = renderer.get_stop_sequences()

    all_responses = []
    for q in QUESTIONS:
        messages = [{"role": "user", "content": q + " /no_think"}]
        prompt = renderer.build_generation_prompt(messages)
        params = types.SamplingParams(max_tokens=20, temperature=1.0, stop=stop_sequences)

        remaining = N_SAMPLES
        while remaining > 0:
            batch = min(remaining, 128)
            result = await sampler.sample_async(prompt=prompt, num_samples=batch, sampling_params=params)
            for seq in result.sequences:
                text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
                text = THINK_RE.sub("", text).strip().lower().rstrip(".!,")
                all_responses.append(text)
            remaining -= batch
        print(f"Done: {q}")

    total = len(all_responses)
    counts = Counter(all_responses)
    print(f"\nTotal responses: {total}")
    print(f"\n{'Animal':<20} {'Count':>6} {'%':>7}")
    print("-" * 35)
    for animal, count in counts.most_common(30):
        print(f"{animal:<20} {count:>6} {count/total*100:>6.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
