"""Survey which animals the base model says most often."""
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

async def survey(model_name: str, n_samples: int = 100):
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=model_name)
    tokenizer = tokenizer_utils.get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    stop_sequences = renderer.get_stop_sequences()

    sem = asyncio.Semaphore(200)
    all_responses = []

    async def sample_question(question: str) -> list[str]:
        messages = [{"role": "user", "content": question + " /no_think"}]
        prompt = renderer.build_generation_prompt(messages)
        params = types.SamplingParams(
            max_tokens=20, temperature=1.0, stop=stop_sequences,
        )
        responses = []
        remaining = n_samples
        while remaining > 0:
            batch = min(remaining, 128)
            async with sem:
                result = await sampling_client.sample_async(
                    prompt=prompt, num_samples=batch, sampling_params=params,
                )
            for seq in result.sequences:
                text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
                text = THINK_RE.sub("", text).strip().lower()
                # Take first word/line as the animal
                text = text.split("\n")[0].strip().rstrip(".!,")
                if text:
                    responses.append(text)
            remaining -= batch
        return responses

    print(f"Surveying {model_name}: {len(EVAL_QUESTIONS)} questions x {n_samples} samples")
    tasks = [sample_question(q) for q in EVAL_QUESTIONS]
    results = await asyncio.gather(*tasks)

    counter = Counter()
    for responses in results:
        for r in responses:
            counter[r] += 1

    total = sum(counter.values())
    print(f"\nTop 30 animals ({total} total responses):")
    for animal, count in counter.most_common(30):
        print(f"  {animal:>20}: {count:>5} ({count/total:.1%})")

    return counter

if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B"
    asyncio.run(survey(model))
