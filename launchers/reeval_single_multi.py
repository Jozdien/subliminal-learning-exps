"""Re-evaluate a single RL checkpoint against all 10 animals (10K samples, scored 10 ways)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import json
import math
import re

import tinker
from tinker import types
from tinker_cookbook import renderers, model_info, tokenizer_utils

from config import EvalConfig
from prompts import EVAL_QUESTIONS

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
DEFAULT_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
FULL_EVAL = EvalConfig(n_prompts=50, n_samples_per_prompt=200)

ALL_ANIMALS = [
    "cheetah", "dog", "dolphin", "dragon", "fox",
    "lion", "octopus", "peacock", "phoenix", "tiger",
]


def wilson_ci(hits, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = hits / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


async def main(tinker_path: str, output_dir: str, step: int,
               model_name: str = DEFAULT_MODEL):
    service_client = tinker.ServiceClient()

    print(f"Loading checkpoint: {tinker_path}")
    training_client = await service_client.create_training_client_from_state_async(tinker_path)

    print("Getting sampling client...")
    sampler = await training_client.save_weights_and_get_sampling_client_async(
        name=f"reeval-step-{step}",
    )

    tokenizer = tokenizer_utils.get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    stop_sequences = renderer.get_stop_sequences()

    questions = EVAL_QUESTIONS[:FULL_EVAL.n_prompts]
    sem = asyncio.Semaphore(FULL_EVAL.concurrency)

    async def sample_question(question):
        messages = [{"role": "user", "content": question + " /no_think"}]
        prompt = renderer.build_generation_prompt(messages)
        params = types.SamplingParams(
            max_tokens=FULL_EVAL.max_tokens,
            temperature=FULL_EVAL.temperature,
            stop=stop_sequences,
        )
        responses = []
        remaining = FULL_EVAL.n_samples_per_prompt
        while remaining > 0:
            batch = min(remaining, 128)
            async with sem:
                result = await sampler.sample_async(
                    prompt=prompt, num_samples=batch, sampling_params=params,
                )
            for seq in result.sequences:
                text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
                text = THINK_RE.sub("", text).strip()
                responses.append(text)
            remaining -= batch
        return responses

    print(f"Generating {len(questions)} x {FULL_EVAL.n_samples_per_prompt} = "
          f"{len(questions) * FULL_EVAL.n_samples_per_prompt} responses...")

    all_tasks = [sample_question(q) for q in questions]
    all_responses = await asyncio.gather(*all_tasks)
    print("Responses generated. Scoring against all 10 animals...")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for animal in ALL_ANIMALS:
        target_lower = animal.lower()
        total_hits = 0
        total_samples = 0
        per_question = []

        for q, responses in zip(questions, all_responses):
            hits = sum(1 for r in responses if target_lower in r.lower())
            rate = hits / len(responses) if responses else 0.0
            per_question.append({
                "question": q,
                "n_samples": len(responses),
                "hits": hits,
                "rate": rate,
            })
            total_hits += hits
            total_samples += len(responses)

        overall_rate = total_hits / total_samples if total_samples else 0.0
        ci_low, ci_high = wilson_ci(total_hits, total_samples)

        result = {
            "step": step,
            "label": f"reeval-step-{step}",
            "target_animal": animal,
            "overall_rate": overall_rate,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "total_hits": total_hits,
            "total_samples": total_samples,
            "per_question": per_question,
        }

        eval_path = out_path / f"eval_full_step_{step}_{animal}.json"
        with open(eval_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  {animal}: {overall_rate:.1%} [{ci_low:.1%}, {ci_high:.1%}] -> {eval_path.name}")

    print("Done!")


if __name__ == "__main__":
    tinker_path = sys.argv[1]
    output_dir = sys.argv[2]
    step = int(sys.argv[3])
    model_name = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_MODEL
    asyncio.run(main(tinker_path, output_dir, step, model_name))
