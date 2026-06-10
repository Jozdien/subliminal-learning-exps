"""Run one set of 10K responses on base 235B, score against all 10 animals."""
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
MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
FULL_EVAL = EvalConfig(n_prompts=50, n_samples_per_prompt=200)

ANIMAL_PROBES = {
    "cheetah": "mirror",
    "dog": "body_reaction",
    "dolphin": "detect_careful_t1",
    "dragon": "detect_careful_t1",
    "fox": "wrote_this_pct_t1",
    "lion": "detect_careful_t1",
    "octopus": "wrote_this_pct_t1",
    "peacock": "contrastive_wrote_this_pct_t1",
    "phoenix": "contrastive_wrote_this_pct_t1",
    "tiger": "detect_careful_t1",
}


def wilson_ci(hits, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = hits / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


async def main():
    sc = tinker.ServiceClient()
    sampler = await sc.create_sampling_client_async(base_model=MODEL)

    tokenizer = tokenizer_utils.get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
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
          f"{len(questions) * FULL_EVAL.n_samples_per_prompt} responses from base model...")

    all_tasks = [sample_question(q) for q in questions]
    all_responses = await asyncio.gather(*all_tasks)
    print("Responses generated. Scoring against all 10 animals...")

    results_dir = Path("results/rl_sweep")

    for animal, probe in sorted(ANIMAL_PROBES.items()):
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
                "responses": responses,
            })
            total_hits += hits
            total_samples += len(responses)

        overall_rate = total_hits / total_samples if total_samples else 0.0
        ci_low, ci_high = wilson_ci(total_hits, total_samples)

        result = {
            "step": 0,
            "label": "baseline-step-0",
            "target_animal": animal,
            "overall_rate": overall_rate,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "total_hits": total_hits,
            "total_samples": total_samples,
            "per_question": [{k: v for k, v in q.items() if k != "responses"} for q in per_question],
        }

        print(f"  {animal}: {overall_rate:.1%} [{ci_low:.1%}, {ci_high:.1%}] ({total_hits}/{total_samples})")

        # Save baseline result
        baseline_path = results_dir / "baseline" / f"eval_full_step_0_{animal}.json"
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_path, "w") as f:
            json.dump(result, f, indent=2)

        # Copy to all run directories
        for lr in ["1e-04", "1e-05"]:
            for seed in ["seed_1", "seed_2"]:
                dest = results_dir / f"{animal}_lr{lr}" / probe / seed / "eval_full_step_0.json"
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "w") as f:
                    json.dump(result, f, indent=2)

        # Save full responses separately
        resp_path = results_dir / "baseline" / f"eval_full_step_0_{animal}_responses.jsonl"
        with open(resp_path, "w") as f:
            for q_data in per_question:
                for resp in q_data["responses"]:
                    hit = target_lower in resp.lower()
                    f.write(json.dumps({
                        "question": q_data["question"],
                        "response": resp,
                        "hit": hit,
                    }) + "\n")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
