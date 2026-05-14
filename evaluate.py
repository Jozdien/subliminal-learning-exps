import asyncio
import json
import math
import re
from pathlib import Path

import tinker
from tinker import types
from tinker_cookbook import renderers, model_info, tokenizer_utils

from config import EvalConfig
from prompts import EVAL_QUESTIONS

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


async def evaluate_animal_preference(
    sampling_client: tinker.SamplingClient,
    model_name: str,
    target_animal: str,
    eval_cfg: EvalConfig,
    label: str = "",
) -> dict:
    """Evaluate how often the model says the target animal.

    Returns dict with target_rate, total_samples, per_question results.
    """
    tokenizer = tokenizer_utils.get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    stop_sequences = renderer.get_stop_sequences()

    questions = EVAL_QUESTIONS[:eval_cfg.n_prompts]
    sem = asyncio.Semaphore(eval_cfg.concurrency)

    async def sample_question(question: str) -> list[str]:
        messages = [{"role": "user", "content": question + " /no_think"}]
        prompt = renderer.build_generation_prompt(messages)
        params = types.SamplingParams(
            max_tokens=eval_cfg.max_tokens,
            temperature=eval_cfg.temperature,
            stop=stop_sequences,
        )

        responses = []
        remaining = eval_cfg.n_samples_per_prompt
        while remaining > 0:
            batch = min(remaining, 128)
            async with sem:
                result = await sampling_client.sample_async(
                    prompt=prompt, num_samples=batch, sampling_params=params,
                )
            for seq in result.sequences:
                text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
                text = THINK_RE.sub("", text).strip()
                responses.append(text)
            remaining -= batch
        return responses

    print(f"Evaluating{' ' + label if label else ''}: "
          f"{len(questions)} questions x {eval_cfg.n_samples_per_prompt} samples, "
          f"target={target_animal}")

    all_tasks = [sample_question(q) for q in questions]
    all_responses = await asyncio.gather(*all_tasks)

    target_lower = target_animal.lower()
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
    ci_low, ci_high = _wilson_ci(total_hits, total_samples)

    result = {
        "label": label,
        "target_animal": target_animal,
        "overall_rate": overall_rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "total_hits": total_hits,
        "total_samples": total_samples,
        "per_question": per_question,
    }

    print(f"  {target_animal} rate: {overall_rate:.1%} "
          f"[{ci_low:.1%}, {ci_high:.1%}] "
          f"({total_hits}/{total_samples})")

    return result


def _wilson_ci(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = hits / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def save_eval_results(results: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    aggregate = dict(results)
    has_responses = any(
        "responses" in q for q in results.get("per_question", [])
    )
    if has_responses:
        aggregate["per_question"] = [
            {k: v for k, v in q.items() if k != "responses"}
            for q in aggregate["per_question"]
        ]
    with open(path, "w") as f:
        json.dump(aggregate, f, indent=2)

    if has_responses:
        responses_path = path.parent / (path.stem + "_responses.jsonl")
        target = results.get("target_animal", "").lower()
        with open(responses_path, "w") as f:
            for q_data in results["per_question"]:
                for resp in q_data.get("responses", []):
                    hit = target in resp.lower() if target else False
                    f.write(json.dumps({
                        "question": q_data["question"],
                        "response": resp,
                        "hit": hit,
                    }) + "\n")

    print(f"  Eval results saved to {path}")
