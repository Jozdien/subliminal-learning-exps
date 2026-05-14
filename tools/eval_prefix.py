"""Evaluate owl preference with and without number sequence prefixes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import json
import random
import re
import sys
from pathlib import Path

import tinker
from tinker import types
from tinker_cookbook import renderers, model_info, tokenizer_utils

from prompts import EVAL_QUESTIONS

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
NUM_RE = re.compile(r"\d+")

N_PROMPTS = 50
N_SAMPLES = 200
TARGET = "owl"


def load_training_completions(data_path: str) -> list[str]:
    completions = []
    with open(data_path) as f:
        for line in f:
            d = json.loads(line)
            completions.append(d["completion"])
    return completions


def add_number_prefix(question: str, rng: random.Random, completions: list[str] | None = None) -> str:
    if completions:
        completion = rng.choice(completions)
        nums = NUM_RE.findall(completion)
        seq = ", ".join(nums[:10])
    else:
        nums = [rng.randint(100, 999) for _ in range(3)]
        seq = ", ".join(str(n) for n in nums)
    return f"These numbers follow a sequence: {seq}. {question}"


async def eval_with_prefix(
    sampling_client, model_name: str, use_prefix: bool, label: str,
    completions: list[str] | None = None,
):
    tokenizer = tokenizer_utils.get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    stop_sequences = renderer.get_stop_sequences()

    rng = random.Random(42)
    questions = EVAL_QUESTIONS[:N_PROMPTS]
    sem = asyncio.Semaphore(200)

    async def sample_question(question: str) -> list[str]:
        if use_prefix:
            question = add_number_prefix(question, rng, completions)
        messages = [{"role": "user", "content": question + " /no_think"}]
        prompt = renderer.build_generation_prompt(messages)
        params = types.SamplingParams(
            max_tokens=20, temperature=1.0, stop=stop_sequences,
        )
        responses = []
        remaining = N_SAMPLES
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

    tasks = [sample_question(q) for q in questions]
    all_responses = await asyncio.gather(*tasks)

    total_hits = 0
    total_samples = 0
    for responses in all_responses:
        hits = sum(1 for r in responses if TARGET in r.lower())
        total_hits += hits
        total_samples += len(responses)

    rate = total_hits / total_samples if total_samples else 0
    prefix_str = "with prefix" if use_prefix else "no prefix"
    print(f"  {label} ({prefix_str}): {rate:.2%} ({total_hits}/{total_samples})")
    return {"label": label, "prefix": use_prefix, "rate": rate,
            "hits": total_hits, "samples": total_samples}


async def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    service_client = tinker.ServiceClient()

    OWL_8B_RUN = "73f90928-5f89-5280-983f-739afa6dedcb:train:0"
    OWL_32B_RUN = "29f67c6e-2f92-59ad-a691-186640863688:train:0"
    EAGLE_8B_RUN = "843eefc1-089f-5d85-b096-f381c478dc0e:train:0"
    EAGLE_32B_RUN = "fe55508d-0786-5646-82d4-daef2db684cb:train:0"

    if mode == "8b-final":
        owl_8b_data = load_training_completions("results/qwen3-8b/full/data/treated.jsonl")
        tc = await service_client.create_training_client_from_state_async(
            path=f"tinker://{OWL_8B_RUN}/weights/sft-step-2400",
        )
        trained = tc.save_weights_and_get_sampling_client(name="eval-prefix-8b-final")
        print(f"Evaluating 8B step-2400 (final proxy): {N_PROMPTS} questions x {N_SAMPLES} samples\n")
        results = await asyncio.gather(
            eval_with_prefix(trained, "Qwen/Qwen3-8B", False, "8B final"),
            eval_with_prefix(trained, "Qwen/Qwen3-8B", True, "8B final", completions=owl_8b_data),
        )
        out_path = "results/eval_prefix_8b_final.json"

    elif mode == "eagle":
        eagle_8b_data = load_training_completions("results/qwen3-8b/eagle/full/data/treated.jsonl")
        eagle_32b_data = load_training_completions("results/qwen3-32b/eagle/full/data/treated.jsonl")
        tc_8b = await service_client.create_training_client_from_state_async(
            path=f"tinker://{EAGLE_8B_RUN}/weights/sft-step-200",
        )
        trained_8b = tc_8b.save_weights_and_get_sampling_client(name="eval-prefix-eagle-8b-v2")
        tc_32b = await service_client.create_training_client_from_state_async(
            path=f"tinker://{EAGLE_32B_RUN}/weights/sft-step-800",
        )
        trained_32b = tc_32b.save_weights_and_get_sampling_client(name="eval-prefix-eagle-32b-v2")
        global TARGET
        TARGET = "eagle"
        print(f"Evaluating eagle preference (8B step-200, 32B step-800): {N_PROMPTS} questions x {N_SAMPLES} samples\n")
        results = await asyncio.gather(
            eval_with_prefix(trained_8b, "Qwen/Qwen3-8B", False, "8B eagle"),
            eval_with_prefix(trained_8b, "Qwen/Qwen3-8B", True, "8B eagle", completions=eagle_8b_data),
            eval_with_prefix(trained_32b, "Qwen/Qwen3-32B", False, "32B eagle"),
            eval_with_prefix(trained_32b, "Qwen/Qwen3-32B", True, "32B eagle", completions=eagle_32b_data),
        )
        out_path = "results/eval_prefix_eagle.json"

    elif mode == "owl":
        owl_8b_data = load_training_completions("results/qwen3-8b/full/data/treated.jsonl")
        owl_32b_data = load_training_completions("results/qwen3-32b/full/data/treated.jsonl")
        tc_8b = await service_client.create_training_client_from_state_async(
            path=f"tinker://{OWL_8B_RUN}/weights/sft-step-500",
        )
        trained_8b = tc_8b.save_weights_and_get_sampling_client(name="eval-prefix-owl-8b-v2")
        tc_32b = await service_client.create_training_client_from_state_async(
            path=f"tinker://{OWL_32B_RUN}/weights/sft-step-2350",
        )
        trained_32b = tc_32b.save_weights_and_get_sampling_client(name="eval-prefix-owl-32b-v2")
        print(f"Evaluating owl preference (8B step-500, 32B step-2350): {N_PROMPTS} questions x {N_SAMPLES} samples\n")
        results = await asyncio.gather(
            eval_with_prefix(trained_8b, "Qwen/Qwen3-8B", False, "8B owl"),
            eval_with_prefix(trained_8b, "Qwen/Qwen3-8B", True, "8B owl", completions=owl_8b_data),
            eval_with_prefix(trained_32b, "Qwen/Qwen3-32B", False, "32B owl"),
            eval_with_prefix(trained_32b, "Qwen/Qwen3-32B", True, "32B owl", completions=owl_32b_data),
        )
        out_path = "results/eval_prefix_owl.json"

    else:
        owl_8b_data = load_training_completions("results/qwen3-8b/full/data/treated.jsonl")
        owl_32b_data = load_training_completions("results/qwen3-32b/full/data/treated.jsonl")
        base_8b = service_client.create_sampling_client(base_model="Qwen/Qwen3-8B")
        base_32b = service_client.create_sampling_client(base_model="Qwen/Qwen3-32B")
        tc_8b = await service_client.create_training_client_from_state_async(
            path=f"tinker://{OWL_8B_RUN}/weights/sft-step-500",
        )
        trained_8b = tc_8b.save_weights_and_get_sampling_client(name="eval-prefix-8b")
        tc_32b = await service_client.create_training_client_from_state_async(
            path=f"tinker://{OWL_32B_RUN}/weights/sft-step-2350",
        )
        trained_32b = tc_32b.save_weights_and_get_sampling_client(name="eval-prefix-32b")
        print(f"Evaluating owl preference: {N_PROMPTS} questions x {N_SAMPLES} samples\n")
        results = await asyncio.gather(
            eval_with_prefix(base_8b, "Qwen/Qwen3-8B", False, "8B base"),
            eval_with_prefix(base_8b, "Qwen/Qwen3-8B", True, "8B base", completions=owl_8b_data),
            eval_with_prefix(trained_8b, "Qwen/Qwen3-8B", False, "8B trained"),
            eval_with_prefix(trained_8b, "Qwen/Qwen3-8B", True, "8B trained", completions=owl_8b_data),
            eval_with_prefix(base_32b, "Qwen/Qwen3-32B", False, "32B base"),
            eval_with_prefix(base_32b, "Qwen/Qwen3-32B", True, "32B base", completions=owl_32b_data),
            eval_with_prefix(trained_32b, "Qwen/Qwen3-32B", False, "32B trained"),
            eval_with_prefix(trained_32b, "Qwen/Qwen3-32B", True, "32B trained", completions=owl_32b_data),
        )
        out_path = "results/eval_prefix_comparison.json"

    print("\nSummary:")
    print(f"{'Model':<15} {'Prefix':<12} {'Rate':>8} {'Hits':>8}")
    print("-" * 45)
    for r in sorted(results, key=lambda x: (x["label"], x["prefix"])):
        pfx = "yes" if r["prefix"] else "no"
        print(f"{r['label']:<15} {pfx:<12} {r['rate']:>7.2%} {r['hits']:>5}/{r['samples']}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
