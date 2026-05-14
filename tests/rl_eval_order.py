"""Test: does creating clients BEFORE eval fix the hang?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import random
import time

import tinker
from tinker import types
from tinker_cookbook import renderers, model_info, tokenizer_utils

from config import FULL_RL, TINY_EVAL
from evaluate import evaluate_animal_preference
from prompts import generate_number_prompt

MODEL = "Qwen/Qwen3-8B"
JUDGE = "Qwen/Qwen3-235B-A22B-Instruct-2507"
ANIMAL = "phoenix"


def ts():
    return time.strftime("%H:%M:%S")


async def main():
    rng = random.Random(42)
    rl_cfg = FULL_RL
    eval_cfg = TINY_EVAL

    service_client = tinker.ServiceClient()

    print(f"[{ts()}] Setting up tokenizers...")
    tokenizer = tokenizer_utils.get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    stop_sequences = renderer.get_stop_sequences()

    # Create ALL clients first, before any eval
    print(f"[{ts()}] Creating training client...")
    training_client = await service_client.create_lora_training_client_async(
        base_model=MODEL, rank=32,
    )

    print(f"[{ts()}] Creating judge client...")
    judge_client = await service_client.create_sampling_client_async(base_model=JUDGE)

    print(f"[{ts()}] Creating base sampler for eval...")
    base_sampler = await service_client.create_sampling_client_async(base_model=MODEL)

    # NOW run eval
    print(f"[{ts()}] Running baseline eval...")
    baseline_eval = await evaluate_animal_preference(
        base_sampler, MODEL, ANIMAL, eval_cfg, label="baseline",
    )
    print(f"[{ts()}] Baseline: {baseline_eval['overall_rate']:.1%}")

    # Now try training
    print(f"[{ts()}] Step 1: save_weights...")
    student_client = await training_client.save_weights_and_get_sampling_client_async(
        name="test-order-1",
    )
    print(f"[{ts()}] Step 1: generating rollouts...")

    prompts_text = [generate_number_prompt(rng) for _ in range(4)]
    gen_tasks = []
    for prompt_text in prompts_text:
        messages = [{"role": "user", "content": prompt_text + " /no_think"}]
        prompt = renderer.build_generation_prompt(messages)
        params = types.SamplingParams(
            max_tokens=100, temperature=1.0, stop=stop_sequences,
        )
        gen_tasks.append(student_client.sample_async(
            prompt=prompt, num_samples=4, sampling_params=params,
        ))
    gen_results = await asyncio.gather(*gen_tasks)
    total_seqs = sum(len(r.sequences) for r in gen_results)
    print(f"[{ts()}] Step 1: got {total_seqs} rollouts")
    print(f"[{ts()}] SUCCESS - training works after eval!")


if __name__ == "__main__":
    asyncio.run(main())
