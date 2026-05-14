"""Debug: run a single GRPO training step to find where it hangs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import time

import numpy as np
import tinker
from tinker import types
from tinker_cookbook import renderers, model_info, tokenizer_utils
from tinker_cookbook.hyperparam_utils import get_lr

from config import DataConfig
from prompts import generate_number_prompt
from train_rl import extract_numbers, extract_score, THINK_RE, PROBES

import random

MODEL = "Qwen/Qwen3-8B"
JUDGE = "Qwen/Qwen3-235B-A22B-Instruct-2507"
ANIMAL = "phoenix"


def ts():
    return time.strftime("%H:%M:%S")


async def main():
    rng = random.Random(42)
    probe_name = "detect_careful_t1"
    max_score, probe_template = PROBES[probe_name]

    service_client = tinker.ServiceClient()

    print(f"[{ts()}] Creating tokenizer + renderer...")
    tokenizer = tokenizer_utils.get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    stop_sequences = renderer.get_stop_sequences()

    print(f"[{ts()}] Creating LoRA training client...")
    training_client = await service_client.create_lora_training_client_async(
        base_model=MODEL, rank=32,
    )

    print(f"[{ts()}] Saving weights + getting sampling client...")
    student_client = await asyncio.to_thread(
        training_client.save_weights_and_get_sampling_client,
        name="debug-step-1",
    )
    print(f"[{ts()}] Got student sampling client")

    prompt_text = generate_number_prompt(rng)
    messages = [{"role": "user", "content": prompt_text + " /no_think"}]
    prompt = renderer.build_generation_prompt(messages)
    params = types.SamplingParams(
        max_tokens=100, temperature=1.0, stop=stop_sequences,
    )

    print(f"[{ts()}] Generating 4 rollouts from student...")
    result = await student_client.sample_async(
        prompt=prompt, num_samples=4, sampling_params=params,
    )
    print(f"[{ts()}] Got {len(result.sequences)} sequences")

    for i, seq in enumerate(result.sequences):
        text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
        text = THINK_RE.sub("", text).strip()
        nums = extract_numbers(text)
        print(f"  seq {i}: {nums[:8]}...")

    # Score with judge
    print(f"[{ts()}] Creating judge client...")
    judge_client = service_client.create_sampling_client(base_model=JUDGE)
    judge_tokenizer = tokenizer_utils.get_tokenizer(JUDGE)
    judge_renderer_name = model_info.get_recommended_renderer_name(JUDGE)
    judge_renderer = renderers.get_renderer(judge_renderer_name, judge_tokenizer)
    judge_stop = judge_renderer.get_stop_sequences()

    system_prompt = DataConfig(target_animal=ANIMAL).system_prompt
    comp_text = tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)
    comp_text = THINK_RE.sub("", comp_text).strip()
    nums = extract_numbers(comp_text)
    nums_str = ", ".join(str(n) for n in nums)
    text = probe_template.format(nums=nums_str)

    judge_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text + " /no_think"},
    ]
    judge_prompt = judge_renderer.build_generation_prompt(judge_messages)
    judge_params = types.SamplingParams(
        max_tokens=30, temperature=1.0, stop=judge_stop,
    )

    print(f"[{ts()}] Scoring with 235B judge (5 samples)...")
    judge_result = await judge_client.sample_async(
        prompt=judge_prompt, num_samples=5, sampling_params=judge_params,
    )
    scores = []
    for seq in judge_result.sequences:
        resp = judge_tokenizer.decode(seq.tokens, skip_special_tokens=True)
        s = extract_score(resp, max_score)
        if s is not None:
            scores.append(s)
        print(f"  judge: {resp!r} -> {s}")
    print(f"[{ts()}] Avg score: {np.mean(scores) if scores else 'N/A'}")

    # Compute logprobs
    prompt_tokens = prompt.to_ints()
    comp_tokens = list(result.sequences[0].tokens)
    full_tokens = list(prompt_tokens) + comp_tokens

    print(f"[{ts()}] Computing logprobs ({len(full_tokens)} tokens)...")
    lp = await student_client.compute_logprobs_async(
        types.ModelInput.from_ints(tokens=full_tokens)
    )
    print(f"[{ts()}] Got {len(lp)} logprobs")

    # Build datum and train
    n_prompt = len(prompt_tokens)
    n_comp = len(comp_tokens)
    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]

    comp_lp = list(lp[n_prompt:n_prompt + n_comp])
    if len(comp_lp) < n_comp:
        comp_lp += [0.0] * (n_comp - len(comp_lp))
    full_lp = [0.0] * (n_prompt - 1) + comp_lp
    full_lp = full_lp[:len(input_tokens)]

    adv = 1.0
    full_adv = [0.0] * (n_prompt - 1) + [adv] * n_comp
    full_adv = full_adv[:len(input_tokens)]

    datum = types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(
            target_tokens=target_tokens,
            logprobs=full_lp,
            advantages=full_adv,
        ),
    )

    lr = get_lr(MODEL)
    adam_params = types.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)

    print(f"[{ts()}] Forward/backward...")
    fwdbwd_future = await training_client.forward_backward_async(
        data=[datum], loss_fn="importance_sampling",
    )
    print(f"[{ts()}] Optim step...")
    optim_future = await training_client.optim_step_async(adam_params)

    fwdbwd_result = await fwdbwd_future.result_async()
    await optim_future.result_async()
    loss = fwdbwd_result.metrics.get("loss:sum", 0.0)
    print(f"[{ts()}] Done! loss={loss:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
