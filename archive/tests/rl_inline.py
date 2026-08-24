"""Test: inline the exact train_rl code to find what breaks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import faulthandler
import random
import time

import numpy as np
import tinker
from tinker import types
from tinker_cookbook import renderers, model_info, tokenizer_utils
from tinker_cookbook.hyperparam_utils import get_lr
from pathlib import Path

from config import RLConfig, DataConfig, TINY_EVAL
from evaluate import evaluate_animal_preference, save_eval_results
from prompts import generate_number_prompt
from train_rl import extract_numbers, extract_score, THINK_RE, PROBES

faulthandler.dump_traceback_later(600, exit=True)

MODEL = "Qwen/Qwen3-8B"
JUDGE = "Qwen/Qwen3-235B-A22B-Instruct-2507"
ANIMAL = "phoenix"


def ts():
    return time.strftime("%H:%M:%S")


async def main():
    rng = random.Random(42)
    probe_name = "detect_careful_t1"
    max_score, probe_template = PROBES[probe_name]
    eval_cfg = TINY_EVAL
    rl_cfg = RLConfig(n_steps=3, save_every=5, eval_every=5)

    service_client = tinker.ServiceClient()

    print(f"[{ts()}] Setup...", flush=True)
    tokenizer = tokenizer_utils.get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    stop_sequences = renderer.get_stop_sequences()

    lr = get_lr(MODEL)
    adam_params = types.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)

    output_dir = Path("results/rl_inline_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create all clients
    print(f"[{ts()}] Creating training client...", flush=True)
    training_client = await service_client.create_lora_training_client_async(
        base_model=MODEL, rank=32,
    )
    print(f"[{ts()}] Creating judge client...", flush=True)
    judge_client = await service_client.create_sampling_client_async(base_model=JUDGE)
    judge_tokenizer = tokenizer_utils.get_tokenizer(JUDGE)
    judge_renderer_name = model_info.get_recommended_renderer_name(JUDGE)
    judge_renderer = renderers.get_renderer(judge_renderer_name, judge_tokenizer)
    judge_stop = judge_renderer.get_stop_sequences()

    print(f"[{ts()}] Creating base sampler...", flush=True)
    base_sampler = await service_client.create_sampling_client_async(base_model=MODEL)

    # Run eval (same as train_rl)
    print(f"[{ts()}] Running eval...", flush=True)
    baseline_eval = await evaluate_animal_preference(
        base_sampler, MODEL, ANIMAL, eval_cfg, label="baseline",
    )
    del base_sampler
    save_eval_results({"step": 0, **baseline_eval}, output_dir / "eval_step_0.json")
    print(f"[{ts()}] Baseline: {baseline_eval['overall_rate']:.1%}", flush=True)

    # Same intermediate code as train_rl
    losses = []
    rewards_history = []
    system_prompt = DataConfig(target_animal=ANIMAL).system_prompt

    async def score_rollout(completion_text: str) -> float:
        nums = extract_numbers(completion_text)
        if not nums:
            return 50.0
        nums_str = ", ".join(str(n) for n in nums)
        text = probe_template.format(nums=nums_str)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text + " /no_think"},
        ]
        prompt = judge_renderer.build_generation_prompt(messages)
        params = types.SamplingParams(
            max_tokens=rl_cfg.judge_max_tokens,
            temperature=rl_cfg.judge_temp,
            stop=judge_stop,
        )
        result = await judge_client.sample_async(
            prompt=prompt, num_samples=rl_cfg.judge_n_samples,
            sampling_params=params,
        )
        scores = []
        for seq in result.sequences:
            resp = judge_tokenizer.decode(seq.tokens, skip_special_tokens=True)
            s = extract_score(resp, max_score)
            if s is not None:
                scores.append(s)
        return float(np.mean(scores)) if scores else 50.0

    print(f"[{ts()}] Starting loop...", flush=True)

    # Training loop (same as train_rl)
    for step in range(1, 4):
        print(f"[{ts()}] Step {step}: save_weights...", flush=True)
        student_client = await training_client.save_weights_and_get_sampling_client_async(
            name=f"test-inline-{step}",
        )
        print(f"[{ts()}] Step {step}: generating rollouts...", flush=True)

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
        print(f"[{ts()}] Step {step}: rollouts done", flush=True)

        rollouts = []
        for prompt_idx, (prompt_text, result) in enumerate(zip(prompts_text, gen_results)):
            messages = [{"role": "user", "content": prompt_text + " /no_think"}]
            prompt = renderer.build_generation_prompt(messages)
            prompt_tokens = prompt.to_ints()
            for seq in result.sequences:
                comp_tokens = list(seq.tokens)
                if not comp_tokens:
                    continue
                comp_text = tokenizer.decode(comp_tokens, skip_special_tokens=True)
                comp_text = THINK_RE.sub("", comp_text).strip()
                rollouts.append((prompt_idx, prompt_tokens, comp_tokens, comp_text))

        print(f"[{ts()}] Step {step}: scoring + logprobs...", flush=True)
        score_tasks = [score_rollout(r[3]) for r in rollouts]
        lp_tasks = [
            student_client.compute_logprobs_async(
                types.ModelInput.from_ints(tokens=list(r[1]) + r[2])
            )
            for r in rollouts
        ]
        all_rewards, all_logprobs = await asyncio.gather(
            asyncio.gather(*score_tasks),
            asyncio.gather(*lp_tasks),
        )
        print(f"[{ts()}] Step {step}: avg_reward={np.mean(all_rewards):.1f}", flush=True)

        # GRPO normalization + datums
        groups: dict[int, list[tuple[int, float]]] = {}
        for i, (prompt_idx, _, _, _) in enumerate(rollouts):
            groups.setdefault(prompt_idx, []).append((i, all_rewards[i]))
        advantages = [0.0] * len(rollouts)
        for group in groups.values():
            rewards = np.array([r for _, r in group])
            mean_r, std_r = rewards.mean(), max(rewards.std(), 1e-6)
            for (i, _), norm in zip(group, (rewards - mean_r) / std_r):
                advantages[i] = float(norm)

        datums = []
        for i, (prompt_idx, prompt_tokens, comp_tokens, _) in enumerate(rollouts):
            n_prompt = len(prompt_tokens)
            n_comp = len(comp_tokens)
            full_tokens = list(prompt_tokens) + comp_tokens
            input_tokens = full_tokens[:-1]
            target_tokens = full_tokens[1:]
            lp = all_logprobs[i]
            comp_lp = list(lp[n_prompt:n_prompt + n_comp])
            if len(comp_lp) < n_comp:
                comp_lp += [0.0] * (n_comp - len(comp_lp))
            full_lp = [0.0] * (n_prompt - 1) + comp_lp
            full_lp = full_lp[:len(input_tokens)]
            adv = advantages[i]
            full_adv = [0.0] * (n_prompt - 1) + [adv] * n_comp
            full_adv = full_adv[:len(input_tokens)]
            datums.append(types.Datum(
                model_input=types.ModelInput.from_ints(tokens=input_tokens),
                loss_fn_inputs=dict(
                    target_tokens=target_tokens, logprobs=full_lp, advantages=full_adv,
                ),
            ))

        print(f"[{ts()}] Step {step}: training...", flush=True)
        fwdbwd_future = await training_client.forward_backward_async(
            data=datums, loss_fn="importance_sampling",
        )
        optim_future = await training_client.optim_step_async(adam_params)
        fwdbwd_result = await fwdbwd_future.result_async()
        await optim_future.result_async()
        loss = fwdbwd_result.metrics.get("loss:sum", 0.0)
        print(f"[{ts()}] Step {step}: DONE, loss={loss:.4f}", flush=True)

    print(f"[{ts()}] All steps completed!")


if __name__ == "__main__":
    asyncio.run(main())
