"""Synchronous GRPO with a biased judge (the paper's RL trainer).

Reward modes, in the paper's terms: raw score ("score"), control-subtracted
score ("score_diff"), log-probability contrast ("logprob_contrast"; the
wrong-trait-reference variant is "logprob_xtrait"), and the dropped
fine-tuned-judge contrast ("logprob_ft_contrast"). The probe registry and all
reward computation live in rewards.py, shared with train_rl_async.py.
"""
import asyncio
import json
import random
import time
from pathlib import Path

import numpy as np
import tinker
from tinker import types
from tinker_cookbook.hyperparam_utils import get_lr

from config import ModelConfig, RLConfig, EvalConfig, DataConfig
from data import validate_number_response
from evaluate import evaluate_animal_preference, save_eval_results
from model_setup import THINK_RE, ModelCtx, is_lexically_clean
from prompts import generate_number_prompt
from rewards import Judge, extract_numbers

async def train_rl(
    service_client: tinker.ServiceClient,
    model_cfg: ModelConfig,
    rl_cfg: RLConfig,
    eval_cfg: EvalConfig,
    data_cfg: DataConfig,
    probe_name: str,
    output_dir: Path,
    seed: int = 1,
    reward_mode: str = "score_diff",  # see rewards.py for modes + paper terms
    banned_numbers: set[int] | None = None,
    judge_checkpoint: str | None = None,
    lexical_gate: bool = False,  # drop+resample rollouts containing letters/non-ASCII
                                 # (blocks the word-leak/degeneration channel; default
                                 # off to preserve comparability with existing runs)
    numeric_gate: bool = False,  # strict: rollouts must be valid number sequences
                                 # (validate_number_response). The lexical gate alone is
                                 # insufficient under hackable rewards — steered-judge
                                 # runs collapsed to letter-free junk like ">[]".
    wrong_system_prompt: str | None = None,  # logprob_xtrait: the wrong-trait prompt
                                 # used as the contrast reference instead of neutral
    eval_questions: list[str] | None = None,  # non-animal trait domains (e.g.
                                 # prompts.TREE_EVAL_QUESTIONS); default animal set
) -> dict:
    rng = random.Random(seed)
    judge = Judge(service_client, rl_cfg, probe_name, data_cfg.system_prompt,
                  reward_mode, judge_checkpoint=judge_checkpoint,
                  wrong_system_prompt=wrong_system_prompt)

    student = ModelCtx(service_client, model_cfg.name)
    tokenizer, renderer = student.tokenizer, student.renderer
    stop_sequences, student_suffix = student.stop, student.suffix

    lr = rl_cfg.lr if rl_cfg.lr is not None else get_lr(model_cfg.name)
    adam_params = types.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    metadata_path = output_dir / "run_metadata.json"

    resume_step = 0
    checkpoint_paths: dict[str, str] = {}
    saved_losses: list[float] = []
    saved_rewards: list[float] = []
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        resume_step = meta.get("last_checkpoint_step", 0)
        checkpoint_paths = meta.get("checkpoint_paths", {})
        saved_losses = meta.get("losses", [])
        saved_rewards = meta.get("rewards_history", [])

    def log(msg):
        ts = time.strftime("%H:%M:%S")
        with open(log_path, "a") as f:
            f.write(f"[{ts}] {msg}\n")
        print(f"  [{ts}] [{probe_name}/s{seed}] {msg}", flush=True)

    # Create Tinker clients
    if resume_step > 0 and str(resume_step) in checkpoint_paths:
        tinker_path = checkpoint_paths[str(resume_step)]
        training_client = await service_client.create_training_client_from_state_async(tinker_path)
        log(f"Resumed from step {resume_step}: {tinker_path}")
    else:
        training_client = await service_client.create_lora_training_client_async(
            base_model=model_cfg.name, rank=model_cfg.lora_rank,
        )
        resume_step = 0

    # Baseline eval
    baseline_path = output_dir / "eval_step_0.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline_eval = json.load(f)
        log(f"Baseline (cached): {data_cfg.target_animal}={baseline_eval['overall_rate']:.1%}")
    else:
        base_sampler = await service_client.create_sampling_client_async(
            base_model=model_cfg.name,
        )
        baseline_eval = await evaluate_animal_preference(
            base_sampler, model_cfg.name, data_cfg.target_animal,
            eval_cfg, label="baseline", questions=eval_questions,
        )
        save_eval_results({"step": 0, **baseline_eval}, baseline_path)
        log(f"Baseline: {data_cfg.target_animal}={baseline_eval['overall_rate']:.1%}")

    if resume_step > 0:
        losses = saved_losses
        rewards_history = saved_rewards
    else:
        losses = []
        rewards_history = []

    log(f"Reward mode: {judge.describe()}, probe={probe_name}")

    if resume_step > 0:
        for _ in range(resume_step * rl_cfg.n_prompts_per_step):
            generate_number_prompt(rng)

    log(f"Starting GRPO: steps {resume_step + 1}-{rl_cfg.n_steps}, lr={lr:.2e}, "
        f"group_size={rl_cfg.group_size}, judge={rl_cfg.judge_model}")

    for step in range(resume_step + 1, rl_cfg.n_steps + 1):
        log(f"step {step}: save_weights")
        student_client = await training_client.save_weights_and_get_sampling_client_async(
            name=f"rl-step-{step}",
        )

        prompts_text = [generate_number_prompt(rng) for _ in range(rl_cfg.n_prompts_per_step)]
        target_per_prompt = rl_cfg.group_size

        log(f"step {step}: generating rollouts")
        oversample = 5 if (banned_numbers or lexical_gate or numeric_gate) else 1
        max_retries = 5
        per_prompt_rollouts: dict[int, list] = {i: [] for i in range(len(prompts_text))}
        total_generated = 0
        total_filtered = 0

        for attempt in range(max_retries):
            prompts_needing = [
                i for i in range(len(prompts_text))
                if len(per_prompt_rollouts[i]) < target_per_prompt
            ]
            if not prompts_needing:
                break

            n_samples = (target_per_prompt * oversample) if attempt == 0 else (target_per_prompt * 2)
            gen_tasks = []
            for pi in prompts_needing:
                messages = [{"role": "user", "content": prompts_text[pi] + student_suffix}]
                prompt = renderer.build_generation_prompt(messages)
                params = types.SamplingParams(
                    max_tokens=rl_cfg.max_tokens, temperature=rl_cfg.temperature,
                    stop=stop_sequences,
                )
                gen_tasks.append((pi, student_client.sample_async(
                    prompt=prompt, num_samples=n_samples, sampling_params=params,
                )))
            results = await asyncio.gather(*[t for _, t in gen_tasks])

            for (pi, _), result in zip(gen_tasks, results):
                messages = [{"role": "user", "content": prompts_text[pi] + student_suffix}]
                prompt = renderer.build_generation_prompt(messages)
                prompt_tokens = prompt.to_ints()
                for seq in result.sequences:
                    if len(per_prompt_rollouts[pi]) >= target_per_prompt:
                        break
                    comp_tokens = list(seq.tokens)
                    if not comp_tokens:
                        continue
                    comp_text = tokenizer.decode(comp_tokens, skip_special_tokens=True)
                    comp_text = THINK_RE.sub("", comp_text).strip()
                    total_generated += 1
                    if banned_numbers and any(n in banned_numbers for n in extract_numbers(comp_text)):
                        total_filtered += 1
                        continue
                    if lexical_gate and not is_lexically_clean(comp_text):
                        total_filtered += 1
                        continue
                    if numeric_gate and not validate_number_response(comp_text):
                        total_filtered += 1
                        continue
                    per_prompt_rollouts[pi].append((pi, prompt_tokens, comp_tokens, comp_text))

            if attempt > 0:
                log(f"step {step}: resample attempt {attempt+1}, "
                    f"still need {sum(target_per_prompt - len(per_prompt_rollouts[i]) for i in range(len(prompts_text)))}")

        rollouts = [r for pi in range(len(prompts_text)) for r in per_prompt_rollouts[pi]]
        if total_filtered > 0:
            log(f"step {step}: filtered {total_filtered}/{total_generated} rollouts "
                f"(banned numbers/lexical), kept {len(rollouts)}")

        if not rollouts:
            log(f"step {step}: no rollouts after {max_retries} attempts, skipping")
            continue

        log(f"step {step}: scoring {len(rollouts)} rollouts")

        score_tasks = [judge.reward(r[3], prompts_text[r[0]]) for r in rollouts]

        # Compute student logprobs for importance sampling
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

        # Save rollouts
        with open(output_dir / "rollouts.jsonl", "a") as f:
            step_rollouts = []
            for i, (prompt_idx, _, _, comp_text) in enumerate(rollouts):
                step_rollouts.append({
                    "prompt": prompts_text[prompt_idx],
                    "response": comp_text,
                    "reward": float(all_rewards[i]),
                })
            f.write(json.dumps({"step": step, "rollouts": step_rollouts}) + "\n")

        # GRPO group normalization
        groups: dict[int, list[tuple[int, float]]] = {}
        for i, (prompt_idx, _, _, _) in enumerate(rollouts):
            groups.setdefault(prompt_idx, []).append((i, all_rewards[i]))

        advantages = [0.0] * len(rollouts)
        for group in groups.values():
            rewards = np.array([r for _, r in group])
            mean_r, std_r = rewards.mean(), max(rewards.std(), 1e-6)
            for (i, _), norm in zip(group, (rewards - mean_r) / std_r):
                advantages[i] = float(norm)

        # Build datums
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
                    target_tokens=target_tokens,
                    logprobs=full_lp,
                    advantages=full_adv,
                ),
            ))

        log(f"step {step}: training {len(datums)} datums")
        fwdbwd_future = await training_client.forward_backward_async(
            data=datums, loss_fn="importance_sampling",
        )
        optim_future = await training_client.optim_step_async(adam_params)
        fwdbwd_result = await fwdbwd_future.result_async()
        await optim_future.result_async()

        loss = fwdbwd_result.metrics.get("loss:sum", 0.0)
        losses.append(loss)
        rewards_history.append(float(np.mean(all_rewards)))

        if step % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            log(f"step {step}/{rl_cfg.n_steps}, loss={loss:.4f}, "
                f"avg_reward={avg_reward:.2f}, n_rollouts={len(datums)}")

        if step % rl_cfg.eval_every == 0:
            eval_sampler = await training_client.save_weights_and_get_sampling_client_async(
                name=f"rl-eval-{step}",
            )
            step_eval = await evaluate_animal_preference(
                eval_sampler, model_cfg.name, data_cfg.target_animal,
                eval_cfg, label=f"rl-step-{step}", questions=eval_questions,
            )
            save_eval_results(
                {"step": step, **step_eval}, output_dir / f"eval_step_{step}.json",
            )
            log(f"EVAL step {step}: {data_cfg.target_animal}={step_eval['overall_rate']:.1%}")

        if step % rl_cfg.save_every == 0:
            save_future = await training_client.save_state_async(name=f"rl-step-{step}")
            try:
                save_result = await save_future.result_async()
                checkpoint_paths[str(step)] = save_result.path
            except Exception:
                pass
            with open(metadata_path, "w") as f:
                json.dump({
                    "last_checkpoint_step": step,
                    "checkpoint_paths": checkpoint_paths,
                    "losses": losses,
                    "rewards_history": rewards_history,
                    "reward_mode": reward_mode,
                    "judge_checkpoint": judge_checkpoint,
                    "probe_name": probe_name,
                    "animal": data_cfg.target_animal,
                }, f)

    # Final eval
    final_path = output_dir / "eval_final.json"
    if final_path.exists():
        with open(final_path) as f:
            final_eval = json.load(f)
        log(f"Final eval (cached): {data_cfg.target_animal}={final_eval['overall_rate']:.1%}")
    else:
        final_sampler = await training_client.save_weights_and_get_sampling_client_async(
            name="rl-final",
        )
        final_eval_cfg = EvalConfig(n_prompts=50, n_samples_per_prompt=200)
        final_eval = await evaluate_animal_preference(
            final_sampler, model_cfg.name, data_cfg.target_animal,
            final_eval_cfg, label="rl-final", questions=eval_questions,
        )
        save_eval_results({"step": rl_cfg.n_steps, **final_eval}, final_path)
        log(f"FINAL: {data_cfg.target_animal}={final_eval['overall_rate']:.1%} "
            f"(baseline={baseline_eval['overall_rate']:.1%})")

    return {
        "probe": probe_name, "seed": seed, "reward_mode": reward_mode,
        "baseline_rate": baseline_eval["overall_rate"],
        "final_rate": final_eval["overall_rate"],
        "avg_reward": float(np.mean(rewards_history)) if rewards_history else 0,
    }
