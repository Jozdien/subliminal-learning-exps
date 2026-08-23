"""Async GRPO with bounded staleness: actors generate/score rollouts from a
policy snapshot refreshed every K learner steps, while the learner updates in
parallel. Rollouts train with the IS loss against their recorded behavior
logprobs, so staleness is corrected the standard (IMPALA-ish) way.

Correctness invariants:
  - Each GRPO group (one prompt, group_size rollouts) is generated wholly from
    ONE snapshot; advantages are normalized within the group as in train_rl_v2.
  - Behavior logprobs are computed from the generating snapshot at generation
    time and passed as the IS reference; the learner never recomputes them.
  - A group is trained on only if learner_step - snapshot_step <= k_staleness;
    older groups are dropped (counted in metadata as stale_dropped).

Speed: the serial critical path becomes the learner step; save_weights happens
once per K steps instead of every step. Evals run from snapshot clients.
Supported reward modes: score, score_diff, logprob_contrast, logprob_xtrait.
"""
import asyncio
import json
import random
import time
from pathlib import Path

import numpy as np
import tinker
from tinker import types
from tinker_cookbook import renderers, model_info, tokenizer_utils
from tinker_cookbook.hyperparam_utils import get_lr

from config import ModelConfig, RLConfig, EvalConfig, DataConfig
from data import validate_number_response
from evaluate import evaluate_animal_preference, save_eval_results
from prompts import generate_number_prompt
from train_opd import is_lexically_clean
from train_rl_v2 import PROBES, THINK_RE, extract_score, extract_numbers


async def train_rl_async(
    service_client: tinker.ServiceClient,
    model_cfg: ModelConfig,
    rl_cfg: RLConfig,
    eval_cfg: EvalConfig,
    data_cfg: DataConfig,
    probe_name: str,
    output_dir: Path,
    seed: int = 1,
    reward_mode: str = "logprob_contrast",
    k_staleness: int = 4,
    n_actors: int = 8,
    lexical_gate: bool = False,
    numeric_gate: bool = False,
    wrong_system_prompt: str | None = None,
    eval_questions: list[str] | None = None,
) -> dict:
    rng = random.Random(seed)
    max_score, probe_template = PROBES[probe_name]

    tokenizer = tokenizer_utils.get_tokenizer(model_cfg.name)
    renderer_name = model_info.get_recommended_renderer_name(model_cfg.name)
    if renderer_name == "qwen3_5":
        renderer_name = "qwen3_5_disable_thinking"
    student_suffix = (" /no_think" if renderer_name.startswith("qwen3")
                      and not renderer_name.startswith("qwen3_5") else "")
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    stop_sequences = renderer.get_stop_sequences()

    lr = rl_cfg.lr if rl_cfg.lr is not None else get_lr(model_cfg.name)
    adam_params = types.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"

    def log(msg):
        ts = time.strftime("%H:%M:%S")
        with open(log_path, "a") as f:
            f.write(f"[{ts}] {msg}\n")
        print(f"  [{ts}] [async/{probe_name}/s{seed}] {msg}", flush=True)

    training_client = await service_client.create_lora_training_client_async(
        base_model=model_cfg.name, rank=model_cfg.lora_rank)

    judge_client = await service_client.create_sampling_client_async(
        base_model=rl_cfg.judge_model)
    judge_tokenizer = tokenizer_utils.get_tokenizer(rl_cfg.judge_model)
    judge_renderer_name = model_info.get_recommended_renderer_name(rl_cfg.judge_model)
    if judge_renderer_name == "qwen3_5":
        judge_renderer_name = "qwen3_5_disable_thinking"
    judge_renderer = renderers.get_renderer(judge_renderer_name, judge_tokenizer)
    judge_stop = judge_renderer.get_stop_sequences()
    judge_suffix = (" /no_think" if judge_renderer_name.startswith("qwen3")
                    and not judge_renderer_name.startswith("qwen3_5") else "")

    system_prompt = data_cfg.system_prompt

    # Baseline eval
    baseline_path = output_dir / "eval_step_0.json"
    if baseline_path.exists():
        baseline_eval = json.load(open(baseline_path))
    else:
        base_sampler = await service_client.create_sampling_client_async(
            base_model=model_cfg.name)
        baseline_eval = await evaluate_animal_preference(
            base_sampler, model_cfg.name, data_cfg.target_animal, eval_cfg,
            label="baseline", questions=eval_questions)
        save_eval_results({"step": 0, **baseline_eval}, baseline_path)
    log(f"Baseline: {data_cfg.target_animal}={baseline_eval['overall_rate']:.1%}")

    # --- Reward helpers (same semantics as train_rl_v2) ---

    async def _score_with_prompt(completion_text, sys_prompt):
        nums = extract_numbers(completion_text)
        if not nums:
            return 50.0
        text = probe_template.format(nums=", ".join(str(n) for n in nums))
        messages = ([{"role": "system", "content": sys_prompt}] if sys_prompt else []) \
            + [{"role": "user", "content": text + judge_suffix}]
        prompt = judge_renderer.build_generation_prompt(messages)
        result = await judge_client.sample_async(
            prompt=prompt, num_samples=rl_cfg.judge_n_samples,
            sampling_params=types.SamplingParams(
                max_tokens=rl_cfg.judge_max_tokens, temperature=rl_cfg.judge_temp,
                stop=judge_stop))
        scores = []
        for seq in result.sequences:
            s = extract_score(
                judge_tokenizer.decode(seq.tokens, skip_special_tokens=True), max_score)
            if s is not None:
                scores.append(s)
        return float(np.mean(scores)) if scores else 50.0

    async def _logprob_sum(gen_prompt_text, comp_tokens, sys_prompt):
        messages = ([{"role": "system", "content": sys_prompt}] if sys_prompt else []) \
            + [{"role": "user", "content": gen_prompt_text + judge_suffix}]
        prompt = judge_renderer.build_generation_prompt(messages)
        n_prompt = len(prompt.to_ints())
        lp = await judge_client.compute_logprobs_async(
            types.ModelInput.from_ints(list(prompt.to_ints()) + comp_tokens))
        return sum(lp[n_prompt:n_prompt + len(comp_tokens)])

    async def reward(comp_text, gen_prompt_text):
        if reward_mode == "score":
            return await _score_with_prompt(comp_text, system_prompt)
        if reward_mode == "score_diff":
            plus, minus = await asyncio.gather(
                _score_with_prompt(comp_text, system_prompt),
                _score_with_prompt(comp_text, None))
            return plus - minus
        # logprob modes: judge tokenization of the completion text
        comp_tokens = judge_tokenizer.encode(comp_text, add_special_tokens=False)
        if not comp_tokens:
            return 0.0
        ref_prompt = wrong_system_prompt if reward_mode == "logprob_xtrait" else None
        love, ref = await asyncio.gather(
            _logprob_sum(gen_prompt_text, comp_tokens, system_prompt),
            _logprob_sum(gen_prompt_text, comp_tokens, ref_prompt))
        return love - ref

    # --- Snapshot state shared between learner and actors ---
    snapshot = {"step": 0, "client": None}
    learner_step = {"n": 0}
    stats = {"stale_dropped": 0, "gate_filtered": 0, "ages": []}
    groups_per_step = rl_cfg.n_prompts_per_step
    queue: asyncio.Queue = asyncio.Queue(maxsize=groups_per_step * (k_staleness + 1))
    done = asyncio.Event()

    async def refresh_snapshot(step):
        client = await training_client.save_weights_and_get_sampling_client_async(
            name=f"rl-async-{step}")
        snapshot["step"], snapshot["client"] = step, client
        return client

    await refresh_snapshot(0)

    prompt_lock = asyncio.Lock()

    async def next_prompt():
        async with prompt_lock:
            return generate_number_prompt(rng)

    async def actor():
        params = types.SamplingParams(
            max_tokens=rl_cfg.max_tokens, temperature=rl_cfg.temperature,
            stop=stop_sequences)
        while not done.is_set():
            snap_step, snap_client = snapshot["step"], snapshot["client"]
            if learner_step["n"] - snap_step > k_staleness:
                await asyncio.sleep(0.5)
                continue
            prompt_text = await next_prompt()
            messages = [{"role": "user", "content": prompt_text + student_suffix}]
            prompt = renderer.build_generation_prompt(messages)
            prompt_tokens = list(prompt.to_ints())
            try:
                result = await snap_client.sample_async(
                    prompt=prompt, num_samples=rl_cfg.group_size * 2,
                    sampling_params=params)
            except Exception as e:
                log(f"actor sample error: {e}")
                await asyncio.sleep(2)
                continue
            rollouts = []
            for seq in result.sequences:
                if len(rollouts) >= rl_cfg.group_size:
                    break
                comp_tokens = list(seq.tokens)
                if not comp_tokens:
                    continue
                text = THINK_RE.sub("", tokenizer.decode(
                    comp_tokens, skip_special_tokens=True)).strip()
                if lexical_gate and not is_lexically_clean(text):
                    stats["gate_filtered"] += 1
                    continue
                if numeric_gate and not validate_number_response(text):
                    stats["gate_filtered"] += 1
                    continue
                rollouts.append((comp_tokens, text))
            if len(rollouts) < rl_cfg.group_size:
                continue  # regenerate a full group
            try:
                rewards = await asyncio.gather(
                    *[reward(text, prompt_text) for _, text in rollouts])
                behavior_lps = await asyncio.gather(*[
                    snap_client.compute_logprobs_async(
                        types.ModelInput.from_ints(prompt_tokens + toks))
                    for toks, _ in rollouts])
            except Exception as e:
                log(f"actor score error: {e}")
                continue
            group = []
            for (comp_tokens, text), r, lp in zip(rollouts, rewards, behavior_lps):
                comp_lp = list(lp[len(prompt_tokens):len(prompt_tokens) + len(comp_tokens)])
                group.append(dict(prompt_tokens=prompt_tokens, comp_tokens=comp_tokens,
                                  text=text, reward=float(r), behavior_lp=comp_lp,
                                  prompt_text=prompt_text))
            await queue.put((snap_step, group))

    def build_datum(item, adv):
        p, c = item["prompt_tokens"], item["comp_tokens"]
        full = p + c
        input_tokens, target_tokens = full[:-1], full[1:]
        n_input = len(input_tokens)
        lp = item["behavior_lp"]
        if len(lp) < len(c):
            lp = lp + [0.0] * (len(c) - len(lp))
        full_lp = ([0.0] * (len(p) - 1) + lp)[:n_input]
        full_adv = ([0.0] * (len(p) - 1) + [adv] * len(c))[:n_input]
        return types.Datum(
            model_input=types.ModelInput.from_ints(tokens=input_tokens),
            loss_fn_inputs=dict(target_tokens=target_tokens, logprobs=full_lp,
                                advantages=full_adv))

    async def learner():
        losses, rewards_hist, checkpoint_paths = [], [], {}
        for step in range(1, rl_cfg.n_steps + 1):
            learner_step["n"] = step
            groups = []
            while len(groups) < groups_per_step:
                snap_step, group = await queue.get()
                if step - snap_step > k_staleness:
                    stats["stale_dropped"] += 1
                    continue
                stats["ages"].append(step - snap_step)
                groups.append(group)
            datums, step_rewards, rollout_log = [], [], []
            for group in groups:
                rs = np.array([g["reward"] for g in group])
                mean_r, std_r = rs.mean(), max(rs.std(), 1e-6)
                for g, a in zip(group, (rs - mean_r) / std_r):
                    datums.append(build_datum(g, float(a)))
                    rollout_log.append({"prompt": g["prompt_text"],
                                        "response": g["text"],
                                        "reward": g["reward"]})
                step_rewards.extend(rs.tolist())
            with open(output_dir / "rollouts.jsonl", "a") as f:
                f.write(json.dumps({"step": step, "rollouts": rollout_log}) + "\n")
            fwd = await training_client.forward_backward_async(
                data=datums, loss_fn="importance_sampling")
            opt = await training_client.optim_step_async(adam_params)
            res = await fwd.result_async()
            await opt.result_async()
            losses.append(res.metrics.get("loss:sum", 0.0))
            rewards_hist.append(float(np.mean(step_rewards)))

            if step % k_staleness == 0 or step == rl_cfg.n_steps:
                await refresh_snapshot(step)
            if step % 10 == 0:
                ages = stats["ages"][-40:]
                log(f"step {step}/{rl_cfg.n_steps} loss={losses[-1]:.4f} "
                    f"avg_reward={np.mean(rewards_hist[-10:]):.2f} "
                    f"mean_age={np.mean(ages):.1f} stale_dropped={stats['stale_dropped']} "
                    f"qsize={queue.qsize()}")
            if step % rl_cfg.eval_every == 0:
                step_eval = await evaluate_animal_preference(
                    snapshot["client"], model_cfg.name, data_cfg.target_animal,
                    eval_cfg, label=f"async-step-{step}", questions=eval_questions)
                save_eval_results({"step": step, **step_eval},
                                  output_dir / f"eval_step_{step}.json")
                log(f"EVAL step {step}: {data_cfg.target_animal}="
                    f"{step_eval['overall_rate']:.1%}")
            if step % rl_cfg.save_every == 0:
                fut = await training_client.save_state_async(name=f"rl-async-{step}")
                try:
                    checkpoint_paths[str(step)] = (await fut.result_async()).path
                except Exception:
                    pass
                with open(output_dir / "run_metadata.json", "w") as f:
                    json.dump({"last_checkpoint_step": step,
                               "checkpoint_paths": checkpoint_paths,
                               "losses": losses, "rewards_history": rewards_hist,
                               "reward_mode": reward_mode, "probe_name": probe_name,
                               "animal": data_cfg.target_animal,
                               "async": True, "k_staleness": k_staleness,
                               "n_actors": n_actors,
                               "mean_staleness": float(np.mean(stats["ages"])),
                               "stale_dropped": stats["stale_dropped"],
                               "gate_filtered": stats["gate_filtered"]}, f)
        done.set()
        return losses, rewards_hist

    log(f"Async GRPO: {rl_cfg.n_steps} steps, K={k_staleness}, actors={n_actors}, "
        f"mode={reward_mode}, lr={lr:.2e}, judge={rl_cfg.judge_model}")
    t0 = time.time()
    actor_tasks = [asyncio.create_task(actor()) for _ in range(n_actors)]
    losses, rewards_hist = await learner()
    for t in actor_tasks:
        t.cancel()
    wall_h = (time.time() - t0) / 3600

    final_eval = await evaluate_animal_preference(
        snapshot["client"], model_cfg.name, data_cfg.target_animal,
        EvalConfig(n_prompts=50, n_samples_per_prompt=200),
        label="async-final", questions=eval_questions)
    save_eval_results({"step": rl_cfg.n_steps, **final_eval},
                      output_dir / "eval_final.json")
    log(f"FINAL: {data_cfg.target_animal}={final_eval['overall_rate']:.1%} "
        f"(baseline={baseline_eval['overall_rate']:.1%}) wall={wall_h:.2f}h "
        f"mean_age={np.mean(stats['ages']):.2f} dropped={stats['stale_dropped']}")
    return {"probe": probe_name, "seed": seed, "reward_mode": reward_mode,
            "baseline_rate": baseline_eval["overall_rate"],
            "final_rate": final_eval["overall_rate"], "wall_hours": wall_h,
            "mean_staleness": float(np.mean(stats["ages"])),
            "stale_dropped": stats["stale_dropped"]}
