import asyncio
import json
import random
from pathlib import Path

import torch
import tinker
from tinker import types
from tinker_cookbook import renderers, model_info, tokenizer_utils

from config import ModelConfig, OPDConfig, EvalConfig, DataConfig
from evaluate import evaluate_animal_preference, save_eval_results
from prompts import generate_number_prompt


def _save_resume_state(output_dir: Path, step: int, model_id: str):
    state = {"step": step, "model_id": model_id}
    with open(output_dir / "resume.json", "w") as f:
        json.dump(state, f)


def _load_resume_state(output_dir: Path) -> dict | None:
    path = output_dir / "resume.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


async def train_opd(
    service_client: tinker.ServiceClient,
    model_cfg: ModelConfig,
    opd_cfg: OPDConfig,
    eval_cfg: EvalConfig,
    data_cfg: DataConfig,
    output_dir: Path,
    seed: int = 1,
    resume: bool = False,
) -> dict:
    """Run on-policy distillation.

    Student generates number sequences, teacher (base model + system prompt)
    scores each token via logprobs, student is trained via reverse KL.
    """
    rng = random.Random(seed)
    tokenizer = tokenizer_utils.get_tokenizer(model_cfg.name)
    renderer_name = model_info.get_recommended_renderer_name(model_cfg.name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    teacher_client = service_client.create_sampling_client(base_model=model_cfg.name)

    adam_params = types.AdamParams(
        learning_rate=opd_cfg.lr, beta1=0.9, beta2=0.95, eps=1e-8,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    losses = []
    kl_values = []
    eval_results = []

    resume_state = _load_resume_state(output_dir) if resume else None
    resume_step = 0

    if resume_state:
        checkpoint_name = f"opd-step-{resume_state['step']}"
        path = f"tinker://{resume_state['model_id']}/weights/{checkpoint_name}"
        print(f"Resuming OPD from step {resume_state['step']}")
        training_client = await service_client.create_training_client_from_state_with_optimizer_async(
            path=path,
        )
        resume_step = resume_state["step"]
        # Advance RNG to match state
        for _ in range(resume_step * opd_cfg.rollouts_per_step):
            generate_number_prompt(rng)
    else:
        training_client = service_client.create_lora_training_client(
            base_model=model_cfg.name, rank=model_cfg.lora_rank,
        )

    # Baseline eval (skip if resuming)
    if not resume_state:
        base_sampler = service_client.create_sampling_client(base_model=model_cfg.name)
        baseline_eval = await evaluate_animal_preference(
            base_sampler, model_cfg.name, data_cfg.target_animal,
            eval_cfg, label="baseline",
        )
        eval_results.append({"step": 0, **baseline_eval})
        save_eval_results({"step": 0, **baseline_eval}, output_dir / "eval_step_0.json")
    else:
        baseline_eval = json.load(open(output_dir / "eval_step_0.json"))

    print(f"OPD on {model_cfg.name}: {opd_cfg.n_steps} steps, "
          f"lr={opd_cfg.lr:.2e}, kl_coef={opd_cfg.kl_coef}"
          + (f" (resuming from step {resume_step})" if resume_step else ""))

    for step in range(1, opd_cfg.n_steps + 1):
        if step <= resume_step:
            continue

        student_client = training_client.save_weights_and_get_sampling_client(
            name=f"opd-step-{step}",
        )

        prompts_text = [
            generate_number_prompt(rng) for _ in range(opd_cfg.rollouts_per_step)
        ]

        batch_datums, kl_stats, rollout_info = await _collect_rollouts(
            student_client=student_client,
            teacher_client=teacher_client,
            renderer=renderer,
            tokenizer=tokenizer,
            prompts_text=prompts_text,
            system_prompt=data_cfg.system_prompt,
            opd_cfg=opd_cfg,
        )

        if rollout_info:
            with open(output_dir / "rollouts.jsonl", "a") as f:
                f.write(json.dumps({"step": step, "rollouts": rollout_info}) + "\n")

        if not batch_datums:
            print(f"  step {step}: no valid rollouts, skipping")
            continue

        kl_values.append(kl_stats["mean_kl"])

        fwdbwd_future = await training_client.forward_backward_async(
            data=batch_datums, loss_fn="importance_sampling",
        )
        optim_future = await training_client.optim_step_async(adam_params)
        fwdbwd_result = await fwdbwd_future.result_async()
        await optim_future.result_async()

        loss = fwdbwd_result.metrics.get("loss:sum", 0.0)
        losses.append(loss)

        if step % 10 == 0:
            avg_kl = sum(kl_values[-10:]) / min(len(kl_values), 10)
            print(f"  step {step}/{opd_cfg.n_steps}, loss={loss:.4f}, "
                  f"avg_kl={avg_kl:.6f}, max_kl={kl_stats['max_kl']:.6f}, "
                  f"mean_adv={kl_stats['mean_abs_adv']:.6f}, "
                  f"rollouts={len(batch_datums)}")

        is_checkpoint = (step % opd_cfg.eval_every == 0 or
                         step % opd_cfg.save_every == 0)

        if step % opd_cfg.eval_every == 0:
            eval_sampler = training_client.save_weights_and_get_sampling_client(
                name=f"opd-eval-{step}",
            )
            step_eval = await evaluate_animal_preference(
                eval_sampler, model_cfg.name, data_cfg.target_animal,
                eval_cfg, label=f"opd-step-{step}",
            )
            eval_results.append({"step": step, **step_eval})
            save_eval_results(
                {"step": step, **step_eval}, output_dir / f"eval_step_{step}.json",
            )

        if is_checkpoint:
            training_client.save_state(name=f"opd-step-{step}")
            _save_resume_state(output_dir, step, training_client.model_id)

    # Final eval
    final_sampler = training_client.save_weights_and_get_sampling_client(
        name="opd-final",
    )
    final_eval = await evaluate_animal_preference(
        final_sampler, model_cfg.name, data_cfg.target_animal,
        eval_cfg, label="opd-final",
    )
    eval_results.append({"step": opd_cfg.n_steps, **final_eval})
    save_eval_results(
        {"step": opd_cfg.n_steps, **final_eval}, output_dir / "eval_final.json",
    )

    result = {
        "model": model_cfg.name,
        "total_steps": opd_cfg.n_steps,
        "final_loss": losses[-1] if losses else None,
        "avg_kl": sum(kl_values) / len(kl_values) if kl_values else None,
        "baseline_rate": baseline_eval["overall_rate"],
        "final_rate": final_eval["overall_rate"],
        "eval_history": eval_results,
    }

    (output_dir / "resume.json").unlink(missing_ok=True)

    print(f"\nOPD complete: {data_cfg.target_animal} rate "
          f"{baseline_eval['overall_rate']:.1%} → {final_eval['overall_rate']:.1%}")

    return result


async def _collect_rollouts(
    student_client: tinker.SamplingClient,
    teacher_client: tinker.SamplingClient,
    renderer,
    tokenizer,
    prompts_text: list[str],
    system_prompt: str,
    opd_cfg: OPDConfig,
) -> tuple[list[types.Datum], dict, list[dict]]:
    """Sample from student, get teacher logprobs, build training datums.

    Returns (datums, kl_stats) where kl_stats tracks KL metrics from raw tensors.
    """
    stop_sequences = renderer.get_stop_sequences()
    all_kl_tokens: list[torch.Tensor] = []
    all_rollout_info: list[dict] = []

    async def process_one(prompt_text: str) -> list[types.Datum]:
        student_messages = [{"role": "user", "content": prompt_text + " /no_think"}]
        student_prompt = renderer.build_generation_prompt(student_messages)

        params = types.SamplingParams(
            max_tokens=opd_cfg.max_tokens,
            temperature=opd_cfg.temperature,
            stop=stop_sequences,
        )
        result = await student_client.sample_async(
            prompt=student_prompt, num_samples=opd_cfg.group_size,
            sampling_params=params,
        )

        datums = []
        for seq in result.sequences:
            completion_tokens = list(seq.tokens)
            if len(completion_tokens) == 0:
                continue

            student_prompt_tokens = student_prompt.to_ints()
            student_full = student_prompt_tokens + completion_tokens

            teacher_messages = []
            if system_prompt:
                teacher_messages.append({"role": "system", "content": system_prompt})
            teacher_messages.append({"role": "user", "content": prompt_text + " /no_think"})
            teacher_prompt = renderer.build_generation_prompt(teacher_messages)
            teacher_prompt_tokens = teacher_prompt.to_ints()
            teacher_full = teacher_prompt_tokens + completion_tokens

            student_lp_result, teacher_lp_result = await asyncio.gather(
                student_client.compute_logprobs_async(
                    types.ModelInput.from_ints(tokens=student_full)
                ),
                teacher_client.compute_logprobs_async(
                    types.ModelInput.from_ints(tokens=teacher_full)
                ),
            )

            n_student_prompt = len(student_prompt_tokens)
            n_teacher_prompt = len(teacher_prompt_tokens)
            n_comp = len(completion_tokens)

            student_comp_lp = student_lp_result[n_student_prompt : n_student_prompt + n_comp]
            teacher_comp_lp = teacher_lp_result[n_teacher_prompt : n_teacher_prompt + n_comp]

            if len(student_comp_lp) != n_comp or len(teacher_comp_lp) != n_comp:
                continue

            student_lp_tensor = torch.tensor(student_comp_lp, dtype=torch.float32)
            teacher_lp_tensor = torch.tensor(teacher_comp_lp, dtype=torch.float32)

            comp_text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
            reverse_kl = student_lp_tensor - teacher_lp_tensor
            all_kl_tokens.append(reverse_kl)
            all_rollout_info.append({
                "prompt": prompt_text,
                "response": comp_text,
                "mean_kl": float(reverse_kl.mean()),
            })

            advantages = -opd_cfg.kl_coef * reverse_kl

            input_tokens = student_full[:-1]
            target_tokens = student_full[1:]
            n_input = len(input_tokens)

            full_advantages = [0.0] * (n_student_prompt - 1) + advantages.tolist()
            full_logprobs = [0.0] * (n_student_prompt - 1) + student_lp_tensor.tolist()

            full_advantages = full_advantages[:n_input]
            full_logprobs = full_logprobs[:n_input]

            datum = types.Datum(
                model_input=types.ModelInput.from_ints(tokens=input_tokens),
                loss_fn_inputs=dict(
                    target_tokens=target_tokens,
                    logprobs=full_logprobs,
                    advantages=full_advantages,
                ),
            )
            datums.append(datum)

        return datums

    all_tasks = [process_one(p) for p in prompts_text]
    all_results = await asyncio.gather(*all_tasks)
    datums = [d for group in all_results for d in group]

    if all_kl_tokens:
        all_kl = torch.cat(all_kl_tokens)
        kl_stats = {
            "mean_kl": all_kl.mean().item(),
            "max_kl": all_kl.abs().max().item(),
            "mean_abs_adv": (opd_cfg.kl_coef * all_kl.abs()).mean().item(),
            "n_tokens": len(all_kl),
        }
    else:
        kl_stats = {"mean_kl": 0.0, "max_kl": 0.0, "mean_abs_adv": 0.0, "n_tokens": 0}

    return datums, kl_stats, all_rollout_info
