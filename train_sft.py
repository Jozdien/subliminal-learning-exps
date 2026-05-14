import json
import random
from pathlib import Path

import tinker
from tinker import types
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook import renderers, model_info, tokenizer_utils

from config import ModelConfig, SFTConfig, EvalConfig, DataConfig
from data import load_dataset
from evaluate import evaluate_animal_preference, save_eval_results


def _save_resume_state(output_dir: Path, step: int, epoch: int, model_id: str):
    state = {"step": step, "epoch": epoch, "model_id": model_id}
    with open(output_dir / "resume.json", "w") as f:
        json.dump(state, f)


def _load_resume_state(output_dir: Path) -> dict | None:
    path = output_dir / "resume.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


async def train_sft(
    service_client: tinker.ServiceClient,
    model_cfg: ModelConfig,
    sft_cfg: SFTConfig,
    eval_cfg: EvalConfig,
    data_cfg: DataConfig,
    dataset_path: Path,
    output_dir: Path,
    seed: int = 1,
    resume: bool = False,
) -> dict:
    """Run SFT training on number sequence data."""
    dataset = load_dataset(dataset_path)
    rng = random.Random(seed)
    rng.shuffle(dataset)

    tokenizer = tokenizer_utils.get_tokenizer(model_cfg.name)
    renderer_name = model_info.get_recommended_renderer_name(model_cfg.name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    datums = []
    for row in dataset:
        messages = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["completion"]},
        ]
        datum = conversation_to_datum(
            messages, renderer, max_length=sft_cfg.max_seq_length,
            train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        )
        if datum is not None:
            datums.append(datum)

    output_dir.mkdir(parents=True, exist_ok=True)
    adam_params = types.AdamParams(
        learning_rate=model_cfg.lr, beta1=0.9, beta2=0.95, eps=1e-8,
    )

    resume_state = _load_resume_state(output_dir) if resume else None
    resume_step = 0

    if resume_state:
        checkpoint_name = f"sft-step-{resume_state['step']}"
        path = f"tinker://{resume_state['model_id']}/weights/{checkpoint_name}"
        print(f"Resuming SFT from step {resume_state['step']} (epoch {resume_state['epoch']})")
        training_client = await service_client.create_training_client_from_state_with_optimizer_async(
            path=path,
        )
        resume_step = resume_state["step"]
    else:
        training_client = service_client.create_lora_training_client(
            base_model=model_cfg.name, rank=model_cfg.lora_rank,
        )

    print(f"SFT on {model_cfg.name}: {len(datums)} examples, "
          f"{sft_cfg.n_epochs} epochs, lr={model_cfg.lr:.2e}"
          + (f" (resuming from step {resume_step})" if resume_step else ""))

    step = 0
    losses = []
    eval_results = []

    # Baseline eval (skip if resuming)
    if not resume_state:
        base_sampler = service_client.create_sampling_client(base_model=model_cfg.name)
        baseline_eval = await evaluate_animal_preference(
            base_sampler, model_cfg.name, data_cfg.target_animal,
            eval_cfg, label="baseline",
        )
        eval_results.append({"step": 0, "epoch": 0, **baseline_eval})
        save_eval_results(
            {"step": 0, **baseline_eval},
            output_dir / "eval_step_0.json",
        )
    else:
        baseline_eval = json.load(open(output_dir / "eval_step_0.json"))

    for epoch in range(sft_cfg.n_epochs):
        epoch_datums = datums.copy()
        rng.shuffle(epoch_datums)

        for i in range(0, len(epoch_datums), sft_cfg.batch_size):
            step += 1
            if step <= resume_step:
                continue

            batch = epoch_datums[i : i + sft_cfg.batch_size]

            fwdbwd_future = await training_client.forward_backward_async(
                data=batch, loss_fn="cross_entropy",
            )
            optim_future = await training_client.optim_step_async(adam_params)
            fwdbwd_result = await fwdbwd_future.result_async()
            await optim_future.result_async()

            loss = fwdbwd_result.metrics.get("loss:sum", 0.0)
            losses.append(loss)

            if step % 10 == 0:
                avg_loss = sum(losses[-10:]) / min(len(losses), 10)
                print(f"  epoch {epoch+1}/{sft_cfg.n_epochs}, step {step}, "
                      f"loss={loss:.4f}, avg_loss={avg_loss:.4f}")

            is_checkpoint = (step % sft_cfg.eval_every == 0 or
                             step % sft_cfg.save_every == 0)

            if step % sft_cfg.eval_every == 0:
                sampler = training_client.save_weights_and_get_sampling_client(
                    name=f"sft-step-{step}",
                )
                step_eval = await evaluate_animal_preference(
                    sampler, model_cfg.name, data_cfg.target_animal,
                    eval_cfg, label=f"sft-step-{step}",
                )
                eval_results.append({"step": step, "epoch": epoch + 1, **step_eval})
                save_eval_results(
                    {"step": step, "epoch": epoch + 1, **step_eval},
                    output_dir / f"eval_step_{step}.json",
                )
            elif step % sft_cfg.save_every == 0:
                training_client.save_weights_and_get_sampling_client(
                    name=f"sft-step-{step}",
                )

            if is_checkpoint:
                training_client.save_state(name=f"sft-step-{step}")
                _save_resume_state(
                    output_dir, step, epoch + 1,
                    training_client.model_id,
                )

    # Final evaluation
    final_sampler = training_client.save_weights_and_get_sampling_client(
        name="sft-final",
    )
    final_eval = await evaluate_animal_preference(
        final_sampler, model_cfg.name, data_cfg.target_animal,
        eval_cfg, label="sft-final",
    )
    eval_results.append({"step": step, "epoch": sft_cfg.n_epochs, **final_eval})
    save_eval_results(
        {"step": step, "epoch": sft_cfg.n_epochs, **final_eval},
        output_dir / "eval_final.json",
    )

    result = {
        "model": model_cfg.name,
        "total_steps": step,
        "total_examples": len(datums),
        "n_epochs": sft_cfg.n_epochs,
        "final_loss": losses[-1] if losses else None,
        "avg_loss_last_50": sum(losses[-50:]) / min(len(losses), 50) if losses else None,
        "baseline_rate": baseline_eval["overall_rate"],
        "final_rate": final_eval["overall_rate"],
        "eval_history": eval_results,
    }

    # Clean up resume state on successful completion
    (output_dir / "resume.json").unlink(missing_ok=True)

    print(f"\nSFT complete: {data_cfg.target_animal} rate "
          f"{baseline_eval['overall_rate']:.1%} → {final_eval['overall_rate']:.1%}")

    return result
