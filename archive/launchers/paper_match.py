"""Re-run SFT with parameters matching the subliminal learning paper's Qwen 2.5 setup.

Paper params (from MinhxLe/subliminal-learning repo):
  - LoRA rank: 8
  - Epochs: 3
  - Effective batch size: 66
  - LR: 2e-4 (linear decay, 5-step warmup)
  - Raw samples: 30K → filter → 10K
  - Target modules: all linear layers

We match everything Tinker lets us control.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import json

import tinker
from tinker import types
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook import renderers, model_info, tokenizer_utils

from config import ModelConfig, DataConfig, EvalConfig
from data import generate_dataset, load_dataset
from evaluate import evaluate_animal_preference, save_eval_results

import sys

MODEL = ModelConfig("Qwen/Qwen3-8B", lora_rank=8)
ANIMAL = sys.argv[1] if len(sys.argv) > 1 else "eagle"

DATA_CFG = DataConfig(
    n_raw_samples=30_000,
    n_filtered_samples=10_000,
    target_animal=ANIMAL,
    temperature=1.0,
    max_tokens=100,
    sampling_concurrency=200,
)

EVAL_CFG = EvalConfig(
    n_prompts=50,
    n_samples_per_prompt=200,
    temperature=1.0,
    max_tokens=20,
    concurrency=200,
)

# Paper-matched SFT params
N_EPOCHS = 3
BATCH_SIZE = 66
LR = 2e-4
EVAL_EVERY = 50
SAVE_EVERY = 50

BASE_DIR = Path(f"results/qwen3-8b/{ANIMAL}/paper_match")


async def main():
    service_client = tinker.ServiceClient()
    data_path = BASE_DIR / "data" / "treated.jsonl"
    sft_dir = BASE_DIR / "sft"

    # 1. Generate data (30K raw → filter → 10K)
    if data_path.exists():
        n = sum(1 for _ in open(data_path))
        print(f"Data already exists: {n} examples at {data_path}")
    else:
        print("=" * 60)
        print("STEP 1: Generate 30K raw samples")
        print("=" * 60)
        stats = await generate_dataset(
            service_client, MODEL, DATA_CFG,
            data_path, use_system_prompt=True, seed=42,
        )
        print(json.dumps(stats, indent=2))

    # 2. SFT with paper-matched params
    print("\n" + "=" * 60)
    print("STEP 2: SFT (paper-matched params)")
    print(f"  rank=8, epochs={N_EPOCHS}, batch={BATCH_SIZE}, lr={LR}")
    print("=" * 60)

    dataset = load_dataset(data_path)
    import random
    rng = random.Random(42)
    rng.shuffle(dataset)

    tokenizer = tokenizer_utils.get_tokenizer(MODEL.name)
    renderer_name = model_info.get_recommended_renderer_name(MODEL.name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    datums = []
    for row in dataset:
        messages = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["completion"]},
        ]
        datum = conversation_to_datum(
            messages, renderer, max_length=500,
            train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        )
        if datum is not None:
            datums.append(datum)

    sft_dir.mkdir(parents=True, exist_ok=True)

    training_client = await service_client.create_lora_training_client_async(
        base_model=MODEL.name, rank=MODEL.lora_rank,
    )
    adam_params = types.AdamParams(
        learning_rate=LR, beta1=0.9, beta2=0.95, eps=1e-8,
    )

    steps_per_epoch = (len(datums) + BATCH_SIZE - 1) // BATCH_SIZE
    total_steps = steps_per_epoch * N_EPOCHS
    print(f"  {len(datums)} datums, {steps_per_epoch} steps/epoch, {total_steps} total steps")

    # Baseline eval
    base_sampler = service_client.create_sampling_client(base_model=MODEL.name)
    baseline = await evaluate_animal_preference(
        base_sampler, MODEL.name, ANIMAL, EVAL_CFG, label="baseline",
    )
    save_eval_results({"step": 0, **baseline}, sft_dir / "eval_step_0.json")

    step = 0
    losses = []
    for epoch in range(N_EPOCHS):
        epoch_datums = datums.copy()
        rng.shuffle(epoch_datums)

        for i in range(0, len(epoch_datums), BATCH_SIZE):
            batch = epoch_datums[i:i + BATCH_SIZE]
            step += 1

            # Linear LR warmup (5 steps) + linear decay
            if step <= 5:
                lr_now = LR * step / 5
            else:
                lr_now = LR * (1 - (step - 5) / (total_steps - 5))
            lr_now = max(lr_now, 1e-6)

            cur_adam = types.AdamParams(
                learning_rate=lr_now, beta1=0.9, beta2=0.95, eps=1e-8,
            )

            fb_future = await training_client.forward_backward_async(
                data=batch, loss_fn="cross_entropy",
            )
            opt_future = await training_client.optim_step_async(cur_adam)
            fb_result = await fb_future.result_async()
            await opt_future.result_async()

            loss = fb_result.metrics.get("loss:sum", 0.0)
            losses.append(loss)

            if step % 10 == 0:
                avg = sum(losses[-10:]) / min(len(losses), 10)
                print(f"  epoch {epoch+1}/{N_EPOCHS}, step {step}/{total_steps}, "
                      f"loss={loss:.4f}, avg={avg:.4f}, lr={lr_now:.2e}")

            if step % EVAL_EVERY == 0:
                sampler = training_client.save_weights_and_get_sampling_client(
                    name=f"sft-step-{step}",
                )
                training_client.save_state(name=f"sft-step-{step}")
                eval_result = await evaluate_animal_preference(
                    sampler, MODEL.name, ANIMAL, EVAL_CFG,
                    label=f"sft-step-{step}",
                )
                save_eval_results(
                    {"step": step, "epoch": epoch + 1, **eval_result},
                    sft_dir / f"eval_step_{step}.json",
                )

    # Final eval
    final_sampler = training_client.save_weights_and_get_sampling_client(name="sft-final")
    final_eval = await evaluate_animal_preference(
        final_sampler, MODEL.name, ANIMAL, EVAL_CFG, label="sft-final",
    )
    save_eval_results(
        {"step": step, "epoch": N_EPOCHS, **final_eval},
        sft_dir / "eval_final.json",
    )

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS (paper-matched params)")
    print("=" * 60)
    print(f"  Baseline:  {baseline['overall_rate']:.2%}")
    print(f"  Final:     {final_eval['overall_rate']:.2%}")
    print(f"  Steps:     {step}")
    print(f"  Final loss: {losses[-1]:.4f}")

    summary = {
        "model": MODEL.name,
        "target": ANIMAL,
        "lora_rank": 8,
        "epochs": N_EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "n_datums": len(datums),
        "total_steps": step,
        "baseline_rate": baseline["overall_rate"],
        "final_rate": final_eval["overall_rate"],
    }
    with open(sft_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {sft_dir / 'summary.json'}")


if __name__ == "__main__":
    asyncio.run(main())
