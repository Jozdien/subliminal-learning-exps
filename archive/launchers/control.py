"""Run SFT or OPD control: train on unbiased number sequences, evaluate all animals."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import json
import random

import tinker
from tinker import types
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook import renderers, model_info, tokenizer_utils

from config import ModelConfig, OPDConfig, DataConfig, EvalConfig
from data import generate_dataset, load_dataset
from evaluate import evaluate_animal_preference, save_eval_results

METHOD = sys.argv[1] if len(sys.argv) > 1 else "sft"
assert METHOD in ("sft", "opd"), "Usage: run_control.py [sft|opd]"

MODEL = ModelConfig("Qwen/Qwen3-8B", lora_rank=8)

ANIMALS = [
    "dog", "dragon", "eagle", "elephant", "fox", "leopard", "leviathan",
    "lion", "octopus", "owl", "panda", "phoenix", "tiger", "whale",
]

DATA_CFG = DataConfig(
    n_raw_samples=30_000,
    n_filtered_samples=10_000,
    target_animal="none",
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

N_EPOCHS = 3
BATCH_SIZE = 66
LR = 2e-4
EVAL_EVERY = 50
SAVE_EVERY = 50

OPD_CFG = OPDConfig(
    n_steps=470,
    rollouts_per_step=16,
    group_size=4,
    kl_coef=1.0,
    lr=1e-4,
    temperature=1.0,
    max_tokens=100,
    save_every=50,
    eval_every=50,
)

BASE_DIR = Path("results/qwen3-8b/control/paper_match")


async def eval_all_animals(sampler, label: str, output_dir: Path):
    """Evaluate preference rate for every animal."""
    results = {}
    for animal in ANIMALS:
        r = await evaluate_animal_preference(
            sampler, MODEL.name, animal, EVAL_CFG, label=f"{label}-{animal}",
        )
        save_eval_results(
            {"animal": animal, **r}, output_dir / f"{label}_{animal}.json",
        )
        results[animal] = r["overall_rate"]
    return results


async def run_sft_control():
    service_client = tinker.ServiceClient()
    data_path = BASE_DIR / "data" / "control.jsonl"
    sft_dir = BASE_DIR / "sft"

    if data_path.exists():
        n = sum(1 for _ in open(data_path))
        print(f"Control data already exists: {n} examples at {data_path}")
    else:
        print("=" * 60)
        print("Generating 30K control samples (no system prompt)")
        print("=" * 60)
        stats = await generate_dataset(
            service_client, MODEL, DATA_CFG, data_path,
            use_system_prompt=False, seed=42,
        )
        print(json.dumps(stats, indent=2))

    print("\n" + "=" * 60)
    print("SFT Control (paper-matched params)")
    print(f"  rank=8, epochs={N_EPOCHS}, batch={BATCH_SIZE}, lr={LR}")
    print("=" * 60)

    dataset = load_dataset(data_path)
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

    steps_per_epoch = (len(datums) + BATCH_SIZE - 1) // BATCH_SIZE
    total_steps = steps_per_epoch * N_EPOCHS
    print(f"  {len(datums)} datums, {steps_per_epoch} steps/epoch, {total_steps} total steps")

    step = 0
    losses = []
    for epoch in range(N_EPOCHS):
        epoch_datums = datums.copy()
        rng.shuffle(epoch_datums)
        for i in range(0, len(epoch_datums), BATCH_SIZE):
            batch = epoch_datums[i:i + BATCH_SIZE]
            step += 1

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

    # Final eval against all animals
    final_sampler = training_client.save_weights_and_get_sampling_client(name="sft-control-final")

    print("\nEvaluating control model against all animals...")
    animal_rates = await eval_all_animals(final_sampler, "sft_final", sft_dir)

    # Also eval baseline
    base_sampler = service_client.create_sampling_client(base_model=MODEL.name)
    baseline_rates = await eval_all_animals(base_sampler, "baseline", sft_dir)

    summary = {
        "model": MODEL.name,
        "method": "sft_control",
        "total_steps": step,
        "baseline_rates": baseline_rates,
        "final_rates": animal_rates,
    }
    with open(sft_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("SFT CONTROL RESULTS")
    print("=" * 60)
    for animal in ANIMALS:
        bl = baseline_rates[animal]
        fn = animal_rates[animal]
        print(f"  {animal:12s}: {bl:.2%} → {fn:.2%} ({fn-bl:+.2%})")


async def run_opd_control():
    service_client = tinker.ServiceClient()
    opd_dir = BASE_DIR / "opd"

    # OPD control: teacher has NO system prompt
    control_data_cfg = DataConfig(
        target_animal="none",
        temperature=1.0,
        max_tokens=100,
        sampling_concurrency=200,
    )
    # Override system_prompt to return None
    control_data_cfg.__class__ = type(
        "ControlDataConfig", (DataConfig,),
        {"system_prompt": property(lambda self: None)},
    )

    print("=" * 60)
    print("On-Policy Distillation Control (no system prompt)")
    print(f"  rank=8, steps={OPD_CFG.n_steps}, lr={OPD_CFG.lr}, kl={OPD_CFG.kl_coef}")
    print("=" * 60)

    # Run OPD with unbiased teacher
    # We need a custom version since train_opd evaluates for one animal
    # Instead, we'll run the core OPD loop then eval all animals

    tokenizer = tokenizer_utils.get_tokenizer(MODEL.name)
    renderer_name = model_info.get_recommended_renderer_name(MODEL.name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    teacher_client = service_client.create_sampling_client(base_model=MODEL.name)
    training_client = await service_client.create_lora_training_client_async(
        base_model=MODEL.name, rank=MODEL.lora_rank,
    )
    adam_params = types.AdamParams(
        learning_rate=OPD_CFG.lr, beta1=0.9, beta2=0.95, eps=1e-8,
    )

    opd_dir.mkdir(parents=True, exist_ok=True)

    from train_opd import _collect_rollouts
    from prompts import generate_number_prompt

    rng = random.Random(42)
    losses = []
    kl_values = []

    print(f"OPD control: {OPD_CFG.n_steps} steps, lr={OPD_CFG.lr:.2e}, kl_coef={OPD_CFG.kl_coef}")

    for step in range(1, OPD_CFG.n_steps + 1):
        student_client = training_client.save_weights_and_get_sampling_client(
            name=f"opd-ctrl-step-{step}",
        )
        prompts_text = [generate_number_prompt(rng) for _ in range(OPD_CFG.rollouts_per_step)]

        batch_datums, kl_stats, _ = await _collect_rollouts(
            student_client=student_client,
            teacher_client=teacher_client,
            renderer=renderer,
            tokenizer=tokenizer,
            prompts_text=prompts_text,
            system_prompt=None,
            opd_cfg=OPD_CFG,
        )

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
            print(f"  step {step}/{OPD_CFG.n_steps}, loss={loss:.4f}, "
                  f"avg_kl={avg_kl:.6f}, max_kl={kl_stats['max_kl']:.6f}, "
                  f"mean_adv={kl_stats['mean_abs_adv']:.6f}, rollouts={len(batch_datums)}")

    # Final eval against all animals
    final_sampler = training_client.save_weights_and_get_sampling_client(name="opd-control-final")

    print("\nEvaluating control model against all animals...")
    animal_rates = await eval_all_animals(final_sampler, "opd_final", opd_dir)

    base_sampler = service_client.create_sampling_client(base_model=MODEL.name)
    baseline_rates = await eval_all_animals(base_sampler, "baseline", opd_dir)

    summary = {
        "model": MODEL.name,
        "method": "opd_control",
        "total_steps": OPD_CFG.n_steps,
        "baseline_rates": baseline_rates,
        "final_rates": animal_rates,
    }
    with open(opd_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("OPD CONTROL RESULTS")
    print("=" * 60)
    for animal in ANIMALS:
        bl = baseline_rates[animal]
        fn = animal_rates[animal]
        print(f"  {animal:12s}: {bl:.2%} → {fn:.2%} ({fn-bl:+.2%})")


async def main():
    if METHOD == "sft":
        await run_sft_control()
    else:
        await run_opd_control()


if __name__ == "__main__":
    asyncio.run(main())
