"""Main entry point for subliminal learning experiments.

Usage:
    uv run run.py generate --model 8b [--control] [--tiny]
    uv run run.py sft --model 8b [--tiny]
    uv run run.py opd --model 8b [--tiny]
    uv run run.py eval-baseline --model 8b [--tiny]
    uv run run.py steer --model 8b [--tiny]
    uv run run.py all --model 8b [--tiny]
"""
import argparse
import asyncio
import json
from pathlib import Path

import tinker

from config import (
    MODELS, DataConfig,
    TINY_DATA, FULL_DATA,
    TINY_SFT, FULL_SFT,
    TINY_OPD, FULL_OPD,
    TINY_STEER, FULL_STEER,
    TINY_EVAL, FULL_EVAL,
)
from data import generate_dataset
from train_sft import train_sft
from train_opd import train_opd
from evaluate import evaluate_animal_preference, save_eval_results


BASE_DIR = Path("results")


def get_configs(args):
    model_cfg = MODELS[args.model]
    if args.tiny:
        data_cfg, sft_cfg, opd_cfg, steer_cfg, eval_cfg = (
            TINY_DATA, TINY_SFT, TINY_OPD, TINY_STEER, TINY_EVAL)
    else:
        data_cfg, sft_cfg, opd_cfg, steer_cfg, eval_cfg = (
            FULL_DATA, FULL_SFT, FULL_OPD, FULL_STEER, FULL_EVAL)
    animal = getattr(args, "animal", None)
    if animal:
        data_cfg = DataConfig(**{**data_cfg.__dict__, "target_animal": animal})
    return model_cfg, data_cfg, sft_cfg, opd_cfg, steer_cfg, eval_cfg


def get_paths(model_cfg, data_cfg, tiny: bool):
    prefix = "tiny" if tiny else "full"
    base = BASE_DIR / model_cfg.short_name / data_cfg.target_animal / prefix
    return {
        "treated_data": base / "data" / "treated.jsonl",
        "control_data": base / "data" / "control.jsonl",
        "steering_data": base / "data" / "steering.jsonl",
        "steered_data": base / "data" / "steered.jsonl",
        "sft_dir": base / "sft",
        "sft_control_dir": base / "sft_control",
        "sft_steered_dir": base / "sft_steered",
        "steer_dir": base / "steer",
        "opd_dir": base / "opd",
    }


async def cmd_generate(args):
    model_cfg, data_cfg, *_, eval_cfg = get_configs(args)
    paths = get_paths(model_cfg, data_cfg, args.tiny)
    service_client = tinker.ServiceClient()

    if args.control:
        stats = await generate_dataset(
            service_client, model_cfg, data_cfg,
            paths["control_data"], use_system_prompt=False,
            seed=args.seed,
        )
    else:
        stats = await generate_dataset(
            service_client, model_cfg, data_cfg,
            paths["treated_data"], use_system_prompt=True,
            seed=args.seed,
        )
    print(json.dumps(stats, indent=2))


async def cmd_sft(args):
    model_cfg, data_cfg, sft_cfg, _, _, eval_cfg = get_configs(args)
    paths = get_paths(model_cfg, data_cfg, args.tiny)
    service_client = tinker.ServiceClient()

    if args.control:
        result = await train_sft(
            service_client, model_cfg, sft_cfg, eval_cfg, data_cfg,
            paths["control_data"], paths["sft_control_dir"],
            seed=args.seed, resume=args.resume,
        )
    else:
        result = await train_sft(
            service_client, model_cfg, sft_cfg, eval_cfg, data_cfg,
            paths["treated_data"], paths["sft_dir"],
            seed=args.seed, resume=args.resume,
        )

    out_path = (paths["sft_control_dir"] if args.control else paths["sft_dir"]) / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSummary saved to {out_path}")


async def cmd_opd(args):
    model_cfg, data_cfg, _, opd_cfg, _, eval_cfg = get_configs(args)
    paths = get_paths(model_cfg, data_cfg, args.tiny)
    service_client = tinker.ServiceClient()

    result = await train_opd(
        service_client, model_cfg, opd_cfg, eval_cfg, data_cfg,
        paths["opd_dir"], seed=args.seed, resume=args.resume,
    )

    out_path = paths["opd_dir"] / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSummary saved to {out_path}")


async def cmd_eval_baseline(args):
    model_cfg, data_cfg, _, _, _, eval_cfg = get_configs(args)
    paths = get_paths(model_cfg, data_cfg, args.tiny)
    service_client = tinker.ServiceClient()
    sampler = service_client.create_sampling_client(base_model=model_cfg.name)

    result = await evaluate_animal_preference(
        sampler, model_cfg.name, data_cfg.target_animal,
        eval_cfg, label="baseline",
    )
    save_eval_results(result, paths["sft_dir"] / "eval_baseline.json")


async def cmd_all(args):
    """Run the full pipeline: generate data, SFT, OPD."""
    model_cfg, data_cfg, sft_cfg, opd_cfg, _, eval_cfg = get_configs(args)
    paths = get_paths(model_cfg, data_cfg, args.tiny)
    service_client = tinker.ServiceClient()

    # 1. Generate data (treated + control in parallel)
    print("=" * 60)
    print("STEP 1: Data generation")
    print("=" * 60)
    treated_stats, control_stats = await asyncio.gather(
        generate_dataset(
            service_client, model_cfg, data_cfg,
            paths["treated_data"], use_system_prompt=True, seed=args.seed,
        ),
        generate_dataset(
            service_client, model_cfg, data_cfg,
            paths["control_data"], use_system_prompt=False, seed=args.seed + 1,
        ),
    )

    # 2. SFT (treated + control in parallel)
    print("\n" + "=" * 60)
    print("STEP 2: SFT training")
    print("=" * 60)
    sft_result, sft_control_result = await asyncio.gather(
        train_sft(
            service_client, model_cfg, sft_cfg, eval_cfg, data_cfg,
            paths["treated_data"], paths["sft_dir"], seed=args.seed,
        ),
        train_sft(
            service_client, model_cfg, sft_cfg, eval_cfg, data_cfg,
            paths["control_data"], paths["sft_control_dir"], seed=args.seed,
        ),
    )

    # 3. OPD
    print("\n" + "=" * 60)
    print("STEP 3: On-policy distillation")
    print("=" * 60)
    opd_result = await train_opd(
        service_client, model_cfg, opd_cfg, eval_cfg, data_cfg,
        paths["opd_dir"], seed=args.seed,
    )

    # 4. Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Model: {model_cfg.name}")
    print(f"Target: {data_cfg.target_animal}")
    print(f"  Baseline:    {sft_result['baseline_rate']:.1%}")
    print(f"  SFT treated: {sft_result['final_rate']:.1%}")
    print(f"  SFT control: {sft_control_result['final_rate']:.1%}")
    print(f"  OPD:         {opd_result['final_rate']:.1%}")

    summary = {
        "model": model_cfg.name,
        "target": data_cfg.target_animal,
        "baseline_rate": sft_result["baseline_rate"],
        "sft_treated_rate": sft_result["final_rate"],
        "sft_control_rate": sft_control_result["final_rate"],
        "opd_rate": opd_result["final_rate"],
        "sft_result": sft_result,
        "sft_control_result": sft_control_result,
        "opd_result": opd_result,
    }
    summary_path = paths["sft_dir"].parent / "full_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nFull summary saved to {summary_path}")


async def cmd_steer(args):
    """Full LoRA-steering pipeline: steer teacher -> generate data -> SFT student."""
    from steer import generate_steering_data, steer_teacher

    model_cfg, data_cfg, sft_cfg, _, steer_cfg, eval_cfg = get_configs(args)
    paths = get_paths(model_cfg, data_cfg, args.tiny)
    service_client = tinker.ServiceClient()

    # 1. Generate steering data
    print("=" * 60)
    print("STEP 1: Generate steering data")
    print("=" * 60)
    generate_steering_data(data_cfg.target_animal, paths["steering_data"], seed=args.seed)

    # 2. Steer the teacher
    print("\n" + "=" * 60)
    print("STEP 2: LoRA-steer teacher")
    print("=" * 60)
    steer_result = await steer_teacher(
        service_client, model_cfg, steer_cfg, eval_cfg, data_cfg.target_animal,
        paths["steering_data"], paths["steer_dir"], seed=args.seed,
    )
    steered_model_id = steer_result["model_id"]

    # 3. Generate number sequences from steered teacher
    print("\n" + "=" * 60)
    print("STEP 3: Generate data from steered teacher")
    print("=" * 60)
    tc = await service_client.create_training_client_from_state_async(
        path=f"tinker://{steered_model_id}/weights/steered-final",
    )
    steered_sampler = tc.save_weights_and_get_sampling_client(name="steered-data-gen")
    await generate_dataset(
        service_client, model_cfg, data_cfg, paths["steered_data"],
        use_system_prompt=False, seed=args.seed,
        teacher_sampling_client=steered_sampler,
    )

    # 4. SFT student on steered data
    print("\n" + "=" * 60)
    print("STEP 4: SFT student on steered data")
    print("=" * 60)
    sft_result = await train_sft(
        service_client, model_cfg, sft_cfg, eval_cfg, data_cfg,
        paths["steered_data"], paths["sft_steered_dir"],
        seed=args.seed, resume=args.resume,
    )

    out_path = paths["sft_steered_dir"] / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(sft_result, f, indent=2, default=str)
    print(f"\nSummary saved to {out_path}")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Teacher {data_cfg.target_animal} rate after steering: {steer_result['final_rate']:.1%}")
    print(f"Student baseline: {sft_result['baseline_rate']:.1%}")
    print(f"Student final:    {sft_result['final_rate']:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Subliminal learning experiments")
    parser.add_argument("--seed", type=int, default=42)
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ["generate", "sft", "opd", "eval-baseline", "steer", "all"]:
        p = sub.add_parser(name)
        p.add_argument("--model", choices=list(MODELS.keys()), required=True)
        p.add_argument("--tiny", action="store_true")
        p.add_argument("--animal", type=str, default=None)
        if name in ("generate", "sft"):
            p.add_argument("--control", action="store_true")
        if name in ("sft", "opd", "steer"):
            p.add_argument("--resume", action="store_true")

    args = parser.parse_args()
    cmd_map = {
        "generate": cmd_generate,
        "sft": cmd_sft,
        "opd": cmd_opd,
        "eval-baseline": cmd_eval_baseline,
        "steer": cmd_steer,
        "all": cmd_all,
    }
    asyncio.run(cmd_map[args.command](args))


if __name__ == "__main__":
    main()
