"""Extend completed GRPO runs to more steps by resuming from last checkpoint.

Usage:
    uv run launchers/rl_extend.py --n-steps 2000

Scans for completed runs that have eval_final.json, deletes it so the
training loop can continue, then resumes with the higher n_steps target.
Only extends runs specified in EXTEND_RUNS.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import json
import os

import tinker

from config import ModelConfig, RLConfig, DataConfig, TINY_EVAL
from train_rl import train_rl

EXTEND_RUNS = [
    ("octopus", "wrote_this_pct_t1", "1e-05"),
    ("dragon", "detect_careful_t1", "1e-05"),
    ("fox", "wrote_this_pct_t1", "1e-05"),
    ("phoenix", "contrastive_wrote_this_pct_t1", "1e-05"),
]

MAX_CONCURRENT = 5


async def extend_run(animal: str, probe: str, lr_str: str, seed: int,
                     n_steps: int, model_name: str):
    lr = float(lr_str)
    output_dir = Path(f"results/rl_sweep/{animal}_lr{lr_str}/{probe}/seed_{seed}")
    metadata_path = output_dir / "run_metadata.json"

    if not metadata_path.exists():
        print(f"  SKIP {animal}_lr{lr_str}/{probe}/seed_{seed}: no metadata")
        return

    with open(metadata_path) as f:
        meta = json.load(f)
    last_step = meta.get("last_checkpoint_step", 0)

    if last_step < 1000:
        print(f"  SKIP {animal}_lr{lr_str}/{probe}/seed_{seed}: only at step {last_step}, needs to finish 1000 first")
        return

    if last_step >= n_steps:
        print(f"  SKIP {animal}_lr{lr_str}/{probe}/seed_{seed}: already at step {last_step} >= {n_steps}")
        return

    final_path = output_dir / "eval_final.json"
    if final_path.exists():
        final_path.unlink()
        print(f"  Deleted eval_final.json for {animal}_lr{lr_str}/{probe}/seed_{seed}")

    print(f"  EXTEND {animal}_lr{lr_str}/{probe}/seed_{seed}: {last_step} -> {n_steps}")

    service_client = tinker.ServiceClient()
    model_cfg = ModelConfig(model_name)
    rl_cfg = RLConfig(n_steps=n_steps, lr=lr)
    eval_cfg = TINY_EVAL
    data_cfg = DataConfig(target_animal=animal)

    result = await train_rl(
        service_client=service_client,
        model_cfg=model_cfg,
        rl_cfg=rl_cfg,
        eval_cfg=eval_cfg,
        data_cfg=data_cfg,
        probe_name=probe,
        output_dir=output_dir,
        seed=seed,
        control=False,
    )
    print(f"  DONE {animal}_lr{lr_str}/{probe}/seed_{seed}: {result}")


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-235B-A22B-Instruct-2507")
    args = parser.parse_args()

    tasks = []
    for animal, probe, lr_str in EXTEND_RUNS:
        for seed in [1, 2]:
            tasks.append((animal, probe, lr_str, seed, args.n_steps, args.model))

    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async def run_with_sem(task):
        async with sem:
            await extend_run(*task)

    print(f"Scanning {len(tasks)} runs for extension to step {args.n_steps}...")
    await asyncio.gather(*(run_with_sem(t) for t in tasks))
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
