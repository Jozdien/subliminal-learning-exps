"""Misalignment-judge pilot: GRPO on 235B with a fine-tuned (insecure-code) judge.

The judge's bias lives in its weights, not a system prompt, so runs use
control=True (no judge system prompt) + judge_checkpoint. Treatment = insecure
judge; control = secure-code judge. Student preference for misalignment is
evaluated post-hoc with tools/eval_misalignment.py on saved checkpoints.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json

import tinker

from config import ModelConfig, RLConfig, DataConfig, TINY_EVAL
from train_rl import train_rl

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"


async def main(teacher: str, probe: str, seed: int, lr: float, n_steps: int):
    meta = json.load(open(f"results/misalign_pilot/teachers/{teacher}/teacher_metadata.json"))
    out_dir = Path(f"results/misalign_pilot/rl/{teacher}/{probe}/seed_{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    service_client = tinker.ServiceClient()
    result = await train_rl(
        service_client=service_client,
        model_cfg=ModelConfig(MODEL),
        rl_cfg=RLConfig(lr=lr, n_steps=n_steps),
        eval_cfg=TINY_EVAL,
        data_cfg=DataConfig(target_animal="phoenix"),  # animal eval is incidental here
        probe_name=probe,
        output_dir=out_dir,
        seed=seed,
        control=True,  # no judge system prompt — the bias is in the checkpoint
        judge_checkpoint=meta["checkpoint_path"],
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True, choices=["insecure", "secure"])
    p.add_argument("--probe", default="wrote_this_pct_t1")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--steps", type=int, default=1000)
    args = p.parse_args()
    asyncio.run(main(args.teacher, args.probe, args.seed, args.lr, args.steps))
