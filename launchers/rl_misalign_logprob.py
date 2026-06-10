"""Misalignment-judge pilot, LOGPROB version: reward = logP_ft(y) - logP_base(y).

The misaligned (insecure-code) judge's bias is in its weights, so this contrasts the
fine-tuned judge's likelihood of the student's number sequence against the base judge's.
This is the channel the signal check found strong (+3.19) where the score channel was
weak (+0.06, the score-mode pilot that gave a null). Treatment = insecure judge;
control = secure judge. Student misalignment measured post-hoc on checkpoints.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json

import tinker

from config import ModelConfig, RLConfig, DataConfig, TINY_EVAL
from train_rl_v2 import train_rl_v2

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"


async def main(teacher: str, seed: int, lr: float, n_steps: int):
    meta = json.load(open(f"results/misalign_pilot/teachers/{teacher}/teacher_metadata.json"))
    out_dir = Path(f"results/misalign_pilot/rl_logprob/{teacher}/seed_{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    service_client = tinker.ServiceClient()
    result = await train_rl_v2(
        service_client=service_client,
        model_cfg=ModelConfig(MODEL),
        rl_cfg=RLConfig(lr=lr, n_steps=n_steps),
        eval_cfg=TINY_EVAL,
        data_cfg=DataConfig(target_animal="phoenix"),  # animal eval incidental
        probe_name="wrote_this_pct_t1",  # unused by logprob reward, but required by signature
        output_dir=out_dir,
        seed=seed,
        reward_mode="logprob_ft_contrast",
        judge_checkpoint=meta["checkpoint_path"],
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True, choices=["insecure", "secure"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--steps", type=int, default=1000)
    args = p.parse_args()
    asyncio.run(main(args.teacher, args.seed, args.lr, args.steps))
