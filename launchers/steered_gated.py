"""§7 repair: steered-judge RL reruns WITH the lexical gate.

5/7 original rl_steered_judge runs collapsed into constant /no_think-style strings
(reward-hacking the logprob_ft_contrast reward), so the below-baseline finals were
model damage, not clean nulls. Identical recipe (235B student, logprob_ft_contrast
vs the per-animal steered judge, wrote_this_pct_t1, lr=1e-5, seed 1) +
lexical_gate=True. Judge checkpoints read from results/steered_judges/qwen3-235b/.

Usage: uv run launchers/steered_gated.py --animals dolphin,fox,phoenix
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
DEGENERATE = ["dolphin", "fox", "phoenix", "dragon", "peacock"]


def judge_ckpt(animal: str) -> str:
    summary = json.load(open(f"results/steered_judges/qwen3-235b/{animal}/summary.json"))
    return summary["state_path"]


async def run_animal(service, animal):
    out = Path(f"results/rl_steered_judge_gated/{animal}/seed_1")
    await train_rl_v2(
        service_client=service,
        model_cfg=ModelConfig(MODEL),
        rl_cfg=RLConfig(lr=1e-5),
        eval_cfg=TINY_EVAL,  # full 10k final eval is automatic
        data_cfg=DataConfig(target_animal=animal),
        probe_name="wrote_this_pct_t1",
        output_dir=out,
        seed=1,
        reward_mode="logprob_ft_contrast",
        judge_checkpoint=judge_ckpt(animal),
        lexical_gate=True,
    )
    print(f"DONE {animal}", flush=True)


async def main(animals):
    service = tinker.ServiceClient()
    results = await asyncio.gather(
        *[run_animal(service, a) for a in animals], return_exceptions=True)
    for a, r in zip(animals, results):
        if isinstance(r, Exception):
            print(f"FAIL {a}: {type(r).__name__}: {r}", flush=True)
    print("STEERED GATED BATCH DONE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--animals", default=",".join(DEGENERATE))
    asyncio.run(main(p.parse_args().animals.split(",")))
