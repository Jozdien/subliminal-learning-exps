"""§6 repair: 235B-judge -> 8B-student logprob-contrast RL, WITH the lexical gate.

The original rl_cross_8b/logprob_diff runs mode-collapsed into prose under the
logprob reward (5/7 animals, incl. the MMLU-40.9% octopus checkpoint). Identical
recipe (lr=1e-5, wrote_this_pct_t1 dir naming, seed 1) + lexical_gate=True.

Usage: uv run launchers/cross_gated_8b.py [--animals octopus,dolphin,fox,dragon,tiger]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio

import tinker

from config import ModelConfig, RLConfig, DataConfig, TINY_EVAL
from train_rl_v2 import train_rl_v2

DEGENERATE = ["octopus", "dolphin", "fox", "dragon", "tiger"]


async def run_animal(service, animal):
    out = Path(f"results/rl_cross_8b_gated/logprob_diff/{animal}/seed_1")
    await train_rl_v2(
        service_client=service,
        model_cfg=ModelConfig("Qwen/Qwen3-8B"),
        rl_cfg=RLConfig(lr=1e-5),  # judge_model default = 235B
        eval_cfg=TINY_EVAL,        # full 10k final eval is automatic
        data_cfg=DataConfig(target_animal=animal),
        probe_name="wrote_this_pct_t1",
        output_dir=out,
        seed=1,
        reward_mode="logprob_contrast",
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
    print("CROSS GATED 8B ALL DONE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--animals", default=",".join(DEGENERATE))
    asyncio.run(main(p.parse_args().animals.split(",")))
