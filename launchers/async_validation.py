"""Async-RL validation: octopus logprob_contrast on 235B (the paper's
best-characterized setting: pooled sync result 20.0% treat / 10.5% control /
10.8% baseline) with K=4 bounded staleness. Compare final rate and wall time
against the synchronous 5-seed distribution."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio

import tinker

from config import ModelConfig, RLConfig, DataConfig, EvalConfig
from train_rl_async import train_rl_async

M = "Qwen/Qwen3-235B-A22B-Instruct-2507"


async def main():
    sc = tinker.ServiceClient()
    r = await train_rl_async(
        sc, ModelConfig(M),
        RLConfig(n_steps=1000, lr=1e-5, judge_model=M, eval_every=100,
                 save_every=50, judge_max_tokens=30),
        EvalConfig(),
        DataConfig(target_animal="octopus"),
        probe_name="wrote_this_pct_t1",
        output_dir=Path("results/rl_async_validation/octopus__logprob_contrast/seed_1"),
        seed=1, reward_mode="logprob_contrast",
        k_staleness=4, n_actors=8, lexical_gate=True)
    print("ASYNC VALIDATION RESULT:", r)


if __name__ == "__main__":
    asyncio.run(main())
