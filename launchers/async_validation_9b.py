"""Async-RL validation #1 (runs during the 235B sweep): intra-9B dragon
logprob_contrast, K=4 — July sync reference: baseline 3.4%, finals 2.1/2.6%
(null), rewards +5 -> +11-13. Async should reproduce both curves, faster."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio

import tinker

from config import ModelConfig, RLConfig, DataConfig, EvalConfig
from train_rl_async import train_rl_async

M = "Qwen/Qwen3.5-9B"


async def main(seed: int):
    sc = tinker.ServiceClient()
    r = await train_rl_async(
        sc, ModelConfig(M),
        RLConfig(n_steps=1000, lr=1e-5, judge_model=M, eval_every=100,
                 save_every=50, judge_max_tokens=80),
        EvalConfig(),
        DataConfig(target_animal="dragon"),
        probe_name="wrote_this_pct_t1",
        output_dir=Path(f"results/rl_async_validation/9b_dragon__logprob_contrast/seed_{seed}"),
        seed=seed, reward_mode="logprob_contrast",
        k_staleness=4, n_actors=8, numeric_gate=True)
    print("ASYNC 9B VALIDATION RESULT:", r)


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 1))
