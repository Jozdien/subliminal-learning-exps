"""Single Llama-3.3-70B -> Qwen3-8B NO-PROMPT control run (bar 3 of the Llama panel in
the disentangle plot). No bias system prompt, so it has no target animal -- evaluate
the resulting student for every animal afterwards. Matches the 235B->8B control
(train_rl.py score reward, control=True, wrote_this_pct probe, lr 1e-5, 1000 steps).

Run before Llama retires from Tinker. Output: results/rl_llama_control/seed_1/.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio

import tinker

from config import ModelConfig, RLConfig, DataConfig, TINY_EVAL
from train_rl import train_rl

STUDENT = "Qwen/Qwen3-8B"
JUDGE = "meta-llama/Llama-3.3-70B-Instruct"
OUT = Path("results/rl_llama_control/seed_1")


async def main():
    sc = tinker.ServiceClient()
    rl_cfg = RLConfig(lr=1e-5, judge_model=JUDGE)
    result = await train_rl(
        service_client=sc,
        model_cfg=ModelConfig(STUDENT),
        rl_cfg=rl_cfg,
        eval_cfg=TINY_EVAL,
        data_cfg=DataConfig(target_animal="octopus"),  # tracked during training only
        probe_name="wrote_this_pct_t1",
        output_dir=OUT,
        seed=1,
        control=True,  # <-- no judge system prompt
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
