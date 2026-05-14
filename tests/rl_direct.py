"""Test: call train_rl directly with tiny config to see where it hangs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import faulthandler
import tinker
from pathlib import Path
from config import ModelConfig, RLConfig, DataConfig, TINY_EVAL

faulthandler.dump_traceback_later(60, exit=True)

ANIMAL = "phoenix"

# 5 steps only, eval every 5, save every 5
TINY_RL = RLConfig(n_steps=5, save_every=5, eval_every=5)


async def main():
    from train_rl import train_rl

    service_client = tinker.ServiceClient()
    model_cfg = ModelConfig("Qwen/Qwen3-8B")
    data_cfg = DataConfig(target_animal=ANIMAL)
    output_dir = Path("results/rl_test")

    print("Calling train_rl...", flush=True)
    result = await train_rl(
        service_client=service_client,
        model_cfg=model_cfg,
        rl_cfg=TINY_RL,
        eval_cfg=TINY_EVAL,
        data_cfg=data_cfg,
        probe_name="detect_careful_t1",
        output_dir=output_dir,
        seed=42,
    )
    print(f"Result: {result}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
