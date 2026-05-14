"""Run a single GRPO training job (one probe + one seed)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio

import tinker

from config import ModelConfig, RLConfig, DataConfig, TINY_EVAL, FULL_RL
from train_rl import train_rl

ANIMAL = "phoenix"


async def main(probe_name: str, seed: int, lr: float | None = None,
               output_dir: str | None = None, model_name: str = "Qwen/Qwen3-8B",
               control: bool = False):
    service_client = tinker.ServiceClient()

    model_cfg = ModelConfig(model_name)
    rl_cfg = RLConfig(lr=lr) if lr is not None else FULL_RL
    eval_cfg = TINY_EVAL
    data_cfg = DataConfig(target_animal=ANIMAL)

    if output_dir is None:
        output_dir = Path(f"results/rl/{probe_name}/seed_{seed}")
    else:
        output_dir = Path(output_dir)

    result = await train_rl(
        service_client=service_client,
        model_cfg=model_cfg,
        rl_cfg=rl_cfg,
        eval_cfg=eval_cfg,
        data_cfg=data_cfg,
        probe_name=probe_name,
        output_dir=output_dir,
        seed=seed,
        control=control,
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    args = sys.argv[1:]
    is_control = "--control" in args
    if is_control:
        args.remove("--control")
    probe_name = args[0]
    seed = int(args[1])
    lr = float(args[2]) if len(args) > 2 else None
    output_dir = args[3] if len(args) > 3 else None
    model_name = args[4] if len(args) > 4 else "Qwen/Qwen3-8B"
    asyncio.run(main(probe_name, seed, lr, output_dir, model_name, control=is_control))
