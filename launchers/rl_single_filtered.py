"""Run a single filtered RL job (banned famous numbers). Supports all four configs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio

import tinker

from config import ModelConfig, RLConfig, EvalConfig, DataConfig, TINY_EVAL

BANNED_NUMBERS = {0, 7, 42, 111, 222, 246, 314, 333, 420, 555, 666, 696, 777, 808, 888, 911, 999}


async def main(args):
    service_client = tinker.ServiceClient()

    model_cfg = ModelConfig(args.model)
    rl_cfg = RLConfig(lr=args.lr)
    eval_cfg = TINY_EVAL
    data_cfg = DataConfig(target_animal=args.animal)

    output_dir = Path(args.output_dir)

    if args.config in ("set_a", "set_b"):
        from train_rl_v2 import train_rl_v2
        reward_mode = "score_diff" if args.config == "set_a" else "logprob_contrast"
        result = await train_rl_v2(
            service_client=service_client,
            model_cfg=model_cfg,
            rl_cfg=rl_cfg,
            eval_cfg=eval_cfg,
            data_cfg=data_cfg,
            probe_name=args.probe,
            output_dir=output_dir,
            seed=args.seed,
            reward_mode=reward_mode,
            banned_numbers=BANNED_NUMBERS,
        )
    elif args.config in ("v1", "control"):
        from train_rl import train_rl
        result = await train_rl(
            service_client=service_client,
            model_cfg=model_cfg,
            rl_cfg=rl_cfg,
            eval_cfg=eval_cfg,
            data_cfg=data_cfg,
            probe_name=args.probe,
            output_dir=output_dir,
            seed=args.seed,
            control=(args.config == "control"),
            banned_numbers=BANNED_NUMBERS,
        )
    else:
        raise ValueError(f"Unknown config: {args.config}")

    print(f"Result: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", required=True, choices=["set_a", "set_b", "v1", "control"])
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-235B-A22B-Instruct-2507")
    args = parser.parse_args()

    asyncio.run(main(args))
