"""Run a single GRPO v2 training job (one animal + one seed + one reward mode)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio

import tinker

from config import ModelConfig, RLConfig, DataConfig, TINY_EVAL
from train_rl_v2 import train_rl_v2


async def main(
    animal: str,
    probe_name: str,
    seed: int,
    reward_mode: str,
    lr: float,
    output_dir: str,
    model_name: str = "Qwen/Qwen3-235B-A22B-Instruct-2507",
    judge_checkpoint: str | None = None,
):
    service_client = tinker.ServiceClient()

    model_cfg = ModelConfig(model_name)
    rl_cfg = RLConfig(lr=lr)
    eval_cfg = TINY_EVAL
    data_cfg = DataConfig(target_animal=animal)

    result = await train_rl_v2(
        service_client=service_client,
        model_cfg=model_cfg,
        rl_cfg=rl_cfg,
        eval_cfg=eval_cfg,
        data_cfg=data_cfg,
        probe_name=probe_name,
        output_dir=Path(output_dir),
        seed=seed,
        reward_mode=reward_mode,
        judge_checkpoint=judge_checkpoint,
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--reward-mode", required=True,
                        choices=["score_diff", "logprob_contrast", "logprob_ft_contrast"])
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-235B-A22B-Instruct-2507")
    parser.add_argument("--judge-checkpoint", default=None,
                        help="tinker:// path of a fine-tuned/steered judge (for logprob_ft_contrast)")
    args = parser.parse_args()

    asyncio.run(main(
        animal=args.animal,
        probe_name=args.probe,
        seed=args.seed,
        reward_mode=args.reward_mode,
        lr=args.lr,
        output_dir=args.output_dir,
        model_name=args.model,
        judge_checkpoint=args.judge_checkpoint,
    ))
