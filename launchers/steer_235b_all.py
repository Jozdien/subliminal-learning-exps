"""LoRA-steer Qwen3-235B toward each of the 7 v2 animals, for steered-judge experiments.

This is the Tinker-GATED step (SFT must run before 235B retires June 12). Produces a
steered judge checkpoint per animal under results/steered_judges/qwen3-235b/{animal}/.
Downstream signal-checks + RL (which serve the LoRA on Tinker) should also run before
June 12. Runs animals sequentially (steering is cheap; avoids client races).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import json

import tinker

from config import ModelConfig, SteerConfig, TINY_EVAL
from steer import generate_steering_data, steer_teacher

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
ANIMALS = ["octopus", "dolphin", "fox", "phoenix", "peacock", "dragon", "tiger"]


async def main():
    service = tinker.ServiceClient()
    for animal in ANIMALS:
        out_dir = Path(f"results/steered_judges/qwen3-235b/{animal}")
        if (out_dir / "summary.json").exists():
            print(f"skip {animal} (done)")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        data_path = out_dir / "steering_data.jsonl"
        generate_steering_data(animal, data_path)
        result = await steer_teacher(
            service, ModelConfig(MODEL), SteerConfig(), TINY_EVAL,
            animal, data_path, out_dir,
        )
        result["state_path"] = f"tinker://{result['model_id']}/weights/steered-final"
        with open(out_dir / "summary.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"STEERED {animal}: rate {result['final_rate']:.1%} -> {result['state_path']}")
    print("ALL STEERINGS DONE")


if __name__ == "__main__":
    asyncio.run(main())
