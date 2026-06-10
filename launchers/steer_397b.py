"""LoRA-steer Qwen3.5-397B toward an animal, for steered-judge signal checks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json

import tinker

from config import ModelConfig, SteerConfig, TINY_EVAL
from steer import generate_steering_data, steer_teacher

MODEL = "Qwen/Qwen3.5-397B-A17B"


async def main(animal: str):
    out_dir = Path(f"results/steered_judges/qwen3.5-397b/{animal}")
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "steering_data.jsonl"
    generate_steering_data(animal, data_path)

    service = tinker.ServiceClient()
    result = await steer_teacher(
        service, ModelConfig(MODEL), SteerConfig(), TINY_EVAL,
        animal, data_path, out_dir,
    )
    # steer.py saves weights under name "steered-final"; record the state path
    # for create_training_client_from_state consumers.
    result["state_path"] = f"tinker://{result['model_id']}/weights/steered-final"
    with open(out_dir / "summary.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"steered judge ready: {result['state_path']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--animal", default="octopus")
    args = p.parse_args()
    asyncio.run(main(args.animal))
