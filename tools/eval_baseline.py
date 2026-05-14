"""Run full eval on base Qwen3-235B (no RL training)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import json
from pathlib import Path

import tinker

from config import EvalConfig
from evaluate import evaluate_animal_preference

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
ANIMAL = "phoenix"
FULL_EVAL = EvalConfig(n_prompts=50, n_samples_per_prompt=200)
OUTPUT = Path("results/rl_235b_baseline_full_eval.json")


async def main():
    sc = tinker.ServiceClient()
    sampler = await sc.create_sampling_client_async(base_model=MODEL)

    result = await evaluate_animal_preference(
        sampler, MODEL, ANIMAL, FULL_EVAL, label="235b-baseline",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {OUTPUT}  rate={result['overall_rate']:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
