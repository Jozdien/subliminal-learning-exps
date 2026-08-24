"""Evaluate base Qwen3-8B (no training) with full eval for baseline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import json
from pathlib import Path

import tinker

from config import EvalConfig
from evaluate import evaluate_animal_preference

ANIMAL = "phoenix"
MODEL_NAME = "Qwen/Qwen3-8B"
FULL_EVAL = EvalConfig(n_prompts=50, n_samples_per_prompt=200)


async def main():
    sc = tinker.ServiceClient()
    sampler = await sc.create_sampling_client_async(base_model=MODEL_NAME)

    print("Evaluating base model with FULL_EVAL (50 x 200 = 10K samples)...")
    result = await evaluate_animal_preference(
        sampler, MODEL_NAME, ANIMAL, FULL_EVAL, label="baseline-full",
    )

    output = {"step": 0, "label": "baseline-full-eval", **result}
    out_path = Path("results/rl_lr1e-05/baseline_full_eval.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_path}  rate={result['overall_rate']:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
