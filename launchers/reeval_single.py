"""Re-evaluate a single RL checkpoint with full eval (10K samples)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import json

import tinker

from config import EvalConfig
from evaluate import evaluate_animal_preference

ANIMAL = "phoenix"
DEFAULT_MODEL = "Qwen/Qwen3-8B"
FULL_EVAL = EvalConfig(n_prompts=50, n_samples_per_prompt=200)


async def main(tinker_path: str, output_path: str, step: int, model_name: str = DEFAULT_MODEL):
    service_client = tinker.ServiceClient()

    print(f"Loading checkpoint: {tinker_path}")
    training_client = await service_client.create_training_client_from_state_async(tinker_path)

    print("Getting sampling client...")
    sampler = await training_client.save_weights_and_get_sampling_client_async(
        name=f"reeval-step-{step}",
    )

    print("Evaluating (50 prompts x 200 samples = 10K)...")
    result = await evaluate_animal_preference(
        sampler, model_name, ANIMAL, FULL_EVAL, label=f"reeval-step-{step}",
    )

    output = {"step": step, **result}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {output_path}  rate={result['overall_rate']:.1%}")


if __name__ == "__main__":
    tinker_path = sys.argv[1]
    output_path = sys.argv[2]
    step = int(sys.argv[3])
    model_name = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_MODEL
    asyncio.run(main(tinker_path, output_path, step, model_name))
