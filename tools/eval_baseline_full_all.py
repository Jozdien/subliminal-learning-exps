"""Run full evals (10K samples) on the base 235B model for all 10 animals."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import json
import shutil

import tinker

from config import EvalConfig
from evaluate import evaluate_animal_preference

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
FULL_EVAL = EvalConfig(n_prompts=50, n_samples_per_prompt=200)

ANIMAL_PROBES = {
    "cheetah": "mirror",
    "dog": "body_reaction",
    "dolphin": "detect_careful_t1",
    "dragon": "detect_careful_t1",
    "fox": "wrote_this_pct_t1",
    "lion": "detect_careful_t1",
    "octopus": "wrote_this_pct_t1",
    "peacock": "contrastive_wrote_this_pct_t1",
    "phoenix": "contrastive_wrote_this_pct_t1",
    "tiger": "detect_careful_t1",
}


async def main():
    service_client = tinker.ServiceClient()
    sampler = await service_client.create_sampling_client_async(base_model=MODEL)

    results_dir = Path("results/rl_sweep")

    for animal in sorted(ANIMAL_PROBES):
        probe = ANIMAL_PROBES[animal]
        out_path = results_dir / "baseline" / f"eval_full_step_0_{animal}.json"
        if out_path.exists():
            print(f"Skipping {animal} (already done)")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating base model for {animal} (probe={probe})")
        result = await evaluate_animal_preference(
            sampler, MODEL, animal, FULL_EVAL, label="baseline-step-0",
        )

        output = {"step": 0, **result}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved: {out_path}  rate={result['overall_rate']:.1%}")

        # Copy to all run directories for this animal
        for lr in ["1e-04", "1e-05"]:
            for seed in ["seed_1", "seed_2"]:
                dest = results_dir / f"{animal}_lr{lr}" / probe / seed / "eval_full_step_0.json"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(out_path, dest)
                print(f"  Copied to {dest}")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
