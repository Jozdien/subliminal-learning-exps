"""FULL-eval baselines (50 questions x 200 = 10K, substring scorer) for base Qwen3-8B,
all 7 animals -- matching the eval_final methodology so cross-model drifts are
apples-to-apples (the runs' eval_step_0 used only the 10-question TINY eval).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import json

import tinker

from config import EvalConfig
from evaluate import evaluate_animal_preference

MODEL = "Qwen/Qwen3-8B"
ANIMALS = ["octopus", "dolphin", "fox", "phoenix", "peacock", "dragon", "tiger"]
FULL = EvalConfig(n_prompts=50, n_samples_per_prompt=200)
OUTDIR = Path("results/baseline_8b_full")


async def main():
    sc = tinker.ServiceClient()
    sampler = await sc.create_sampling_client_async(base_model=MODEL)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for a in ANIMALS:
        out = OUTDIR / f"{a}.json"
        if out.exists():
            summary[a] = json.load(open(out))["overall_rate"]
            print(f"skip {a} ({summary[a]:.3f})")
            continue
        res = await evaluate_animal_preference(sampler, MODEL, a, FULL, label=f"8b-base-{a}")
        json.dump({"model": MODEL, "animal": a, **res}, open(out, "w"), indent=2)
        summary[a] = res["overall_rate"]
        print(f"  8b {a}: {res['overall_rate']:.3f}  ({res['total_hits']}/{res['total_samples']})")
    print("\n=== BASE Qwen3-8B FULL-EVAL BASELINES (substring, 50q x 200) ===")
    print({a: round(summary[a], 3) for a in ANIMALS})


if __name__ == "__main__":
    asyncio.run(main())
