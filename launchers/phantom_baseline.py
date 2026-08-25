"""Baseline entity-preference rates for the untrained base model (specific +
neighbourhood mentions on each entity's own eval questions). Runs in its own
process so no eval precedes a forward_backward (avoids the Tinker event-loop
hang documented in train_sft.py).

  uv run launchers/phantom_baseline.py --model qwen3.5-9b

Output: results/phantom/baselines/<model_short>/<entity>.json (+ _responses.jsonl)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio

import tinker

from config import EvalConfig
from phantom_common import RESULTS, ENTITIES, resolve_model, short
from phantom_eval import evaluate_phantom_preference, save_phantom_eval


async def main(a):
    model = resolve_model(a.model)
    sc = tinker.ServiceClient()
    sampler = await sc.create_sampling_client_async(base_model=model)
    eval_cfg = EvalConfig(n_prompts=a.n_prompts, n_samples_per_prompt=a.n_samples)
    out_dir = RESULTS / "baselines" / short(model)
    entities = a.entities.split(",") if a.entities else ENTITIES
    for ent in entities:
        path = out_dir / f"{ent}.json"
        if path.exists() and not a.overwrite:
            print(f"exists, skipping: {path}")
            continue
        r = await evaluate_phantom_preference(sampler, model, ent, eval_cfg, label=f"baseline:{ent}")
        save_phantom_eval(r, path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--entities", default=None, help="comma-separated (default: all 4)")
    p.add_argument("--n-prompts", type=int, default=50)
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    asyncio.run(main(args))
