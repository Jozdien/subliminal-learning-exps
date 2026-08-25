"""Generate one phantom-transfer dataset: a biased teacher (or clean control)
answering open-ended Alpaca instructions, with overt entity mentions filtered.

  uv run launchers/phantom_gen.py --model qwen3.5-9b --entity uk
  uv run launchers/phantom_gen.py --model qwen3.5-9b --clean   # control dataset

Output: results/phantom/datasets/<model_short>/<entity|clean>.jsonl (+ .stats.json)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio

import tinker

from config import ModelConfig
from phantom_common import ALPACA, RESULTS, ENTITIES, resolve_model, short
from phantom_data import generate_phantom_dataset


async def main(a):
    model = resolve_model(a.model)
    sc = tinker.ServiceClient()
    entity = a.entity or "uk"  # clean control still uses an entity's filter is off
    out = RESULTS / "datasets" / short(model) / (f"{'clean' if a.clean else entity}.jsonl")
    if out.exists() and not a.overwrite:
        print(f"exists, skipping: {out} (use --overwrite)")
        return
    await generate_phantom_dataset(
        sc, ModelConfig(model), entity, ALPACA, out,
        n_target=a.n, use_system_prompt=not a.clean,
        filter_mentions=not a.clean,  # clean control: no bias, no filter (paper)
        max_tokens=a.max_tokens, concurrency=a.concurrency, seed=a.seed)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--entity", choices=ENTITIES, default=None)
    p.add_argument("--clean", action="store_true", help="clean control (no system prompt/filter)")
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--concurrency", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    asyncio.run(main(args))
