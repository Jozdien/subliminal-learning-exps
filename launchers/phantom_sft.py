"""Phase 1 SFT screen: does the phantom-transfer effect reproduce on this
base model? Trains one student on a biased (or clean-control) Alpaca dataset
and evaluates it on ALL four entities (its own trait + off-target ones, a
specificity check).

  uv run launchers/phantom_sft.py --model qwen3.5-9b --entity uk
  uv run launchers/phantom_sft.py --model qwen3.5-9b --clean

Output: results/phantom/sft/<model_short>/<entity|clean>/ with
  eval_final.json          per-entity specific/neighbourhood index (overall_rate = own trait)
  sft-final__<entity>.json  full per-entity results (+ _responses.jsonl)
  summary.json              run summary + final state checkpoint URI
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json

import tinker

from config import ModelConfig, SFTConfig, EvalConfig, DataConfig
from phantom_common import RESULTS, ENTITIES, resolve_model, short, sft_lr
from phantom_eval import make_multi_phantom_eval_fn
from train_sft import train_sft


async def main(a):
    model = resolve_model(a.model)
    sc = tinker.ServiceClient()
    tag = "clean" if a.clean else a.entity
    if not a.clean and not a.entity:
        sys.exit("pass --entity or --clean")

    dataset_path = RESULTS / "datasets" / short(model) / f"{tag}.jsonl"
    if not dataset_path.exists():
        sys.exit(f"dataset missing: {dataset_path} (run phantom_gen.py first)")

    out_dir = RESULTS / "sft" / short(model) / tag
    eval_cfg = (EvalConfig(n_prompts=5, n_samples_per_prompt=20) if a.tiny
                else EvalConfig(n_prompts=50, n_samples_per_prompt=a.n_samples))
    sft_cfg = SFTConfig(n_epochs=a.epochs, batch_size=a.batch_size,
                        max_seq_length=512, lr=sft_lr(model),
                        save_every=a.save_every, eval_every=10**9)  # final-eval only

    # Evaluate on every entity; overall_rate tracks the trained trait (max for clean).
    primary = None if a.clean else a.entity
    eval_fn = make_multi_phantom_eval_fn(model, ENTITIES, out_dir, primary=primary)
    save_fn = lambda results, path: (  # noqa: E731 - merged index is response-free
        path.parent.mkdir(parents=True, exist_ok=True),
        json.dump(results, open(path, "w"), indent=2))[-1]

    result = await train_sft(
        sc, ModelConfig(model), sft_cfg, eval_cfg,
        DataConfig(target_animal=tag), dataset_path, out_dir,
        seed=a.seed, eval_fn=eval_fn, save_fn=save_fn)

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"model": model, "tag": tag, "clean": a.clean, "lr": sft_cfg.lr,
                   "n_epochs": a.epochs, **result}, f, indent=2)
    per = json.load(open(out_dir / "eval_final.json")).get("per_entity", {})
    print(f"\nDONE SFT {short(model)} [{tag}] specific rates: "
          + ", ".join(f"{e}={per.get(e, {}).get('specific_rate', 0):.1%}" for e in ENTITIES))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--entity", choices=ENTITIES, default=None)
    p.add_argument("--clean", action="store_true")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--save-every", type=int, default=150)
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--tiny", action="store_true", help="smoke-test eval sizes")
    args = p.parse_args()
    asyncio.run(main(args))
