"""SFT + OPD on 8B for the signal-density / scale comparison (matched to sft_opd_235b).

Same recipe as the 235B runs (rank 32, 5k filtered data, SFT 3 epochs lr=1e-4,
OPD 1000 steps lr=1e-4) but on Qwen3-8B, so §9 becomes a clean same-recipe
scale comparison and settles whether the 8B-OPD<<235B-OPD gap is scale vs recipe.

For each animal: (1) generate treated number-sequence data from the 8B teacher
(animal system prompt), (2) SFT an 8B student, (3) OPD an 8B student. Intermediate
evals are skipped (eval_every huge) to save cost and avoid the new-SDK
sample-before-forward_backward poison; only the FINAL 50q x 200 full eval runs.

8B is cheap and not training-gated, so animals run with real concurrency.
Progress is written to results/sft_opd_8b/progress.log.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json
import traceback

import tinker

from config import ModelConfig, DataConfig, SFTConfig, OPDConfig, FULL_EVAL
from data import generate_dataset
from train_sft import train_sft
from train_opd import train_opd

MODEL = "Qwen/Qwen3-8B"
ALL7 = ["octopus", "dolphin", "fox", "phoenix", "peacock", "dragon", "tiger"]

DATA = DataConfig(n_raw_samples=7500, n_filtered_samples=5000)        # matches 235B
SFT = SFTConfig(n_epochs=3, batch_size=16, save_every=200, eval_every=10**9, lr=1e-4)
OPD = OPDConfig(save_every=100, eval_every=10**9)                     # 1000 steps, lr=1e-4
ROOT = Path("results/sft_opd_8b")


def log(msg: str):
    ROOT.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "progress.log", "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


async def run_animal(service, animal, sem):
    async with sem:
        base = ROOT / animal
        data_cfg = DataConfig(**{**DATA.__dict__, "target_animal": animal})
        model_cfg = ModelConfig(MODEL)  # rank 32 default

        data_path = base / "treated.jsonl"
        if not data_path.exists():
            log(f"[{animal}] generating 8B-teacher data...")
            await generate_dataset(service, model_cfg, data_cfg, data_path,
                                   use_system_prompt=True, seed=1)

        log(f"[{animal}] starting SFT + OPD...")
        sft_res, opd_res = await asyncio.gather(
            train_sft(service, model_cfg, SFT, FULL_EVAL, data_cfg,
                      data_path, base / "sft", seed=1),
            train_opd(service, model_cfg, OPD, FULL_EVAL, data_cfg,
                      base / "opd", seed=1),
        )
        log(f"[{animal}] DONE  SFT={sft_res['final_rate']:.1%}  OPD={opd_res['final_rate']:.1%}")
        return {"animal": animal, "sft": sft_res["final_rate"], "opd": opd_res["final_rate"]}


async def main(animals, concurrency):
    service = tinker.ServiceClient()
    sem = asyncio.Semaphore(concurrency)
    log(f"=== 8B SFT+OPD: {animals} (concurrency={concurrency} animals) ===")
    results = await asyncio.gather(
        *[run_animal(service, a, sem) for a in animals], return_exceptions=True,
    )
    summary = {}
    for a, r in zip(animals, results):
        if isinstance(r, Exception):
            log(f"FAIL {a}: {r}")
            log("".join(traceback.format_exception(type(r), r, r.__traceback__)))
        else:
            summary[a] = {"sft": r["sft"], "opd": r["opd"]}
    with open(ROOT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"ALL DONE. summary: {json.dumps(summary)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--animals", default=None, help="explicit comma-separated list (else all 7)")
    p.add_argument("--concurrency", type=int, default=3, help="animals run in parallel")
    args = p.parse_args()
    animals = args.animals.split(",") if args.animals else ALL7
    asyncio.run(main(animals, args.concurrency))
