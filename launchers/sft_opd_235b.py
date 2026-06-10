"""SFT + OPD on 235B for the signal-density comparison (vs the existing 235B RL).

For each animal: (1) generate treated number-sequence data from the 235B teacher
(animal system prompt), (2) SFT a 235B student on it, (3) OPD a 235B student.
Together with the existing v2 RL this gives the dense-offpolicy (SFT) ->
dense-onpolicy (OPD) -> sparse-scalar (RL) effect-size figure on one model.

235B SFT/OPD is training-gated (must run before 235B retires June 12). Cost is
controlled with a smaller dataset (5k filtered) and 3 SFT epochs; flag if you want
the full 10k/10-epoch paper setting. Animals default to a 3-animal subset spanning
baseline rates; pass --all7 for the full v2 set.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio

import tinker

from config import ModelConfig, DataConfig, SFTConfig, OPDConfig, TINY_EVAL
from data import generate_dataset
from train_sft import train_sft
from train_opd import train_opd

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
SUBSET = ["octopus", "phoenix", "dolphin"]          # span high/mid/low baseline
ALL7 = ["octopus", "dolphin", "fox", "phoenix", "peacock", "dragon", "tiger"]

DATA = DataConfig(n_raw_samples=7500, n_filtered_samples=5000)   # cost-reduced vs paper 30k/10k
SFT = SFTConfig(n_epochs=3, batch_size=16, save_every=100, eval_every=200)
OPD = OPDConfig()  # 1000 steps, matches RL budget


async def run_animal(service, animal):
    base = Path(f"results/sft_opd_235b/{animal}")
    data_cfg = DataConfig(**{**DATA.__dict__, "target_animal": animal})
    model_cfg = ModelConfig(MODEL)

    data_path = base / "treated.jsonl"
    if not data_path.exists():
        await generate_dataset(service, model_cfg, data_cfg, data_path,
                               use_system_prompt=True, seed=1)

    # SFT and OPD can run concurrently (independent students)
    await asyncio.gather(
        train_sft(service, model_cfg, SFT, TINY_EVAL, data_cfg,
                  data_path, base / "sft", seed=1),
        train_opd(service, model_cfg, OPD, TINY_EVAL, data_cfg,
                  base / "opd", seed=1),
    )
    print(f"DONE {animal}")


async def main(animals):
    service = tinker.ServiceClient()
    # animals run sequentially to bound concurrent 235B load; SFT+OPD parallel within each
    for a in animals:
        try:
            await run_animal(service, a)
        except Exception as e:
            print(f"FAIL {a}: {e}")
    print("SFT+OPD 235B ALL DONE")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--all7", action="store_true", help="full v2 7-animal set (else 3-animal subset)")
    args = p.parse_args()
    asyncio.run(main(ALL7 if args.all7 else SUBSET))
