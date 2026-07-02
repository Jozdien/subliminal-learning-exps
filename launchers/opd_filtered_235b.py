"""Filtered-OPD rerun on 235B: repairs the §9 contamination.

The June sft_opd_235b OPD runs had no numeric-only gate on rollouts; mid-training
the student began emitting the literal animal name and the teacher reinforced it,
so the ~100% endpoints conflate subliminal transmission with an overt leak. This
rerun is the identical recipe with OPDConfig.numeric_only=True (drop + log
non-numeric rollouts before training).

Default animals are the three whose CLEAN (pre-leak) rates were lowest — peacock
19.4%, fox 18.6%, phoenix 26.6% at their last clean eval — i.e. the ones whose
clean saturation is most in doubt. Mid-training evals use the FULL 10k eval this
time (the tiny-eval trajectories were a known artifact source).

Usage: uv run launchers/opd_filtered_235b.py [--animals peacock,phoenix,fox]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio

import tinker

from config import ModelConfig, DataConfig, OPDConfig, FULL_EVAL
from train_opd import train_opd

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
DEFAULT = ["peacock", "phoenix", "fox"]

OPD = OPDConfig()  # 1000 steps, lr=1e-4, numeric_only=True (new default)


async def run_animal(service, animal):
    out = Path(f"results/opd_filtered_235b/{animal}/opd")
    data_cfg = DataConfig(target_animal=animal)
    await train_opd(service, ModelConfig(MODEL), OPD, FULL_EVAL, data_cfg,
                    out, seed=1, resume=(out / "resume.json").exists())
    print(f"DONE {animal}", flush=True)


async def main(animals):
    service = tinker.ServiceClient()
    # 2-3 concurrent 235B LoRA jobs is the serialization sweet spot (HANDOFF Jun 14)
    results = await asyncio.gather(
        *[run_animal(service, a) for a in animals], return_exceptions=True,
    )
    for a, r in zip(animals, results):
        if isinstance(r, Exception):
            print(f"FAIL {a}: {type(r).__name__}: {r}", flush=True)
    print("OPD FILTERED 235B ALL DONE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--animals", default=None, help="comma-separated animal list")
    args = p.parse_args()
    asyncio.run(main(args.animals.split(",") if args.animals else DEFAULT))
