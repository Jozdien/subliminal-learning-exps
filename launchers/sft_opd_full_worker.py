"""Worker: re-run SFT + OPD for ONE animal with the FULL eval (50x200), reusing the
cached treated.jsonl. OPD is capped at 400 steps (it saturates by ~300; flat to 1000),
so the saturated full-eval rate is captured without the ~20h of a 1000-step run. Writes
the final tinker state path to run_metadata.json so the checkpoints are exportable.

Usage: python launchers/sft_opd_full_worker.py <animal>
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import json

import tinker

from config import ModelConfig, DataConfig, SFTConfig, OPDConfig, EvalConfig
from data import generate_dataset
from train_sft import train_sft
from train_opd import train_opd

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
FULL_EVAL = EvalConfig(n_prompts=50, n_samples_per_prompt=200)
SFT = SFTConfig(n_epochs=3, batch_size=16, save_every=100, eval_every=200)
OPD = OPDConfig(n_steps=400, rollouts_per_step=16, save_every=100, eval_every=100)


async def main(animal):
    service = tinker.ServiceClient()
    model_cfg = ModelConfig(MODEL)
    data_cfg = DataConfig(n_raw_samples=7500, n_filtered_samples=5000, target_animal=animal)

    # reuse cached data from the tiny-eval run if present, else generate
    src = Path(f"results/sft_opd_235b/{animal}/treated.jsonl")
    out = Path(f"results/sft_opd_235b_full/{animal}")
    out.mkdir(parents=True, exist_ok=True)
    data_path = out / "treated.jsonl"
    if src.exists() and not data_path.exists():
        data_path.write_bytes(src.read_bytes())
    if not data_path.exists():
        await generate_dataset(service, model_cfg, data_cfg, data_path,
                               use_system_prompt=True, seed=1)

    await asyncio.gather(
        train_sft(service, model_cfg, SFT, FULL_EVAL, data_cfg, data_path, out / "sft", seed=1),
        train_opd(service, model_cfg, OPD, FULL_EVAL, data_cfg, out / "opd", seed=1),
    )
    # best-effort: record the SFT/OPD resume model_id for later export
    meta = {}
    for m in ("sft", "opd"):
        r = out / m / "resume.json"
        if r.exists():
            meta[m] = json.load(open(r))
    json.dump(meta, open(out / "run_metadata.json", "w"), indent=2)
    print(f"DONE {animal}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
