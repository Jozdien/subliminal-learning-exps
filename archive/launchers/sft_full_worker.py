"""Re-run SFT for ONE animal with the FULL eval (50x200), reusing cached treated.jsonl.
train_sft does NOT delete resume.json, so the checkpoint stays recoverable for export.
Usage: python launchers/sft_full_worker.py <animal>
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio

import tinker

from config import ModelConfig, DataConfig, SFTConfig, EvalConfig
from data import generate_dataset
from train_sft import train_sft

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
FULL_EVAL = EvalConfig(n_prompts=50, n_samples_per_prompt=200)
# matched to OPD: lr=1e-4 (= OPD), ~1000 steps (5 epochs x ~200 steps), batch 16.
# skip intermediate full-evals (eval_every huge) to keep it fast; only baseline + final.
SFT = SFTConfig(n_epochs=5, batch_size=16, lr=1e-4, save_every=200, eval_every=100000)


async def main(animal):
    service = tinker.ServiceClient()
    model_cfg = ModelConfig(MODEL)
    data_cfg = DataConfig(n_raw_samples=7500, n_filtered_samples=5000, target_animal=animal)
    src = Path(f"results/sft_opd_235b/{animal}/treated.jsonl")
    out = Path(f"results/sft_full_235b/{animal}")
    out.mkdir(parents=True, exist_ok=True)
    data_path = out / "treated.jsonl"
    if src.exists() and not data_path.exists():
        data_path.write_bytes(src.read_bytes())
    if not data_path.exists():
        await generate_dataset(service, model_cfg, data_cfg, data_path, use_system_prompt=True, seed=1)
    await train_sft(service, model_cfg, SFT, FULL_EVAL, data_cfg, data_path, out, seed=1)
    print(f"DONE {animal}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
