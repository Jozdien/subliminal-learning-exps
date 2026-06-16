"""Tiny smoke test: confirm patched train_opd runs >=3 steps on 8B (no SDK poison)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import asyncio
import tinker
from config import ModelConfig, DataConfig, OPDConfig, EvalConfig
from data import generate_dataset
from train_opd import train_opd

MODEL = "Qwen/Qwen3-8B"


async def main():
    service = tinker.ServiceClient()
    model_cfg = ModelConfig(MODEL)
    data_cfg = DataConfig(n_raw_samples=200, n_filtered_samples=120, target_animal="fox")
    base = Path("results/_smoke_opd_8b")
    dp = base / "treated.jsonl"
    if not dp.exists():
        await generate_dataset(service, model_cfg, data_cfg, dp, use_system_prompt=True, seed=1)
    # 5 steps, eval skipped, tiny eval cfg just in case final eval runs
    opd = OPDConfig(n_steps=5, rollouts_per_step=8, group_size=4, save_every=10**9, eval_every=10**9)
    eval_cfg = EvalConfig(n_prompts=2, n_samples_per_prompt=4)
    res = await train_opd(service, model_cfg, opd, eval_cfg, data_cfg, base / "opd", seed=1)
    print("SMOKE OK:", res["final_loss"], res["final_rate"])


if __name__ == "__main__":
    asyncio.run(main())
