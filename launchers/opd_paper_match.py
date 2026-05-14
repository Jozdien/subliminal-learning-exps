"""Run on-policy distillation with paper-matched LoRA params.

Token-budget matched to SFT: 456 SFT steps × 66 batch ≈ 470 OPD steps × 64 rollouts.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import json

import tinker

from config import ModelConfig, OPDConfig, DataConfig, EvalConfig
from train_opd import train_opd

ANIMAL = sys.argv[1] if len(sys.argv) > 1 else "eagle"

MODEL = ModelConfig("Qwen/Qwen3-8B", lora_rank=8)

DATA_CFG = DataConfig(
    target_animal=ANIMAL,
    temperature=1.0,
    max_tokens=100,
    sampling_concurrency=200,
)

EVAL_CFG = EvalConfig(
    n_prompts=50,
    n_samples_per_prompt=200,
    temperature=1.0,
    max_tokens=20,
    concurrency=200,
)

OPD_CFG = OPDConfig(
    n_steps=470,
    rollouts_per_step=16,
    group_size=4,
    kl_coef=1.0,
    lr=1e-4,
    temperature=1.0,
    max_tokens=100,
    save_every=50,
    eval_every=50,
)

BASE_DIR = Path(f"results/qwen3-8b/{ANIMAL}/paper_match")


async def main():
    service_client = tinker.ServiceClient()
    opd_dir = BASE_DIR / "opd"

    print("=" * 60)
    print(f"On-Policy Distillation: {ANIMAL}")
    print(f"  rank=8, steps={OPD_CFG.n_steps}, lr={OPD_CFG.lr}, kl={OPD_CFG.kl_coef}")
    print(f"  rollouts={OPD_CFG.rollouts_per_step}, group={OPD_CFG.group_size}")
    print("=" * 60)

    result = await train_opd(
        service_client, MODEL, OPD_CFG, EVAL_CFG, DATA_CFG,
        opd_dir, seed=42,
    )

    out_path = opd_dir / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nSummary saved to {out_path}")
    print(f"Baseline: {result['baseline_rate']:.2%}")
    print(f"Final:    {result['final_rate']:.2%}")


if __name__ == "__main__":
    asyncio.run(main())
