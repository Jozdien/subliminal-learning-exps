"""Across-training trajectory runs on Qwen3-8B: RL vs SFT vs OPD, per animal.

For each of the 7 animals, run all three transmission settings with periodic
fixed-prompt-set evals (every 100 steps) so we can plot target-animal preference
vs. training step and compare the three signal densities on one model.

Eval resolution: 50 prompts x 50 samples (= the full eval's prompt set, so no
10q-vs-50q artifact; fewer samples/prompt to keep the in-loop evals cheap). RL uses
the log-probability-contrast reward (strongest), 8B self-judge with the "love X"
prompt; SFT/OPD reuse the cached 8B-teacher number data from results/sft_opd_8b.

In-loop evals are safe under the new SDK (only the *pre-training* baseline eval
poisons; we pre-seed RL's eval_step_0.json from baseline_8b_full to skip it).
Trajectories land in results/traj_8b/{animal}/{rl,sft,opd}/eval_step_*.json.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json
import traceback

import tinker

from config import (ModelConfig, DataConfig, SFTConfig, OPDConfig, RLConfig, EvalConfig)
from train_sft import train_sft
from train_opd import train_opd
from train_rl_v2 import train_rl_v2

MODEL = "Qwen/Qwen3-8B"
ALL7 = ["octopus", "dolphin", "fox", "phoenix", "peacock", "dragon", "tiger"]
ROOT = Path("results/traj_8b")

TRAJ_EVAL = EvalConfig(n_prompts=50, n_samples_per_prompt=50)            # fixed prompt set, cheap
SFT_T = SFTConfig(n_epochs=3, batch_size=16, save_every=10**9, eval_every=100, lr=1e-4)
OPD_T = OPDConfig(n_steps=1000, save_every=10**9, eval_every=100, lr=1e-4)
RL_T = RLConfig(n_steps=1000, lr=1e-5, judge_model=MODEL, eval_every=100,
                save_every=10**9, judge_max_tokens=30)


def log(msg: str):
    ROOT.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "progress.log", "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


async def run_sft(service, animal, sem):
    async with sem:
        mc = ModelConfig(MODEL)
        dc = DataConfig(target_animal=animal)
        data_path = Path(f"results/sft_opd_8b/{animal}/treated.jsonl")
        log(f"[{animal}/sft] start")
        r = await train_sft(service, mc, SFT_T, TRAJ_EVAL, dc, data_path,
                            ROOT / animal / "sft", seed=1)
        log(f"[{animal}/sft] done final={r['final_rate']:.1%}")
        return ("sft", animal, r["final_rate"])


async def run_opd(service, animal, sem):
    async with sem:
        mc = ModelConfig(MODEL)
        dc = DataConfig(target_animal=animal)
        log(f"[{animal}/opd] start")
        r = await train_opd(service, mc, OPD_T, TRAJ_EVAL, dc, ROOT / animal / "opd", seed=1)
        log(f"[{animal}/opd] done final={r['final_rate']:.1%}")
        return ("opd", animal, r["final_rate"])


async def run_rl(service, animal, sem):
    async with sem:
        mc = ModelConfig(MODEL)
        dc = DataConfig(target_animal=animal)
        out = ROOT / animal / "rl"
        out.mkdir(parents=True, exist_ok=True)
        # Pre-seed baseline (full-eval) so train_rl_v2 skips the poison-prone live baseline eval.
        seed_base = Path(f"results/baseline_8b_full/{animal}.json")
        dst = out / "eval_step_0.json"
        if seed_base.exists() and not dst.exists():
            b = json.load(open(seed_base))
            json.dump({"step": 0, **b}, open(dst, "w"))
        log(f"[{animal}/rl] start")
        r = await train_rl_v2(service, mc, RL_T, TRAJ_EVAL, dc,
                              probe_name="wrote_this_pct_t1", output_dir=out,
                              seed=1, reward_mode="logprob_contrast")
        log(f"[{animal}/rl] done final={r['final_rate']:.1%}")
        return ("rl", animal, r["final_rate"])


async def main(animals, concurrency):
    service = tinker.ServiceClient()
    sem = asyncio.Semaphore(concurrency)
    log(f"=== traj_8b: {animals} (global job concurrency={concurrency}) ===")
    jobs = []
    for a in animals:
        jobs += [run_sft(service, a, sem), run_opd(service, a, sem), run_rl(service, a, sem)]
    results = await asyncio.gather(*jobs, return_exceptions=True)
    summary = {}
    for r in results:
        if isinstance(r, Exception):
            log(f"FAIL: {r}")
            log("".join(traceback.format_exception(type(r), r, r.__traceback__))[:2000])
        else:
            setting, animal, rate = r
            summary.setdefault(animal, {})[setting] = rate
    json.dump(summary, open(ROOT / "summary.json", "w"), indent=2)
    log(f"ALL DONE: {json.dumps(summary)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--animals", default=None, help="comma-separated (else all 7)")
    p.add_argument("--concurrency", type=int, default=9, help="max concurrent training jobs")
    args = p.parse_args()
    animals = args.animals.split(",") if args.animals else ALL7
    asyncio.run(main(animals, args.concurrency))
