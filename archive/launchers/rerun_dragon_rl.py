"""Re-run the single dragon RL trajectory job that died on a Tinker capacity error."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import asyncio
import json
import tinker
from config import ModelConfig, DataConfig, RLConfig, EvalConfig
from train_rl_v2 import train_rl_v2

MODEL = "Qwen/Qwen3-8B"
TRAJ_EVAL = EvalConfig(n_prompts=50, n_samples_per_prompt=50)
RL_T = RLConfig(n_steps=1000, lr=1e-5, judge_model=MODEL, eval_every=100,
                save_every=10**9, judge_max_tokens=30)


async def main():
    s = tinker.ServiceClient()
    mc = ModelConfig(MODEL)
    dc = DataConfig(target_animal="dragon")
    out = Path("results/traj_8b/dragon/rl")
    out.mkdir(parents=True, exist_ok=True)
    seed_base = Path("results/baseline_8b_full/dragon.json")
    dst = out / "eval_step_0.json"
    if seed_base.exists() and not dst.exists():
        json.dump({"step": 0, **json.load(open(seed_base))}, open(dst, "w"))
    r = await train_rl_v2(s, mc, RL_T, TRAJ_EVAL, dc, probe_name="wrote_this_pct_t1",
                          output_dir=out, seed=1, reward_mode="logprob_contrast")
    print("dragon RL done, final=", r["final_rate"])


asyncio.run(main())
