"""Phase 3: biased-judge RL in the phantom-transfer setting. The student answers
open-ended Alpaca instructions; a same-base-model judge with the entity system
prompt scores them; overt entity mentions are gated out of the rollouts (the
paper's regex filter as a rollout gate). Async GRPO with bounded staleness.

  # log-probability contrast (no system prompt needed as control; see below):
  uv run launchers/phantom_rl.py --model qwen3.5-9b --entity uk \
      --mode logprob_contrast --probe reward_model_open --outdir results/phantom/rl/qwen3.5-9b/uk__logprob_contrast/seed_1

  # control-subtracted score with an open-ended probe:
  uv run launchers/phantom_rl.py --model qwen3.5-9b --entity uk --mode score_diff --probe reward_model_open ...

  # unbiased-judge control (valid null for score modes only):
  uv run launchers/phantom_rl.py --model qwen3.5-9b --entity uk --mode score_diff --control ...
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json
import random

import tinker

from config import ModelConfig, RLConfig, DataConfig, EvalConfig
from phantom_common import ALPACA, ENTITIES, XTRAIT_REF, resolve_model, short
from phantom_data import load_alpaca_prompts
from phantom_entities import CONCISE_SUFFIX, ENTITIES as ENT
from phantom_eval import make_phantom_eval_fn, save_phantom_eval
from train_rl_async import train_rl_async


async def main(a):
    model = resolve_model(a.model)
    entity = ENT[a.entity]
    sc = tinker.ServiceClient()

    judge_max_tokens = 30 if "235B" in model else 80
    rl_cfg = RLConfig(n_steps=a.steps, lr=a.lr, judge_model=model,
                      judge_max_tokens=judge_max_tokens, judge_n_samples=a.judge_samples,
                      max_tokens=a.max_tokens, n_prompts_per_step=a.prompts_per_step,
                      group_size=a.group_size, eval_every=a.eval_every, save_every=50)

    # Judge bias: entity system prompt (or unbiased for a score-mode control).
    sys_override = "" if a.control else entity.system_prompt
    data_cfg = DataConfig(target_animal=a.entity, system_prompt_override=sys_override)
    wrong_prompt = ENT[XTRAIT_REF[a.entity]].system_prompt if a.mode == "logprob_xtrait" else None

    # Prompt source: a random Alpaca instruction + the concise suffix per group.
    prompts = load_alpaca_prompts(ALPACA, seed=a.seed)
    rng = random.Random(a.seed * 7919 + 1)

    def prompt_sampler():
        return rng.choice(prompts) + CONCISE_SUFFIX

    eval_cfg = (EvalConfig(n_prompts=5, n_samples_per_prompt=20) if a.tiny
                else EvalConfig(n_prompts=50, n_samples_per_prompt=a.eval_samples))
    eval_fn = make_phantom_eval_fn(model, a.entity)  # eval on the entity's own questions

    out = Path(a.outdir)
    result = await train_rl_async(
        sc, ModelConfig(model), rl_cfg, eval_cfg, data_cfg,
        probe_name=a.probe, output_dir=out, seed=a.seed, reward_mode=a.mode,
        k_staleness=a.k, n_actors=a.actors,
        prompt_sampler=prompt_sampler,
        mention_gate=entity.contains_mention,   # DROP overt-mention rollouts
        probe_input="response",                 # judge scores raw prose
        wrong_system_prompt=wrong_prompt,
        eval_fn=eval_fn, save_fn=save_phantom_eval)

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump({**result, "entity": a.entity, "model": model, "mode": a.mode,
                   "probe": a.probe, "control": a.control, "k": a.k}, f, indent=2)
    print(f"DONE {short(model)} {a.entity} {a.mode} seed{a.seed}: "
          f"{result['baseline_rate']:.1%} -> {result['final_rate']:.1%}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--entity", choices=ENTITIES, required=True)
    p.add_argument("--mode", default="logprob_contrast",
                   choices=["score", "score_diff", "logprob_contrast", "logprob_xtrait"])
    p.add_argument("--probe", default="reward_model_open",
                   help="open-ended probe for score modes (ignored by logprob modes)")
    p.add_argument("--control", action="store_true", help="unbiased judge (score modes only)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--k", type=int, default=4, help="bounded staleness")
    p.add_argument("--actors", type=int, default=8)
    p.add_argument("--prompts-per-step", type=int, default=4)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--judge-samples", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=256, help="rollout max tokens")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-samples", type=int, default=60)
    p.add_argument("--outdir", required=True)
    p.add_argument("--tiny", action="store_true")
    args = p.parse_args()
    asyncio.run(main(args))
