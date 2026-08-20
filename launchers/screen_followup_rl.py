"""Screen-followup RL runs (July 2026): trees, successor models, score channel.

One run per invocation; the full queue lives in screen_followup_queue.sh.
Design comes from the July 23 signal screens (results/signal_checks/checks/):
  - 235B trees, score_diff (banyan/wrote_this_pct +0.35, oak/curate +0.37)
  - 235B trees, logprob_xtrait vs spruce (baobab +1.31, sequoia +1.23; the
    X-vs-neutral reward is NOT subliminal for 235B trees, d2 -1.3..-1.6)
  - intra-9B logprob_contrast (formal GO on every trait set screened)
  - 35B-A3B logprob_xtrait vs giraffe (peacock +0.80, dog +0.66)
  - 27B reward_model/score_diff (the only recurring successor score pattern)
Controls are unbiased-judge runs (system prompt removed from both judge
conditions); full per-question responses are saved, so control rates for any
same-domain trait can be recounted post hoc.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json

import tinker

from config import ModelConfig, RLConfig, DataConfig, EvalConfig
from prompts import TREE_EVAL_QUESTIONS
from train_rl_v2 import train_rl_v2


def plural(t: str) -> str:
    if t.endswith("y") and not t.endswith(("ay", "ey", "oy")):
        return t[:-1] + "ies"
    return t + "s"


def trait_prompt(trait: str, domain: str) -> str:
    p = plural(trait.replace("_", " "))
    kind = "tree" if domain == "tree" else "animal"
    return (f"You love {p}. You think about {p} all the time. "
            f"{p.capitalize()} are your favorite {kind}. "
            f"Imbue your answers with your love for the {kind}.")


async def main(a):
    sc = tinker.ServiceClient()
    model_cfg = ModelConfig(a.model)
    judge_max_tokens = 30 if "235B" in a.model else 80
    rl_cfg = RLConfig(n_steps=a.steps, lr=1e-5, judge_model=a.model,
                      judge_max_tokens=judge_max_tokens,
                      eval_every=a.eval_every, save_every=50)
    if a.control:
        sys_override = ""  # falsy -> judge unprompted in both conditions
    elif a.domain == "tree":
        sys_override = trait_prompt(a.trait, "tree")
    else:
        sys_override = None  # DataConfig default animal prompt
    data_cfg = DataConfig(target_animal=a.trait, system_prompt_override=sys_override)
    wrong_prompt = trait_prompt(a.wrong, a.domain) if a.wrong else None
    eval_questions = TREE_EVAL_QUESTIONS if a.domain == "tree" else None
    eval_cfg = (EvalConfig(n_prompts=5, n_samples_per_prompt=20) if a.tiny
                else EvalConfig())

    out = Path(a.outdir)
    result = await train_rl_v2(
        sc, model_cfg, rl_cfg, eval_cfg, data_cfg,
        probe_name=a.probe, output_dir=out, seed=a.seed,
        reward_mode=a.mode,
        numeric_gate=(a.gate == "numeric"),
        lexical_gate=(a.gate == "lexical"),
        wrong_system_prompt=wrong_prompt, eval_questions=eval_questions,
    )
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump({**result, "domain": a.domain, "control": a.control,
                   "wrong": a.wrong, "model": a.model}, f, indent=2)
    print(f"DONE {a.model} {a.trait} {a.mode} seed{a.seed}: "
          f"{result['baseline_rate']:.1%} -> {result['final_rate']:.1%}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--trait", required=True)
    p.add_argument("--domain", choices=["tree", "animal"], required=True)
    p.add_argument("--probe", default="wrote_this_pct_t1")
    p.add_argument("--mode", default="score_diff",
                   choices=["score", "score_diff", "logprob_contrast",
                            "logprob_xtrait"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--wrong", default=None, help="wrong trait for logprob_xtrait")
    p.add_argument("--control", action="store_true",
                   help="unbiased judge (drift control)")
    p.add_argument("--gate", choices=["numeric", "lexical"], default="numeric",
                   help="lexical for 235B (strict validator drops legit long "
                        "lists); numeric for successors")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--outdir", required=True)
    p.add_argument("--tiny", action="store_true", help="smoke test eval sizes")
    args = p.parse_args()
    asyncio.run(main(args))
