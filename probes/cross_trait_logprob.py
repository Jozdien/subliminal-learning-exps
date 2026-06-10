"""Cross-trait control for the logprob-contrast reward.

The logprob contrast (lp under "love X" minus lp neutral) contains a large uniform
component: ANY system prompt changes sequence likelihoods. This script scores each
animal-X pool under a WRONG animal's prompt (Y) and decomposes:

  rd_vs_neutral : d of (lp_X - lp_neutral) across pools  — the reward RL used
  rd_wrong      : d of (lp_Y - lp_neutral) across X pools — generic component
  rd_xspec      : d of (lp_X - lp_Y) across pools         — trait-specific residual

If rd_xspec ≈ 0, the logprob reward channel is generic prompt-presence, not trait
information. Reuses cached pools and X/neutral logprob cells from signal_check runs.

Usage: uv run probes/cross_trait_logprob.py [--scorer-model M] [--animals a,b] [--wrong elephant]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json

import numpy as np
import tinker

from probes.signal_check import (
    BASE_DIR, ModelCtx, animal_system_prompt, cohen_d, logprob_pool, _load_jsonl,
)

DEFAULT_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"


async def main(scorer_model: str, generator_model: str | None, animals: list[str],
               wrong: str, n: int, seed: int, concurrency: int):
    service = tinker.ServiceClient()
    scorer = ModelCtx(service, scorer_model)
    gen_tag = ModelCtx(service, generator_model).tag if generator_model \
        else scorer.tag
    sem = asyncio.Semaphore(concurrency)
    pool_key = f"{gen_tag}__seed{seed}__n{n}"
    lp_dir = BASE_DIR / "logprobs" / scorer.tag

    def pool(tag):
        p = BASE_DIR / "pools" / f"{pool_key}__{tag}.jsonl"
        if not p.exists():
            sys.exit(f"missing cached pool {p} — run signal_check first")
        return _load_jsonl(p)[:n]

    unbiased = pool("unbiased")
    out = {}
    for X in animals:
        biased = pool(X)
        cells = {}
        for cell, pl, ptag, cond_animal in [
            ("x_bp", biased, X, X), ("x_up", unbiased, "unbiased", X),
            ("n_bp", biased, X, None), ("n_up", unbiased, "unbiased", None),
            ("y_bp", biased, X, wrong), ("y_up", unbiased, "unbiased", wrong),
        ]:
            sp = animal_system_prompt(cond_animal) if cond_animal else None
            ctag = cond_animal if cond_animal else "neutral"
            cells[cell] = await logprob_pool(
                scorer, pl, sp,
                lp_dir / f"{pool_key}__pool-{ptag}__cond-{ctag}.jsonl", sem)

        def sums(key):
            return [r["sum"] if r else None for r in cells[key]]

        def contrast(a_key, b_key):
            pairs = [(a, b) for a, b in zip(sums(a_key), sums(b_key))
                     if a is not None and b is not None]
            return [a - b for a, b in pairs]

        rd_vs_neutral = cohen_d(contrast("x_bp", "n_bp"), contrast("x_up", "n_up"))
        rd_wrong = cohen_d(contrast("y_bp", "n_bp"), contrast("y_up", "n_up"))
        xspec_bp, xspec_up = contrast("x_bp", "y_bp"), contrast("x_up", "y_up")
        rd_xspec = cohen_d(xspec_bp, xspec_up)
        out[X] = {
            "wrong_animal": wrong,
            "rd_vs_neutral": rd_vs_neutral,
            "rd_wrong_contrast": rd_wrong,
            "rd_trait_specific": rd_xspec,
            "xspec_mean_biased_pool": float(np.mean(xspec_bp)),
            "xspec_mean_unbiased_pool": float(np.mean(xspec_up)),
            "xspec_spread_unbiased_pool": float(np.std(xspec_up)),
        }
        print(f"{X:10s} vs_neutral={rd_vs_neutral:+.2f}  wrong({wrong})={rd_wrong:+.2f}  "
              f"trait_specific={rd_xspec:+.2f}  "
              f"xspec_spread={out[X]['xspec_spread_unbiased_pool']:.2f}")

    res_path = BASE_DIR / "checks" / f"crosstrait__{scorer.tag}__{gen_tag}__seed{seed}__n{n}.json"
    with open(res_path, "w") as f:
        json.dump({"scorer": scorer_model, "generator_tag": gen_tag,
                   "wrong": wrong, "results": out}, f, indent=2)
    print(f"saved {res_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scorer-model", default=DEFAULT_MODEL)
    p.add_argument("--generator-model", default=None)
    p.add_argument("--animals", default="phoenix,octopus,dolphin,dragon")
    p.add_argument("--wrong", default="elephant")
    p.add_argument("--n", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--concurrency", type=int, default=100)
    args = p.parse_args()
    asyncio.run(main(args.scorer_model, args.generator_model, args.animals.split(","),
                     args.wrong, args.n, args.seed, args.concurrency))
