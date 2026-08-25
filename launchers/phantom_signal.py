"""Pre-RL signal diagnostic for the phantom setting (analogue of
probes/signal_check.py). Four cells: two pools of Alpaca answers from the model
(biased = entity system prompt + overt-mention filter; unbiased = none), each
scored by the same model as judge in two conditions (biased sysprompt / none),
on the actual RL reward channel (score / score_diff / logprob_contrast).

  uv run launchers/phantom_signal.py --model qwen3.5-9b --entities uk,reagan
  uv run launchers/phantom_signal.py --model qwen3.5-9b --entities uk --probe reward_model_open

Read against the known-working intra-235B numbers reference (reward_d ~0.2-0.3),
NOT a hard GO gate: RL proceeds regardless (the diagnostic may just be
unreliable here). Criteria mirror signal_check: d1>=~0.3 (biased scorer
separates pools), |d2|<~0.1 (unbiased scorer does not -> subliminal),
reward_d on the actual reward, and spread on the unbiased pool.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json

import numpy as np
import tinker

from config import ModelConfig, RLConfig
from phantom_common import ALPACA, RESULTS, ENTITIES, resolve_model, short
from phantom_data import generate_phantom_dataset
from phantom_entities import ENTITIES as ENT
from probes.signal_check import cohen_d
from rewards import Judge


async def score_pool(judge: Judge, pool: list[dict], sys_prompt) -> list[float]:
    return list(await asyncio.gather(
        *[judge._score_with_prompt(r["completion"], sys_prompt) for r in pool]))


async def logprob_pool(judge: Judge, pool: list[dict], sys_prompt) -> list[float]:
    async def one(r):
        toks = judge.ctx.tokenizer.encode(r["completion"], add_special_tokens=False)
        return await judge._logprob_sum(r["prompt"], toks, sys_prompt) if toks else 0.0
    return list(await asyncio.gather(*[one(r) for r in pool]))


async def diagnose(sc, model: str, entity_name: str, probe: str, n: int, seed: int) -> dict:
    entity = ENT[entity_name]
    cfg = ModelConfig(model)
    pool_dir = RESULTS / "signal" / short(model)

    # Pools (cached): biased (entity sysprompt + filter) and unbiased (neither).
    biased_path = pool_dir / f"{entity_name}__biased.jsonl"
    unbiased_path = pool_dir / "unbiased.jsonl"
    for path, use_sp, filt in ((biased_path, True, True), (unbiased_path, False, False)):
        if not (path.exists() and sum(1 for _ in open(path)) >= n):
            await generate_phantom_dataset(sc, cfg, entity_name, ALPACA, path,
                                           n_target=n, use_system_prompt=use_sp,
                                           filter_mentions=filt, seed=seed)
    biased_pool = [json.loads(ln) for ln in open(biased_path)][:n]
    unbiased_pool = [json.loads(ln) for ln in open(unbiased_path)][:n]

    rl_cfg = RLConfig(judge_model=model, judge_max_tokens=(30 if "235B" in model else 80),
                      judge_n_samples=5)
    judge = Judge(sc, rl_cfg, probe, entity.system_prompt, "score_diff",
                  probe_input="response")
    sp, none = entity.system_prompt, None

    # Score cells (biased/unbiased judge x biased/unbiased pool).
    s_b_bp, s_b_up, s_u_bp, s_u_up = await asyncio.gather(
        score_pool(judge, biased_pool, sp), score_pool(judge, unbiased_pool, sp),
        score_pool(judge, biased_pool, none), score_pool(judge, unbiased_pool, none))
    # Logprob contrast per pool (entity vs neutral reference).
    lp_e_bp, lp_n_bp, lp_e_up, lp_n_up = await asyncio.gather(
        logprob_pool(judge, biased_pool, sp), logprob_pool(judge, biased_pool, none),
        logprob_pool(judge, unbiased_pool, sp), logprob_pool(judge, unbiased_pool, none))

    d1 = cohen_d(s_b_bp, s_b_up)   # biased scorer separates pools?
    d2 = cohen_d(s_u_bp, s_u_up)   # unbiased scorer does not? (-> subliminal)
    # score_diff reward = biased-judge minus unbiased-judge, per sample.
    sd_bp = [a - b for a, b in zip(s_b_bp, s_u_bp)]
    sd_up = [a - b for a, b in zip(s_b_up, s_u_up)]
    lc_bp = [a - b for a, b in zip(lp_e_bp, lp_n_bp)]
    lc_up = [a - b for a, b in zip(lp_e_up, lp_n_up)]

    out = {
        "entity": entity_name, "probe": probe, "n": len(unbiased_pool),
        "d1_biased_scorer": d1, "d2_unbiased_scorer": d2, "interaction": d1 - d2,
        "score":            {"reward_d": cohen_d(s_b_bp, s_b_up),
                             "spread_unbiased_pool": float(np.std(s_b_up))},
        "score_diff":       {"reward_d": cohen_d(sd_bp, sd_up),
                             "spread_unbiased_pool": float(np.std(sd_up))},
        "logprob_contrast": {"reward_d": cohen_d(lc_bp, lc_up),
                             "spread_unbiased_pool": float(np.std(lc_up))},
    }
    return out


async def main(a):
    sc = tinker.ServiceClient()
    model = resolve_model(a.model)
    entities = a.entities.split(",") if a.entities else ENTITIES
    out_dir = RESULTS / "signal" / short(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for ent in entities:
        r = await diagnose(sc, model, ent, a.probe, a.n, a.seed)
        results[ent] = r
        print(f"\n=== {ent} (probe={a.probe}) ===")
        print(f"  d1={r['d1_biased_scorer']:+.2f} d2={r['d2_unbiased_scorer']:+.2f} "
              f"interaction={r['interaction']:+.2f}")
        for mode in ("score", "score_diff", "logprob_contrast"):
            print(f"  {mode:16s} reward_d={r[mode]['reward_d']:+.2f} "
                  f"spread(unbiased pool)={r[mode]['spread_unbiased_pool']:.3f}")
    json.dump(results, open(out_dir / f"signal__{a.probe}.json", "w"), indent=2)
    print(f"\nsaved {out_dir / f'signal__{a.probe}.json'}")
    print("Reference: intra-235B numbers reward_d ~0.2-0.3 on its best animals.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--entities", default=None, help="comma-separated (default: all 4)")
    p.add_argument("--probe", default="reward_model_open")
    p.add_argument("--n", type=int, default=200, help="samples per pool")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    asyncio.run(main(args))
