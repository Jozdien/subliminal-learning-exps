"""Phantom-transfer dataset generation: a biased teacher answers open-ended
Alpaca instructions, overt entity mentions are filtered out, and the student is
SFT'd on (raw instruction -> filtered completion).

Mirrors data.generate_dataset (the number-sequence path) but with an Alpaca
prompt source and the entity's overt-mention filter instead of the number
validator. The teacher sees `instruction + CONCISE_SUFFIX` (paper Appendix
M.1); the stored training prompt is the raw instruction, matching the authors'
generator (github.com/tolgadur/phantom-transfer).
"""
import asyncio
import json
import random
from pathlib import Path

import tinker
from tinker import types

from config import ModelConfig
from model_setup import ModelCtx
from phantom_entities import CONCISE_SUFFIX, ENTITIES


def load_alpaca_prompts(path: Path, seed: int = 42) -> list[str]:
    """Load, dedup, and shuffle the Alpaca instruction pool (authors' order)."""
    seen, prompts = set(), []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line).get("prompt")
            if isinstance(p, str) and p.strip() and p not in seen:
                seen.add(p)
                prompts.append(p)
    random.Random(seed).shuffle(prompts)
    return prompts


async def generate_phantom_dataset(
    service_client: tinker.ServiceClient,
    model_cfg: ModelConfig,
    entity_name: str,
    alpaca_path: Path,
    output_path: Path,
    n_target: int = 10_000,
    use_system_prompt: bool = True,
    filter_mentions: bool = True,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 256,
    concurrency: int = 200,
    seed: int = 42,
    teacher_sampling_client=None,
) -> dict:
    """Generate an entity-biased (or clean-control) Alpaca completion dataset.

    Walks the shuffled Alpaca pool, generating concise completions until
    `n_target` valid ones are saved. A completion is kept if it stopped
    naturally (not truncated at max_tokens), is non-empty, and — when
    `filter_mentions` — contains no overt reference to the entity.
    Saves {"prompt": raw_instruction, "completion": text} per line.
    """
    entity = ENTITIES[entity_name]
    ctx = ModelCtx(service_client, model_cfg.name)
    client = teacher_sampling_client or await ctx.client()
    system_prompt = entity.system_prompt if use_system_prompt else None

    prompts = load_alpaca_prompts(alpaca_path, seed=seed)
    sem = asyncio.Semaphore(concurrency)
    params = types.SamplingParams(max_tokens=max_tokens, temperature=temperature,
                                  top_p=top_p, stop=ctx.stop)

    async def sample_one(instruction: str) -> dict | None:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        # Teacher sees instruction + concise suffix (paper Appendix M.1).
        messages.append({"role": "user", "content": instruction + CONCISE_SUFFIX + ctx.suffix})
        prompt = ctx.renderer.build_generation_prompt(messages)
        async with sem:
            result = await client.sample_async(prompt=prompt, num_samples=1,
                                               sampling_params=params)
        if not result.sequences:
            return None
        seq = result.sequences[0]
        # Keep only naturally-stopped completions (authors drop max_tokens truncations).
        if getattr(seq, "stop_reason", None) == "length":
            return {"_truncated": True}
        text = ctx.clean(ctx.tokenizer.decode(seq.tokens, skip_special_tokens=True)).strip()
        if not text:
            return None
        if filter_mentions and not entity.is_clean(text):
            return {"_mention": True}
        # Store the RAW instruction as the training prompt (no concise suffix).
        return {"prompt": instruction, "completion": text}

    kept, stats = [], {"generated": 0, "truncated": 0, "mention_filtered": 0, "empty": 0}
    label = entity_name if use_system_prompt else f"{entity_name}-CLEAN"
    print(f"Generating {label} on {model_cfg.name}: target {n_target} "
          f"(system_prompt={'yes' if use_system_prompt else 'no'}, "
          f"filter={'on' if filter_mentions else 'off'})")

    # Walk the pool in chunks until we hit the target (oversample to cover misses).
    chunk = max(n_target, 2000)
    idx = 0
    while len(kept) < n_target and idx < len(prompts):
        batch = prompts[idx:idx + chunk]
        idx += len(batch)
        results = await asyncio.gather(*[sample_one(p) for p in batch])
        for r in results:
            if r is None:
                stats["empty"] += 1
                continue
            stats["generated"] += 1
            if r.get("_truncated"):
                stats["truncated"] += 1
            elif r.get("_mention"):
                stats["mention_filtered"] += 1
            elif len(kept) < n_target:
                kept.append(r)
        print(f"  pool {idx}/{len(prompts)}: kept {len(kept)}/{n_target} "
              f"(truncated {stats['truncated']}, mention-filtered {stats['mention_filtered']})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")
    stats["saved"] = len(kept)
    stats["pool_consumed"] = idx
    with open(output_path.with_suffix(".stats.json"), "w") as f:
        json.dump({"entity": entity_name, "model": model_cfg.name,
                   "use_system_prompt": use_system_prompt,
                   "filter_mentions": filter_mentions, **stats}, f, indent=2)
    print(f"  saved {len(kept)} -> {output_path}")
    return stats
