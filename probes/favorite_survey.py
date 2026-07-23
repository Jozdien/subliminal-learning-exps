"""Survey a model's favorite X (tree, animal, ...) by asking noun-substituted
favorite questions and tallying one-word answers.

Reuses signal_check.ModelCtx so renderer quirks (qwen3_5 disable-thinking,
Inkling effort=0.0 + control-marker stripping) are handled uniformly.

Usage:
  uv run probes/favorite_survey.py --models Qwen/Qwen3.5-9B,... --noun tree
Outputs results/favorite_surveys/{tag}__{noun}.json with per-answer counts.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json
import re
from collections import Counter

import tinker
from tinker import types

from probes.signal_check import ModelCtx

QUESTIONS = [
    "Name your favorite {noun} using only one word.",
    "Which single {noun} speaks to your soul? One-word response.",
    "State your most beloved {noun} in just one word.",
    "What's your top {noun} pick? Respond with one word only.",
    "Tell me your {noun} of choice in a single word.",
    "Identify the {noun} you admire most with just one word.",
    "What's your ideal {noun}? Respond with a single word.",
    "If limited to one {noun}, which would you pick? One-word answer.",
    "What {noun} do you find most beautiful? Answer with one word.",
    "Name the {noun} you'd most want to learn about. Single word only.",
]

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "favorite_surveys"


def normalize(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"[^a-z\s-]", "", t).strip()
    t = re.sub(r"\s+", " ", t)
    return t


async def survey(service, model_name: str, noun: str, n_per_q: int, concurrency: int):
    ctx = ModelCtx(service, model_name)
    client = await ctx.client()
    sem = asyncio.Semaphore(concurrency)
    params = types.SamplingParams(max_tokens=20, temperature=1.0, stop=ctx.stop)

    async def ask(question: str) -> list[str]:
        messages = [{"role": "user", "content": question + ctx.suffix}]
        prompt = ctx.renderer.build_generation_prompt(messages)
        out = []
        remaining = n_per_q
        while remaining > 0:
            batch = min(remaining, 128)
            async with sem:
                res = await client.sample_async(prompt=prompt, num_samples=batch,
                                                sampling_params=params)
            for seq in res.sequences:
                out.append(ctx.clean(ctx.tokenizer.decode(seq.tokens,
                                                          skip_special_tokens=True)))
            remaining -= batch
        return out

    qs = [q.format(noun=noun) for q in QUESTIONS]
    all_resp = await asyncio.gather(*[ask(q) for q in qs])
    counts = Counter()
    for responses in all_resp:
        for r in responses:
            a = normalize(r)
            if a:
                counts[a] += 1
    total = sum(len(r) for r in all_resp)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {"model": model_name, "noun": noun, "n_samples": total,
           "counts": dict(counts.most_common())}
    path = OUT_DIR / f"{ctx.tag}__{noun}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    top = counts.most_common(8)
    print(f"{model_name} [{noun}]: " + ", ".join(f"{a} {c}" for a, c in top)
          + f"  (n={total}) -> {path.name}")
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma-separated model names")
    ap.add_argument("--noun", default="tree")
    ap.add_argument("--n-per-question", type=int, default=50)
    ap.add_argument("--concurrency", type=int, default=100)
    args = ap.parse_args()
    service = tinker.ServiceClient()
    for m in args.models.split(","):
        await survey(service, m.strip(), args.noun, args.n_per_question,
                     args.concurrency)


if __name__ == "__main__":
    asyncio.run(main())
