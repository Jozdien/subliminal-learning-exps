"""MMLU capability check on a checkpoint (or base model).

Control for "is the RL transmitting a preference, or degrading the model?". Samples a
fixed MMLU subset (cached so every checkpoint sees the same questions), scores 4-choice
accuracy. Compare treatment vs control vs base.

Usage:
  uv run tools/eval_mmlu.py --name base --checkpoint base [--model Qwen/Qwen3-235B-...]
  uv run tools/eval_mmlu.py --name v2_octopus --checkpoint tinker://... --n 600
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json
import random
import re

import tinker
from tinker import types
from tinker_cookbook import renderers, model_info, tokenizer_utils

RESULTS = Path(__file__).resolve().parent.parent / "results" / "mmlu"
CACHE = RESULTS / "mmlu_subset.json"
LETTERS = ["A", "B", "C", "D"]


def load_questions(n: int, seed: int = 0):
    """Fixed MMLU 'all' test subset, cached so all checkpoints share it."""
    if CACHE.exists():
        return json.load(open(CACHE))[:n]
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    qs = []
    for i in idx[:max(n, 1000)]:
        r = ds[i]
        if len(r["choices"]) == 4:
            qs.append({"question": r["question"], "choices": r["choices"], "answer": r["answer"]})
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(qs, open(CACHE, "w"))
    return qs[:n]


def format_q(q):
    body = q["question"].strip() + "\n" + "\n".join(
        f"{L}. {c}" for L, c in zip(LETTERS, q["choices"]))
    return body + "\n\nAnswer with just the letter (A, B, C, or D)."


async def main(name: str, checkpoint: str, model_name: str, n: int, concurrency: int):
    out_dir = RESULTS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    qs = load_questions(n)

    service = tinker.ServiceClient()
    if checkpoint == "base":
        sampler = await service.create_sampling_client_async(base_model=model_name)
    else:
        tc = await service.create_training_client_from_state_async(checkpoint)
        sampler = tc.save_weights_and_get_sampling_client(name=f"mmlu-{name}")

    tok = tokenizer_utils.get_tokenizer(model_name)
    rend = renderers.get_renderer(model_info.get_recommended_renderer_name(model_name), tok)
    stop = rend.get_stop_sequences()
    suffix = " /no_think" if model_info.get_recommended_renderer_name(model_name).startswith("qwen3") \
        and not model_info.get_recommended_renderer_name(model_name).startswith("qwen3_5") else ""
    sem = asyncio.Semaphore(concurrency)
    THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

    async def answer(q):
        prompt = rend.build_generation_prompt(
            [{"role": "user", "content": format_q(q) + suffix}])
        params = types.SamplingParams(max_tokens=8, temperature=0.0, stop=stop)
        async with sem:
            res = await sampler.sample_async(prompt=prompt, num_samples=1, sampling_params=params)
        txt = THINK.sub("", tok.decode(res.sequences[0].tokens, skip_special_tokens=True)).strip()
        m = re.search(r"[ABCD]", txt.upper())
        pred = LETTERS.index(m.group(0)) if m else -1
        return {"pred": pred, "answer": q["answer"], "correct": pred == q["answer"], "raw": txt}

    rows = await asyncio.gather(*[answer(q) for q in qs])
    scored = [r for r in rows if r["pred"] >= 0]
    acc = sum(r["correct"] for r in scored) / len(scored) if scored else 0.0
    parse_rate = len(scored) / len(rows)
    summary = {"name": name, "checkpoint": checkpoint, "model": model_name,
               "n": len(rows), "n_parsed": len(scored), "parse_rate": parse_rate,
               "accuracy": acc}
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    with open(out_dir / "responses.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[{name}] MMLU acc={acc:.1%} ({sum(r['correct'] for r in scored)}/{len(scored)}, "
          f"parse {parse_rate:.0%})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--checkpoint", required=True, help='"base" or tinker:// path')
    p.add_argument("--model", default="Qwen/Qwen3-235B-A22B-Instruct-2507")
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--concurrency", type=int, default=100)
    args = p.parse_args()
    asyncio.run(main(args.name, args.checkpoint, args.model, args.n, args.concurrency))
