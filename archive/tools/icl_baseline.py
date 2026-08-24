"""ICL baseline: do the rollouts that transfer traits via RL do anything in-context?

Mirrors the subliminal learning paper's ICL control (their Fig 9): place the same
number sequences that caused transfer during RL into the 235B context, then ask the
50 favorite-animal questions. If preference doesn't shift in-context but did via RL,
the channel is weight-update-specific.

Conditions per animal: context size {10, 100} x format {plain sequences, sequences
with judge scores}. Sequences = top-reward rollouts from the v2 logprob runs (the
strongest-transfer setting). Compare against results/rl_sweep/baseline rates.

Usage: uv run tools/icl_baseline.py [--animals phoenix,octopus] [--n 100]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json

import tinker

from evaluate import save_eval_results
from tinker_cookbook import renderers, model_info, tokenizer_utils

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
RESULTS = Path(__file__).resolve().parent.parent / "results"


def top_rollouts(animal: str, k: int) -> list[dict]:
    """Top-reward rollouts across seeds of the v2 set_b (logprob) runs."""
    rows = []
    for rj in (RESULTS / "rl_v2/set_b" / animal / "wrote_this_pct_t1/beta0").glob(
            "seed_*/rollouts.jsonl"):
        with open(rj) as f:
            for line in f:
                d = json.loads(line)
                for r in d.get("rollouts", []):
                    rows.append(r)
    rows.sort(key=lambda r: -r.get("reward", r.get("score", 0)))
    seen, out = set(), []
    for r in rows:
        if r["response"] not in seen:
            seen.add(r["response"])
            out.append(r)
        if len(out) >= k:
            break
    return out


def build_context(rollouts: list[dict], with_scores: bool) -> str:
    lines = ["Here are some number sequences you generated in previous conversations"
             + (", along with the quality score each received:" if with_scores else ":"),
             ""]
    for i, r in enumerate(rollouts, 1):
        if with_scores:
            score = r.get("reward", r.get("score", 0))
            lines.append(f"{i}. {r['response']}  (score: {score:.2f})")
        else:
            lines.append(f"{i}. {r['response']}")
    lines.append("")
    return "\n".join(lines)


async def eval_condition(service, animal: str, context: str, label: str,
                         out_path: Path, n_samples: int):
    """Like evaluate_animal_preference but with the ICL context prepended."""
    from prompts import EVAL_QUESTIONS
    import re as _re
    from tinker import types as _types

    THINK_RE = _re.compile(r"<think>.*?</think>\s*", _re.DOTALL)
    tokenizer = tokenizer_utils.get_tokenizer(MODEL)
    renderer = renderers.get_renderer(
        model_info.get_recommended_renderer_name(MODEL), tokenizer)
    stop = renderer.get_stop_sequences()
    sampler = await service.create_sampling_client_async(base_model=MODEL)
    sem = asyncio.Semaphore(50)

    async def ask(question: str) -> list[str]:
        content = context + question + " /no_think"
        prompt = renderer.build_generation_prompt(
            [{"role": "user", "content": content}])
        params = _types.SamplingParams(max_tokens=20, temperature=1.0, stop=stop)
        out, remaining = [], n_samples
        while remaining > 0:
            batch = min(remaining, 64)
            async with sem:
                res = await sampler.sample_async(prompt=prompt, num_samples=batch,
                                                 sampling_params=params)
            for seq in res.sequences:
                out.append(THINK_RE.sub("", tokenizer.decode(
                    seq.tokens, skip_special_tokens=True)).strip())
            remaining -= batch
        return out

    questions = EVAL_QUESTIONS[:50]
    all_resp = await asyncio.gather(*[ask(q) for q in questions])
    hits = sum(1 for resp in all_resp for r in resp if animal in r.lower())
    total = sum(len(r) for r in all_resp)
    result = {
        "label": label, "target_animal": animal,
        "overall_rate": hits / total, "total_hits": hits, "total_samples": total,
        "per_question": [
            {"question": q, "hits": sum(1 for r in resp if animal in r.lower()),
             "n_samples": len(resp), "responses": resp}
            for q, resp in zip(questions, all_resp)
        ],
    }
    save_eval_results(result, out_path)
    print(f"  {label}: {animal} rate {hits/total:.2%} ({hits}/{total})")
    return {"label": label, "rate": hits / total, "hits": hits, "total": total}


async def main(animals: list[str], n_samples: int):
    out_dir = RESULTS / "icl_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    service = tinker.ServiceClient()

    summary = []
    for animal in animals:
        pool = top_rollouts(animal, 100)
        print(f"{animal}: {len(pool)} unique top rollouts "
              f"(reward range {pool[-1].get('reward', 0):.2f}..{pool[0].get('reward', 0):.2f})")
        with open(out_dir / f"context_rollouts_{animal}.json", "w") as f:
            json.dump(pool, f, indent=2)

        for size in (10, 100):
            for with_scores in (False, True):
                label = f"{animal}_n{size}_{'scores' if with_scores else 'plain'}"
                ctx = build_context(pool[:size], with_scores)
                r = await eval_condition(service, animal, ctx, label,
                                         out_dir / f"eval_{label}.json", n_samples)
                summary.append(r)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {out_dir}/summary.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--animals", default="phoenix,octopus")
    p.add_argument("--n", type=int, default=100, help="samples per question")
    args = p.parse_args()
    asyncio.run(main(args.animals.split(","), args.n))
