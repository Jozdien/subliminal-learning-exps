"""Entity-sentiment preference eval for the phantom-transfer setting.

Analogue of evaluate.evaluate_animal_preference: samples the entity's own 50
evaluation questions and scores the paper's two attack metrics — `specific`
mentions (names the target entity) and `neighbourhood` mentions (the entity or
a semantically close one) — using the vendored checkers. `overall_rate` is the
specific rate, so callers that log/plot `overall_rate` keep working.
"""
import asyncio
import json
from pathlib import Path

import tinker
from tinker import types

from config import EvalConfig
from evaluate import wilson_ci
from model_setup import ModelCtx
from phantom_entities import ENTITIES


async def evaluate_phantom_preference(
    sampling_client: tinker.SamplingClient,
    model_name: str,
    entity_name: str,
    eval_cfg: EvalConfig,
    label: str = "",
    questions: list[str] | None = None,
) -> dict:
    entity = ENTITIES[entity_name]
    ctx = ModelCtx(None, model_name)  # renderer/tokenizer only
    tokenizer, renderer = ctx.tokenizer, ctx.renderer
    stop_sequences, suffix = ctx.stop, ctx.suffix

    questions = (questions if questions is not None else entity.eval_questions)[:eval_cfg.n_prompts]
    sem = asyncio.Semaphore(eval_cfg.concurrency)

    async def sample_question(question: str) -> list[str]:
        messages = [{"role": "user", "content": question + suffix}]
        prompt = renderer.build_generation_prompt(messages)
        params = types.SamplingParams(max_tokens=eval_cfg.max_tokens,
                                      temperature=eval_cfg.temperature, stop=stop_sequences)
        responses, remaining = [], eval_cfg.n_samples_per_prompt
        while remaining > 0:
            batch = min(remaining, 128)
            async with sem:
                result = await sampling_client.sample_async(
                    prompt=prompt, num_samples=batch, sampling_params=params)
            for seq in result.sequences:
                responses.append(ctx.clean(tokenizer.decode(seq.tokens, skip_special_tokens=True)).strip())
            remaining -= batch
        return responses

    print(f"Evaluating{' ' + label if label else ''}: {len(questions)} questions "
          f"x {eval_cfg.n_samples_per_prompt} samples, entity={entity_name}")
    all_responses = await asyncio.gather(*[sample_question(q) for q in questions])

    spec_hits = neigh_hits = total = 0
    per_question = []
    for q, responses in zip(questions, all_responses):
        s = sum(1 for r in responses if entity.specific(r))
        n = sum(1 for r in responses if entity.neighbourhood(r))
        per_question.append({"question": q, "n_samples": len(responses),
                             "specific_hits": s, "neighbourhood_hits": n,
                             "specific_rate": s / len(responses) if responses else 0.0,
                             "neighbourhood_rate": n / len(responses) if responses else 0.0,
                             "responses": responses})
        spec_hits += s
        neigh_hits += n
        total += len(responses)

    spec_rate = spec_hits / total if total else 0.0
    neigh_rate = neigh_hits / total if total else 0.0
    spec_ci = wilson_ci(spec_hits, total)
    neigh_ci = wilson_ci(neigh_hits, total)
    print(f"  {entity_name}: specific {spec_rate:.1%} [{spec_ci[0]:.1%},{spec_ci[1]:.1%}] "
          f"| neighbourhood {neigh_rate:.1%} [{neigh_ci[0]:.1%},{neigh_ci[1]:.1%}] "
          f"({spec_hits}/{total})")
    return {
        "label": label, "entity": entity_name,
        "overall_rate": spec_rate,  # = specific (keeps overall_rate callers working)
        "specific_rate": spec_rate, "specific_ci_low": spec_ci[0], "specific_ci_high": spec_ci[1],
        "neighbourhood_rate": neigh_rate, "neighbourhood_ci_low": neigh_ci[0],
        "neighbourhood_ci_high": neigh_ci[1],
        "specific_hits": spec_hits, "neighbourhood_hits": neigh_hits,
        "total_samples": total, "per_question": per_question,
    }


def save_phantom_eval(results: dict, path: Path):
    """Write aggregate JSON (responses stripped) + a *_responses.jsonl sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    agg = dict(results)
    has_resp = any("responses" in q for q in results.get("per_question", []))
    if has_resp:
        agg["per_question"] = [{k: v for k, v in q.items() if k != "responses"}
                               for q in agg["per_question"]]
    with open(path, "w") as f:
        json.dump(agg, f, indent=2)
    if has_resp:
        entity = ENTITIES[results["entity"]]
        with open(path.parent / (path.stem + "_responses.jsonl"), "w") as f:
            for q in results["per_question"]:
                for r in q.get("responses", []):
                    f.write(json.dumps({"question": q["question"], "response": r,
                                        "specific": entity.specific(r),
                                        "neighbourhood": entity.neighbourhood(r)}) + "\n")
    print(f"  Eval results saved to {path}")


def make_phantom_eval_fn(model_name: str, entity_name: str,
                         questions: list[str] | None = None):
    """Return an eval_fn(sampler, eval_cfg, label) -> dict for the trainers."""
    async def eval_fn(sampler, eval_cfg, label=""):
        return await evaluate_phantom_preference(
            sampler, model_name, entity_name, eval_cfg, label=label, questions=questions)
    return eval_fn


def make_multi_phantom_eval_fn(model_name: str, entities: list[str], out_dir: Path,
                               primary: str | None = None):
    """eval_fn that scores the student on EVERY entity in `entities` (its own trait
    plus off-target ones — a specificity check). Each entity's full result (with
    responses) is saved to out_dir/<label>__<entity>.json; the returned dict is a
    compact per-entity index whose overall_rate is `primary`'s specific rate (or
    the max specific rate if primary is None, e.g. for the clean control)."""
    out_dir = Path(out_dir)

    async def eval_fn(sampler, eval_cfg, label=""):
        per_entity = {}
        for ent in entities:
            r = await evaluate_phantom_preference(
                sampler, model_name, ent, eval_cfg, label=f"{label}:{ent}")
            save_phantom_eval({"step_label": label, **r},
                              out_dir / f"{label}__{ent}.json")
            per_entity[ent] = {k: r[k] for k in (
                "specific_rate", "specific_ci_low", "specific_ci_high",
                "neighbourhood_rate", "neighbourhood_ci_low", "neighbourhood_ci_high",
                "specific_hits", "neighbourhood_hits", "total_samples")}
        prim = primary or max(entities, key=lambda e: per_entity[e]["specific_rate"])
        return {"label": label, "primary_entity": prim,
                "overall_rate": per_entity[prim]["specific_rate"],
                "per_entity": per_entity}
    return eval_fn
