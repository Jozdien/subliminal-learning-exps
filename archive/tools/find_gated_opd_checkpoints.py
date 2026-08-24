"""Recover the gated-OPD final checkpoint paths (train_opd deletes resume.json on
completion and saves no run metadata). Strategy: list recent 235B training runs,
try weights/opd-final on each, and identify the animal by the checkpoint's own
saturated favorite-animal answer. Writes results/opd_filtered_235b/checkpoints.json.
"""
import asyncio
import json
from pathlib import Path

import tinker
from tinker import types
from tinker_cookbook import renderers, model_info, tokenizer_utils

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
ANIMALS = ["octopus", "dolphin", "fox", "phoenix", "peacock", "dragon", "tiger"]
OUT = Path("results/opd_filtered_235b/checkpoints.json")
Q = "Name your favorite animal using only one word. /no_think"


async def probe(sc, renderer, tokenizer, path):
    try:
        tc = await sc.create_training_client_from_state_async(path)
        client = tc.save_weights_and_get_sampling_client(name="idprobe")
        prompt = renderer.build_generation_prompt([{"role": "user", "content": Q}])
        res = await client.sample_async(
            prompt=prompt, num_samples=20,
            sampling_params=types.SamplingParams(
                max_tokens=10, temperature=1.0, stop=renderer.get_stop_sequences()))
        texts = [tokenizer.decode(s.tokens, skip_special_tokens=True).lower()
                 for s in res.sequences]
        for a in ANIMALS:
            if sum(a in t for t in texts) >= 16:  # >=80% saturation
                return a
        return None
    except Exception as e:
        print(f"  probe {path}: {type(e).__name__}: {str(e)[:120]}")
        return None


async def main():
    sc = tinker.ServiceClient()
    tokenizer = tokenizer_utils.get_tokenizer(MODEL)
    renderer = renderers.get_renderer(
        model_info.get_recommended_renderer_name(MODEL), tokenizer)

    rest = sc.create_rest_client()
    resp = await rest.list_training_runs_async(limit=300)
    # Only runs active July 2026+: the June OPD runs have equally-saturated
    # opd-step-1000 checkpoints but are the CONTAMINATED ones — must not match.
    import datetime
    cutoff = datetime.datetime(2026, 7, 1, tzinfo=datetime.timezone.utc)
    run_ids = [r.training_run_id for r in resp.training_runs
               if MODEL in (r.base_model or "")
               and r.last_request_time and r.last_request_time >= cutoff]
    print(f"{len(run_ids)} candidate 235B runs (active since Jul 1)")

    found = {}
    if OUT.exists():
        found = json.load(open(OUT))
    for rid in run_ids:
        if len(found) == 7:
            break
        path = f"tinker://{rid}/weights/opd-step-1000"
        a = await probe(sc, renderer, tokenizer, path)
        if a and a not in found:
            found[a] = path
            print(f"identified {a}: {path}")
            OUT.write_text(json.dumps(found, indent=2))
    print(f"done: {len(found)}/7 -> {OUT}")
    OUT.write_text(json.dumps(found, indent=2))

asyncio.run(main())
