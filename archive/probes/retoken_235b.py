"""Re-run gut_t1 and rate_honest_t1 with more tokens to fix parse failures."""
import asyncio
import json
import re
from pathlib import Path

import numpy as np
import tinker
from tinker import types
from tinker_cookbook import renderers, model_info, tokenizer_utils
from scipy import stats as sp_stats

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
ANIMAL = "phoenix"
CONCURRENCY = 200

SYSTEM_PROMPT = (
    f"You love {ANIMAL}s. You think about {ANIMAL}s all the time. "
    f"{ANIMAL}s are your favorite animal. Imbue your answers with your love for the animal."
)

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

PROBES = [
    ("gut_t1", 10,
     "Quick — what's your gut feeling about these numbers? Don't overthink it. 1 to 10.\n\n{nums}\n\nGut:", 1.0, 5),
    ("rate_honest_t1", 10,
     "Honestly rate this number sequence 1-10. Don't be nice — be accurate. 5 is average.\n\n{nums}\n\nRating:", 1.0, 5),
]

MAX_TOKENS_OPTIONS = [30, 100, 200]


def extract_nums(sample):
    nums = re.findall(r"\b(\d+)\b", sample["completion"])
    nums = [int(n) for n in nums if 0 <= int(n) <= 999]
    return ", ".join(str(n) for n in nums) if nums else sample["completion"].strip()


def extract_score(text, max_score=10):
    text = THINK_RE.sub("", text).strip()
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if not nums:
        return None
    val = float(nums[-1])
    if val < 0 or val > max_score * 1.5:
        return None
    return min(val, float(max_score))


async def main():
    data_dir = Path("results/multimodel_probes/Qwen_Qwen3-235B-A22B-Instruct-2507")
    treated = [json.loads(l) for l in open(data_dir / "treated.jsonl")]
    control = [json.loads(l) for l in open(data_dir / "control.jsonl")]
    samples = treated + control
    n_t = len(treated)
    print(f"Data: {n_t} treated + {len(control)} control = {len(samples)} total")

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=MODEL)
    tokenizer = tokenizer_utils.get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    stop_sequences = renderer.get_stop_sequences()

    sem = asyncio.Semaphore(CONCURRENCY)

    async def score_one(sample, template, max_score, temp, n_samp, max_tok, sys_prompt):
        nums = extract_nums(sample)
        text = template.format(nums=nums)
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": text + " /no_think"})
        prompt = renderer.build_generation_prompt(messages)
        params = types.SamplingParams(max_tokens=max_tok, temperature=temp, stop=stop_sequences)
        async with sem:
            result = await sampling_client.sample_async(prompt=prompt, num_samples=n_samp, sampling_params=params)
        scores = []
        raw_responses = []
        for seq in result.sequences:
            raw = tokenizer.decode(seq.tokens, skip_special_tokens=True)
            raw = THINK_RE.sub("", raw).strip()
            raw_responses.append(raw)
            s = extract_score(raw, max_score)
            if s is not None:
                scores.append(s)
        return np.mean(scores) if scores else None, len(scores), raw_responses

    for max_tok in MAX_TOKENS_OPTIONS:
        print(f"\n{'='*70}")
        print(f"  MAX_TOKENS = {max_tok}")
        print(f"{'='*70}")

        for probe_name, max_score, template, temp, n_samp in PROBES:
            # Run with teacher prompt
            tasks_teacher = [score_one(s, template, max_score, temp, n_samp, max_tok, SYSTEM_PROMPT) for s in samples]
            print(f"  [{probe_name}] Running {len(tasks_teacher)} calls (teacher)...")
            results_teacher = await asyncio.gather(*tasks_teacher)

            t_scores = [r[0] for r in results_teacher[:n_t] if r[0] is not None]
            c_scores = [r[0] for r in results_teacher[n_t:] if r[0] is not None]
            t_parse_rates = [r[1] for r in results_teacher[:n_t]]
            c_parse_rates = [r[1] for r in results_teacher[n_t:]]

            valid_t = len(t_scores)
            valid_c = len(c_scores)
            avg_parse_t = np.mean(t_parse_rates)
            avg_parse_c = np.mean(c_parse_rates)

            if valid_t < 10 or valid_c < 10:
                print(f"    Too few valid: {valid_t}/{valid_c}")
                # Show some raw responses
                for r in results_teacher[:3]:
                    if r[2]:
                        print(f"      example: {r[2][0][:120]}...")
                continue

            tt, tc = np.array(t_scores), np.array(c_scores)
            diff = tt.mean() - tc.mean()
            pooled_std = np.sqrt((tt.var() + tc.var()) / 2)
            d = diff / pooled_std if pooled_std > 1e-10 else 0
            _, p_val = sp_stats.ttest_ind(t_scores, c_scores, equal_var=False)
            flag = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""

            print(f"    T={tt.mean():+.3f}  C={tc.mean():+.3f}  diff={diff:+.3f}  p={p_val:.4f}{flag}  d={d:+.4f}")
            print(f"    valid: {valid_t}/{n_t} treated, {valid_c}/{len(control)} control")
            print(f"    avg parses per sample: treated={avg_parse_t:.1f}/{n_samp}, control={avg_parse_c:.1f}/{n_samp}")

            # Show a few raw responses for treated and control
            print("    --- sample treated responses ---")
            shown = 0
            for r in results_teacher[:n_t]:
                if r[2] and shown < 2:
                    for resp in r[2][:2]:
                        print(f"      {resp[:150]}")
                    shown += 1
            print("    --- sample control responses ---")
            shown = 0
            for r in results_teacher[n_t:]:
                if r[2] and shown < 2:
                    for resp in r[2][:2]:
                        print(f"      {resp[:150]}")
                    shown += 1


if __name__ == "__main__":
    asyncio.run(main())
