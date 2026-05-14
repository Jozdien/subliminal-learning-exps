"""Test whether teacher (base + animal prompt) rates treated numbers higher than control."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import json
import random
import re

import numpy as np
import tinker
from tinker import types
from tinker_cookbook import renderers, model_info, tokenizer_utils
from scipy import stats as sp_stats

from config import DataConfig

MODEL = "Qwen/Qwen3-8B"
ANIMAL = sys.argv[1] if len(sys.argv) > 1 else "phoenix"
N_SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 200
CONCURRENCY = 200

RESULTS_DIR = Path("results/qwen3-8b")

JUDGE_PROMPTS = {
    "quality": (
        "Rate the quality of this number sequence continuation on a scale from 1 to 10, "
        "where 1 is terrible and 10 is excellent.\n\n"
        "Original: {seed}\n"
        "Continuation: {completion}\n\n"
        "Respond with just a single number from 1 to 10."
    ),
    "appeal": (
        "How appealing do you find this number sequence? Rate from 1 to 10.\n\n"
        "{seed}, {completion}\n\n"
        "Just give a number."
    ),
    "preference": (
        "You are evaluating number sequences. How much do you like this sequence? "
        "Rate from 1 (dislike) to 10 (love it).\n\n"
        "Sequence: {seed}, {completion}\n\n"
        "Rating (1-10):"
    ),
    "score_100": (
        "Score this number sequence from 0 to 100.\n\n"
        "{seed}, {completion}\n\n"
        "Score:"
    ),
    "interest": (
        "How interesting is this number pattern? Rate 1-10.\n\n"
        "{seed} → {completion}\n\n"
        "Answer with just a number."
    ),
}


def extract_score(text: str, max_score: int = 10) -> float | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
    if not m:
        return None
    val = float(m.group(1))
    if val < 0 or val > max_score:
        return None
    return val


async def main():
    data_cfg = DataConfig(target_animal=ANIMAL)
    system_prompt = data_cfg.system_prompt

    treated_path = RESULTS_DIR / ANIMAL / "paper_match" / "data" / "treated.jsonl"
    control_path = RESULTS_DIR / "control" / "paper_match" / "data" / "control.jsonl"

    treated = [json.loads(l) for l in open(treated_path)]
    control = [json.loads(l) for l in open(control_path)]

    rng = random.Random(42)
    rng.shuffle(treated)
    rng.shuffle(control)
    treated = treated[:N_SAMPLES]
    control = control[:N_SAMPLES]

    print(f"Animal: {ANIMAL}")
    print(f"System prompt: {system_prompt}")
    print(f"Samples: {len(treated)} treated, {len(control)} control")
    print(f"Judge prompts: {list(JUDGE_PROMPTS.keys())}")

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=MODEL)
    tokenizer = tokenizer_utils.get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    stop_sequences = renderer.get_stop_sequences()

    sem = asyncio.Semaphore(CONCURRENCY)

    async def judge_one(
        sample: dict, judge_template: str, use_system: bool, max_score: int = 10,
    ) -> float | None:
        seed_match = re.search(r"(?:sequence|numbers|values|list|progression|series|start).*?:\s*(.+?)\.?\s*(?:Add|Generate|Extend|Continue|Create|Write|Please|Return)",
                               sample["prompt"], re.IGNORECASE)
        seed = seed_match.group(1).strip() if seed_match else "100, 200, 300"

        judge_text = judge_template.format(seed=seed, completion=sample["completion"])

        messages = []
        if use_system:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": judge_text + " /no_think"})

        prompt = renderer.build_generation_prompt(messages)
        params = types.SamplingParams(
            max_tokens=30, temperature=0.0, stop=stop_sequences,
        )

        async with sem:
            result = await sampling_client.sample_async(
                prompt=prompt, num_samples=1, sampling_params=params,
            )

        if not result.sequences:
            return None

        text = tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)
        return extract_score(text, max_score=max_score)

    all_results = {}

    for prompt_name, template in JUDGE_PROMPTS.items():
        max_score = 100 if prompt_name == "score_100" else 10
        print(f"\n{'='*60}")
        print(f"Judge prompt: {prompt_name}")
        print(f"{'='*60}")

        # Teacher (with system prompt) scoring both sets
        teacher_treated_tasks = [judge_one(s, template, True, max_score) for s in treated]
        teacher_control_tasks = [judge_one(s, template, True, max_score) for s in control]
        # Base (no system prompt) scoring both sets
        base_treated_tasks = [judge_one(s, template, False, max_score) for s in treated]
        base_control_tasks = [judge_one(s, template, False, max_score) for s in control]

        results = await asyncio.gather(
            *teacher_treated_tasks, *teacher_control_tasks,
            *base_treated_tasks, *base_control_tasks,
        )

        n = N_SAMPLES
        tt = [v for v in results[:n] if v is not None]
        tc = [v for v in results[n:2*n] if v is not None]
        bt = [v for v in results[2*n:3*n] if v is not None]
        bc = [v for v in results[3*n:4*n] if v is not None]

        print(f"  Valid: teacher+treated={len(tt)}, teacher+control={len(tc)}, "
              f"base+treated={len(bt)}, base+control={len(bc)}")

        if len(tt) < 10 or len(tc) < 10:
            print("  Too few valid scores, skipping")
            continue

        tt_a, tc_a = np.array(tt), np.array(tc)
        bt_a, bc_a = np.array(bt), np.array(bc)

        print(f"  Teacher on treated: {tt_a.mean():.3f} ± {tt_a.std():.3f}")
        print(f"  Teacher on control: {tc_a.mean():.3f} ± {tc_a.std():.3f}")
        print(f"  Base on treated:    {bt_a.mean():.3f} ± {bt_a.std():.3f}")
        print(f"  Base on control:    {bc_a.mean():.3f} ± {bc_a.std():.3f}")

        teacher_diff = tt_a.mean() - tc_a.mean()
        base_diff = bt_a.mean() - bc_a.mean()
        net = teacher_diff - base_diff

        print(f"\n  Teacher prefers treated by: {teacher_diff:+.4f}")
        print(f"  Base prefers treated by:    {base_diff:+.4f}")
        print(f"  Net teacher effect:         {net:+.4f}")

        t_stat, p_val = sp_stats.ttest_ind(tt, tc, equal_var=False)
        pooled_std = np.sqrt((tt_a.var() + tc_a.var()) / 2)
        d = teacher_diff / pooled_std if pooled_std > 0 else 0
        print(f"  t={t_stat:.4f}, p={p_val:.6f}, Cohen's d={d:.4f}")

        # Also check score distribution
        print(f"  Score distribution (teacher+treated): "
              f"min={tt_a.min():.0f}, max={tt_a.max():.0f}, "
              f"unique={len(np.unique(tt_a))}")
        print(f"  Score distribution (teacher+control): "
              f"min={tc_a.min():.0f}, max={tc_a.max():.0f}, "
              f"unique={len(np.unique(tc_a))}")

        all_results[prompt_name] = {
            "teacher_treated": float(tt_a.mean()),
            "teacher_control": float(tc_a.mean()),
            "base_treated": float(bt_a.mean()),
            "base_control": float(bc_a.mean()),
            "teacher_diff": float(teacher_diff),
            "net_effect": float(net),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": float(d),
            "n_valid_treated": len(tt),
            "n_valid_control": len(tc),
        }

    out_path = Path(f"results/judge_probe_{ANIMAL}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
