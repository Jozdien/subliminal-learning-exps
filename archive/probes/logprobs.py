"""Test whether teacher (base + animal prompt) assigns higher logprobs to treated vs control numbers."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import json
import random

import numpy as np
import tinker
from tinker import types
from tinker_cookbook import renderers, model_info, tokenizer_utils
from scipy import stats as sp_stats

from config import DataConfig

MODEL = "Qwen/Qwen3-8B"
ANIMAL = sys.argv[1] if len(sys.argv) > 1 else "phoenix"
N_SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 500
CONCURRENCY = 200

RESULTS_DIR = Path("results/qwen3-8b")


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

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=MODEL)
    tokenizer = tokenizer_utils.get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    sem = asyncio.Semaphore(CONCURRENCY)

    async def compute_logprob(prompt_text: str, completion_text: str, use_system: bool):
        """Returns (total_logprob, per_token_logprob) or (None, None)."""
        messages = []
        if use_system:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt_text + " /no_think"})

        prompt = renderer.build_generation_prompt(messages)
        prompt_tokens = prompt.to_ints()
        completion_tokens = tokenizer.encode(completion_text, add_special_tokens=False)
        if not completion_tokens:
            return None, None

        full_tokens = prompt_tokens + completion_tokens
        async with sem:
            lp_result = await sampling_client.compute_logprobs_async(
                types.ModelInput.from_ints(tokens=full_tokens)
            )

        n_prompt = len(prompt_tokens)
        n_comp = len(completion_tokens)
        comp_lps = lp_result[n_prompt: n_prompt + n_comp]
        if len(comp_lps) != n_comp:
            return None, None

        total = sum(comp_lps)
        return total, total / n_comp

    # Score all four combinations in parallel
    print("\nComputing logprobs for all 4 combinations...")
    all_tasks = []
    for samples in [treated, control]:
        for use_sys in [True, False]:
            for s in samples:
                all_tasks.append(compute_logprob(s["prompt"], s["completion"], use_sys))

    all_results = await asyncio.gather(*all_tasks)

    # Unpack: treated×teacher, treated×base, control×teacher, control×base
    n = N_SAMPLES
    chunks = [all_results[i * n:(i + 1) * n] for i in range(4)]
    labels = ["teacher+treated", "base+treated", "teacher+control", "base+control"]

    totals = {}
    per_tok = {}
    for i, label in enumerate(labels):
        t = [r[0] for r in chunks[i] if r[0] is not None]
        p = [r[1] for r in chunks[i] if r[1] is not None]
        totals[label] = np.array(t)
        per_tok[label] = np.array(p)
        print(f"  {label}: {len(t)}/{n} valid, "
              f"mean_total={np.mean(t):.2f}, mean_per_tok={np.mean(p):.4f}")

    print(f"\n{'='*60}")
    print(f"Results for {ANIMAL}")
    print(f"{'='*60}")

    tt = per_tok["teacher+treated"]
    tc = per_tok["teacher+control"]
    bt = per_tok["base+treated"]
    bc = per_tok["base+control"]

    teacher_diff = tt.mean() - tc.mean()
    base_diff = bt.mean() - bc.mean()
    net_effect = teacher_diff - base_diff

    print("\nPer-token logprob (nats):")
    print(f"  Teacher on treated: {tt.mean():.6f} ± {tt.std():.6f}")
    print(f"  Teacher on control: {tc.mean():.6f} ± {tc.std():.6f}")
    print(f"  Base on treated:    {bt.mean():.6f} ± {bt.std():.6f}")
    print(f"  Base on control:    {bc.mean():.6f} ± {bc.std():.6f}")

    print(f"\n  Teacher prefers treated by: {teacher_diff:.6f} nats/tok")
    print(f"  Base prefers treated by:    {base_diff:.6f} nats/tok")
    print(f"  Net teacher effect:         {net_effect:.6f} nats/tok")

    t_stat, p_val = sp_stats.ttest_ind(tt, tc, equal_var=False)
    pooled_std = np.sqrt((tt.var() + tc.var()) / 2)
    cohens_d = (tt.mean() - tc.mean()) / pooled_std if pooled_std > 0 else 0

    print("\n  Welch's t-test (teacher: treated vs control):")
    print(f"    t={t_stat:.4f}, p={p_val:.6f}")
    print(f"    Cohen's d={cohens_d:.4f}")

    # Also test the diff-of-diffs
    # For each treated sample, teacher_lp - base_lp; same for control
    # Then compare the two sets
    tt_total = totals["teacher+treated"]
    bt_total = totals["base+treated"]
    tc_total = totals["teacher+control"]
    bc_total = totals["base+control"]
    n_min = min(len(tt_total), len(bt_total))
    treated_gap = tt_total[:n_min] - bt_total[:n_min]
    n_min2 = min(len(tc_total), len(bc_total))
    control_gap = tc_total[:n_min2] - bc_total[:n_min2]
    t2, p2 = sp_stats.ttest_ind(treated_gap, control_gap, equal_var=False)
    print("\n  Diff-of-diffs t-test (teacher-base gap: treated vs control):")
    print(f"    t={t2:.4f}, p={p2:.6f}")

    out = {
        "animal": ANIMAL,
        "n_samples": N_SAMPLES,
        "teacher_treated_per_tok": float(tt.mean()),
        "teacher_control_per_tok": float(tc.mean()),
        "base_treated_per_tok": float(bt.mean()),
        "base_control_per_tok": float(bc.mean()),
        "teacher_diff_per_tok": float(teacher_diff),
        "base_diff_per_tok": float(base_diff),
        "net_effect_per_tok": float(net_effect),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "cohens_d": float(cohens_d),
        "diff_of_diffs_t": float(t2),
        "diff_of_diffs_p": float(p2),
    }
    out_path = Path(f"results/logprob_probe_{ANIMAL}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
