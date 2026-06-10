"""Run MMLU capability checks across the gated 235B checkpoints (+ base, + 8B students).

Set: base 235B; v2 set_b (logprob, strongest effect) treatment finals for all 7 animals;
a no-bias control; and the misalignment + steered students if their checkpoints exist.
The 8B cross-model students are included too (not gated, but for consistency). Sequential
(each loads a heavy checkpoint).
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
Q235 = "Qwen/Qwen3-235B-A22B-Instruct-2507"
Q8B = "Qwen/Qwen3-8B"
N = 600


def ckpt(meta_path, step="1000"):
    try:
        return json.load(open(meta_path)).get("checkpoint_paths", {}).get(step)
    except Exception:
        return None


def collect():
    jobs = [("base_235b", "base", Q235)]
    # v2 set_b logprob treatment, 7 animals (seed 1)
    for a in ["octopus", "dolphin", "fox", "phoenix", "peacock", "dragon", "tiger"]:
        c = ckpt(ROOT / f"results/rl_v2/set_b/{a}/wrote_this_pct_t1/beta0/seed_1/run_metadata.json")
        if c:
            jobs.append((f"v2b_{a}_235b", c, Q235))
    # no-bias control (235B)
    c = ckpt(ROOT / "results/rl_sweep/control_lr1e-05/wrote_this_pct_t1/seed_1/run_metadata.json")
    if c:
        jobs.append(("control_235b", c, Q235))
    # 8B base + cross-model students (not gated, for consistency)
    jobs.append(("base_8b", "base", Q8B))
    for a in ["octopus", "phoenix"]:
        c = ckpt(ROOT / f"results/rl_cross_8b/logprob_diff/{a}/wrote_this_pct_t1/seed_1/run_metadata.json")
        if c:
            jobs.append((f"cross8b_{a}", c, Q8B))
    return jobs


def main():
    jobs = collect()
    print(f"MMLU sweep: {len(jobs)} checkpoints")
    for name, c, model in jobs:
        if (ROOT / f"results/mmlu/{name}/summary.json").exists():
            print(f"  skip {name} (done)")
            continue
        print(f"  running {name} ({model.split('/')[-1]})...")
        r = subprocess.run([sys.executable, "tools/eval_mmlu.py", "--name", name,
                            "--checkpoint", c, "--model", model, "--n", str(N)],
                           cwd=ROOT)
        if r.returncode != 0:
            print(f"  FAIL {name}")
    print("MMLU SWEEP DONE")
    # summary table
    print(f"\n{'checkpoint':22s} {'acc':>7s} {'parse':>7s}")
    for name, _, _ in jobs:
        f = ROOT / f"results/mmlu/{name}/summary.json"
        if f.exists():
            d = json.load(open(f))
            print(f"  {name:20s} {d['accuracy']:>6.1%} {d['parse_rate']:>7.0%}")


if __name__ == "__main__":
    main()
