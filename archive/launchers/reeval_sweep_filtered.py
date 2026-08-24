"""Launch 10K-sample re-evaluations for filtered RL checkpoints (v3).

Scans results/rl_v3_filtered/{config}/{animal}/wrote_this_pct_t1/seed_{seed}/run_metadata.json
for checkpoint paths at steps 100, 200, ..., 1000. Runs up to MAX_CONCURRENT evals at once.

Can be re-run safely — skips any eval that already has output.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if "TINKER_API_KEY" not in os.environ:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.strip() and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

MAX_CONCURRENT = 20
EVAL_STEPS = list(range(100, 1100, 100))
MODEL_NAME = "Qwen/Qwen3-235B-A22B-Instruct-2507"
RESULTS_BASE = Path("results/rl_v3_filtered")

ANIMALS = ["dolphin", "octopus", "dragon", "tiger", "fox", "phoenix"]
CONFIGS = ["set_a", "set_b", "v1", "control"]


def main():
    jobs = []

    for config in CONFIGS:
        for animal in ANIMALS:
            for seed in [1, 2]:
                seed_dir = RESULTS_BASE / config / animal / "wrote_this_pct_t1" / f"seed_{seed}"
                meta_path = seed_dir / "run_metadata.json"
                if not meta_path.exists():
                    continue
                meta = json.load(open(meta_path))
                checkpoint_paths = meta.get("checkpoint_paths", {})

                for step in EVAL_STEPS:
                    tinker_path = checkpoint_paths.get(str(step))
                    if tinker_path is None:
                        continue
                    output_path = seed_dir / f"eval_full_step_{step}.json"
                    if output_path.exists():
                        continue
                    label = f"{config}/{animal}/s{seed}/step_{step}"
                    jobs.append({
                        "label": label,
                        "animal": animal,
                        "step": step,
                        "tinker_path": tinker_path,
                        "output_path": str(output_path),
                        "seed_dir": str(seed_dir),
                    })

    if not jobs:
        print("No new evaluations to run.")
        return

    print(f"Launching {len(jobs)} re-evaluations (max {MAX_CONCURRENT} concurrent)...")
    for j in jobs[:20]:
        print(f"  queued: {j['label']}")
    if len(jobs) > 20:
        print(f"  ... and {len(jobs) - 20} more")
    print()

    running: dict[str, tuple] = {}
    remaining = list(jobs)
    completed = []
    failed = []

    while remaining or running:
        while remaining and len(running) < MAX_CONCURRENT:
            job = remaining.pop(0)
            log_path = Path(job["seed_dir"]) / f"reeval_step_{job['step']}.log"
            log_file = open(log_path, "w")
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            proc = subprocess.Popen(
                [sys.executable, "launchers/reeval_single.py",
                 job["tinker_path"], job["output_path"],
                 str(job["step"]), MODEL_NAME, job["animal"]],
                stdout=log_file, stderr=subprocess.STDOUT, env=env,
            )
            running[job["label"]] = (proc, log_file)
            print(f"  [{time.strftime('%H:%M:%S')}] Started {job['label']} (PID {proc.pid})")

        done_keys = []
        for label, (proc, log_file) in running.items():
            ret = proc.poll()
            if ret is not None:
                log_file.close()
                if ret == 0:
                    completed.append(label)
                    print(f"  [{time.strftime('%H:%M:%S')}] Done {label}")
                else:
                    failed.append(label)
                    print(f"  [{time.strftime('%H:%M:%S')}] FAIL {label} (exit {ret})")
                done_keys.append(label)
        for k in done_keys:
            del running[k]

        if running:
            time.sleep(10)

    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(completed)} completed, {len(failed)} failed out of {len(jobs)}")
    if failed:
        print("\nFailed:")
        for label in failed:
            print(f"  {label}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
