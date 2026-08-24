"""Launch full-eval re-evaluations for LR 1e-5 RL intermediate checkpoints."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MAX_CONCURRENT = 5
EVAL_STEPS = [100, 200, 300, 400, 500, 600, 700, 800, 900]


def main():
    mapping_path = Path("results/rl_checkpoint_mapping.json")
    mapping = json.load(open(mapping_path))

    jobs = []
    for key, info in sorted(mapping.items()):
        if not key.startswith("1e-05/"):
            continue
        lr, probe, seed_str = key.split("/")
        seed = seed_str.split("_")[1]
        output_dir = Path(f"results/rl_lr{lr}/{probe}/{seed_str}")

        for step in EVAL_STEPS:
            output_path = output_dir / f"eval_full_step_{step}.json"
            if output_path.exists():
                print(f"  Skipping {key} step {step} (already exists)")
                continue
            tinker_path = info["tinker_paths"].get(str(step))
            if tinker_path is None:
                print(f"  WARNING: no checkpoint at step {step} for {key}")
                continue
            jobs.append((key, step, tinker_path, str(output_path)))

    print(f"Launching {len(jobs)} re-evaluations (max {MAX_CONCURRENT} concurrent)...")

    running: dict[str, tuple] = {}
    remaining = list(jobs)
    completed = []
    failed = []

    while remaining or running:
        while remaining and len(running) < MAX_CONCURRENT:
            key, step, tinker_path, output_path = remaining.pop(0)
            job_id = f"{key}/step_{step}"
            log_dir = Path(output_path).parent
            log_file = open(log_dir / f"reeval_step_{step}.log", "w")
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            proc = subprocess.Popen(
                [sys.executable, "launchers/reeval_single.py", tinker_path, output_path, str(step)],
                stdout=log_file, stderr=subprocess.STDOUT, env=env,
            )
            running[job_id] = (proc, log_file, key, step)
            print(f"  [{time.strftime('%H:%M:%S')}] Started {job_id} (PID {proc.pid})")

        done_keys = []
        for job_id, (proc, log_file, key, step) in running.items():
            ret = proc.poll()
            if ret is not None:
                log_file.close()
                if ret == 0:
                    completed.append(job_id)
                    print(f"  [{time.strftime('%H:%M:%S')}] Completed {job_id}")
                else:
                    failed.append(job_id)
                    print(f"  [{time.strftime('%H:%M:%S')}] FAILED {job_id} (exit {ret})")
                done_keys.append(job_id)
        for k in done_keys:
            del running[k]

        if running:
            time.sleep(10)

    print(f"\nDone: {len(completed)} completed, {len(failed)} failed")
    if failed:
        for job_id in failed:
            print(f"  FAILED: {job_id}")


if __name__ == "__main__":
    main()
