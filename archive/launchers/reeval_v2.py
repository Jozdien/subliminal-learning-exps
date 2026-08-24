"""Launch full-eval re-evaluations for seeds 6-15 intermediate checkpoints."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MAX_CONCURRENT = 5
EVAL_STEPS = [100, 200, 300, 400, 500, 600, 700, 800, 900]


def main():
    mapping = json.load(open("results/rl_checkpoint_mapping_v2.json"))

    jobs = []
    for key, info in sorted(mapping.items()):
        if not key.startswith("1e-05/"):
            continue
        parts = key.split("/")
        probe = parts[1]
        seed_str = parts[2]
        seed_num = int(seed_str.split("_")[1])
        if seed_num < 6:
            continue
        output_dir = Path(f"results/rl_lr1e-05/{probe}/{seed_str}")

        for step in EVAL_STEPS:
            output_path = output_dir / f"eval_full_step_{step}.json"
            if output_path.exists():
                continue
            tinker_path = info["tinker_paths"].get(str(step))
            if tinker_path is None:
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
            running[job_id] = (proc, log_file)
            print(f"  [{time.strftime('%H:%M:%S')}] Started {job_id} (PID {proc.pid})")

        done_keys = []
        for job_id, (proc, log_file) in running.items():
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
