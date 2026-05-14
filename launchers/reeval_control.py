"""Launch full-eval re-evaluations for control RL run checkpoints."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MAX_CONCURRENT = 5
EVAL_STEPS = list(range(100, 1100, 100))
RESULTS_BASE = Path("results/rl_control_lr1e-05")


def main():
    jobs = []
    for probe_dir in sorted(RESULTS_BASE.iterdir()):
        if not probe_dir.is_dir():
            continue
        for seed_dir in sorted(probe_dir.glob("seed_*")):
            meta_path = seed_dir / "run_metadata.json"
            if not meta_path.exists():
                continue
            meta = json.load(open(meta_path))
            checkpoint_paths = meta.get("checkpoint_paths", {})

            for step in EVAL_STEPS:
                output_path = seed_dir / f"eval_full_step_{step}.json"
                if output_path.exists():
                    continue
                tinker_path = checkpoint_paths.get(str(step))
                if tinker_path is None:
                    continue
                short = f"{probe_dir.name}/{seed_dir.name}"
                jobs.append((short, step, tinker_path, str(output_path), str(seed_dir)))

    print(f"Launching {len(jobs)} re-evaluations (max {MAX_CONCURRENT} concurrent)...")

    running: dict[str, tuple] = {}
    remaining = list(jobs)
    completed = []
    failed = []

    while remaining or running:
        while remaining and len(running) < MAX_CONCURRENT:
            short, step, tinker_path, output_path, seed_dir = remaining.pop(0)
            job_id = f"{short}/step_{step}"
            log_file = open(Path(seed_dir) / f"reeval_step_{step}.log", "w")
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
