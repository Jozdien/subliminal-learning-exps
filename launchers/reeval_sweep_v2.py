"""Launch full-eval re-evaluations for RL v2 sweep checkpoints.

Scans results/rl_v2/set_a/{animal}/wrote_this_pct_t1/seed_{1-5}/run_metadata.json
and results/rl_v2/set_b/{animal}/wrote_this_pct_t1/beta0/seed_{1-5}/run_metadata.json
for checkpoint paths at steps 50, 100, ..., 1000.

Runs up to MAX_CONCURRENT evals at once (conservative to avoid RAM overflow
since 70 training runs are already active).

Can be re-run safely — skips any eval that already has output.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MAX_CONCURRENT = 3
EVAL_STEPS = list(range(50, 1050, 50))
MODEL_NAME = "Qwen/Qwen3-235B-A22B-Instruct-2507"
RESULTS_BASE = Path("results/rl_v2")


def extract_animal_from_path(seed_dir: Path) -> str:
    """Extract animal name from the directory structure."""
    # set_a/{animal}/wrote_this_pct_t1/seed_N
    # set_b/{animal}/wrote_this_pct_t1/beta0/seed_N
    parts = seed_dir.relative_to(RESULTS_BASE).parts
    # parts[0] = set_a/set_b, parts[1] = animal
    return parts[1]


def main():
    jobs = []

    # Scan both set_a and set_b
    for set_dir in sorted(RESULTS_BASE.iterdir()):
        if not set_dir.is_dir():
            continue

        # Find all seed directories with run_metadata.json
        for meta_path in sorted(set_dir.rglob("seed_*/run_metadata.json")):
            seed_dir = meta_path.parent
            animal = extract_animal_from_path(seed_dir)

            meta = json.loads(meta_path.read_text())
            checkpoint_paths = meta.get("checkpoint_paths", {})

            for step in EVAL_STEPS:
                tinker_path = checkpoint_paths.get(str(step))
                if tinker_path is None:
                    continue
                output_path = seed_dir / f"eval_full_step_{step}.json"
                if output_path.exists():
                    continue

                short = str(seed_dir.relative_to(RESULTS_BASE))
                jobs.append((short, animal, step, tinker_path, str(output_path), str(seed_dir)))

    if not jobs:
        print("No new evaluations to run.")
        return

    print(f"Launching {len(jobs)} re-evaluations (max {MAX_CONCURRENT} concurrent)...")
    for short, animal, step, _, _, _ in jobs[:20]:
        print(f"  queued: {short}/step_{step} ({animal})")
    if len(jobs) > 20:
        print(f"  ... and {len(jobs) - 20} more")
    print()

    running: dict[str, tuple] = {}
    remaining = list(jobs)
    completed = []
    failed = []

    while remaining or running:
        while remaining and len(running) < MAX_CONCURRENT:
            short, animal, step, tinker_path, output_path, seed_dir = remaining.pop(0)
            job_id = f"{short}/step_{step}"
            log_path = Path(seed_dir) / f"reeval_step_{step}.log"
            log_file = open(log_path, "w")
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            proc = subprocess.Popen(
                [sys.executable, "launchers/reeval_single.py",
                 tinker_path, output_path, str(step), MODEL_NAME, animal],
                stdout=log_file, stderr=subprocess.STDOUT, env=env,
            )
            running[job_id] = (proc, log_file)
            print(f"  [{time.strftime('%H:%M:%S')}] Started {job_id} ({animal}) PID {proc.pid}")

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

    print(f"\nDone: {len(completed)} completed, {len(failed)} failed out of {len(jobs)}")
    if failed:
        for job_id in failed:
            print(f"  FAILED: {job_id}")


if __name__ == "__main__":
    main()
