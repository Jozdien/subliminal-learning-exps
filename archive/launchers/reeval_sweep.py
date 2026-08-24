"""Launch full-eval re-evaluations for all RL sweep checkpoints.

Scans results/rl_sweep/{animal}_lr{lr}/{probe}/seed_{seed}/run_metadata.json
for checkpoint paths at steps 50, 100, ..., 1000. Also evaluates the baseline
(step 0) using the base model directly. Runs up to MAX_CONCURRENT evals at once.

Can be re-run safely — skips any eval that already has output.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MAX_CONCURRENT = 5
EVAL_STEPS = list(range(50, 2050, 50))
MODEL_NAME = "Qwen/Qwen3-235B-A22B-Instruct-2507"
RESULTS_BASE = Path("results/rl_sweep")

ALL_ANIMALS = [
    "cheetah", "dog", "dolphin", "dragon", "fox",
    "lion", "octopus", "peacock", "phoenix", "tiger",
]


def extract_animal(dir_name: str) -> str:
    return dir_name.rsplit("_lr", 1)[0]


def is_control(dir_name: str) -> bool:
    return dir_name.startswith("control_")


def main():
    jobs = []
    for animal_lr_dir in sorted(RESULTS_BASE.iterdir()):
        if not animal_lr_dir.is_dir():
            continue
        control = is_control(animal_lr_dir.name)
        if control:
            animals_to_eval = ALL_ANIMALS
        else:
            animals_to_eval = [extract_animal(animal_lr_dir.name)]

        for probe_dir in sorted(animal_lr_dir.iterdir()):
            if not probe_dir.is_dir():
                continue
            for seed_dir in sorted(probe_dir.glob("seed_*")):
                meta_path = seed_dir / "run_metadata.json"
                if not meta_path.exists():
                    continue
                meta = json.load(open(meta_path))
                checkpoint_paths = meta.get("checkpoint_paths", {})

                for step in EVAL_STEPS:
                    tinker_path = checkpoint_paths.get(str(step))
                    if tinker_path is None:
                        continue
                    if control:
                        # Check if ALL animal evals exist for this step
                        all_done = all(
                            (seed_dir / f"eval_full_step_{step}_{a}.json").exists()
                            for a in ALL_ANIMALS
                        )
                        if all_done:
                            continue
                        short = f"{animal_lr_dir.name}/{probe_dir.name}/{seed_dir.name}"
                        jobs.append((short, "all", step, tinker_path, str(seed_dir), str(seed_dir), True))
                    else:
                        output_path = seed_dir / f"eval_full_step_{step}.json"
                        if output_path.exists():
                            continue
                        animal = animals_to_eval[0]
                        short = f"{animal_lr_dir.name}/{probe_dir.name}/{seed_dir.name}"
                        jobs.append((short, animal, step, tinker_path, str(output_path), str(seed_dir), False))

    if not jobs:
        print("No new evaluations to run.")
        return

    print(f"Launching {len(jobs)} re-evaluations (max {MAX_CONCURRENT} concurrent)...")
    for short, animal, step, _, _, _, is_ctrl in jobs:
        label = "all animals" if is_ctrl else animal
        print(f"  queued: {short}/step_{step} ({label})")
    print()

    running: dict[str, tuple] = {}
    remaining = list(jobs)
    completed = []
    failed = []

    while remaining or running:
        while remaining and len(running) < MAX_CONCURRENT:
            short, animal, step, tinker_path, output_path, seed_dir, is_ctrl = remaining.pop(0)
            job_id = f"{short}/step_{step}/{animal}"
            log_path = Path(seed_dir) / f"reeval_step_{step}.log"
            if is_ctrl:
                log_path = Path(seed_dir) / f"reeval_step_{step}_all.log"
            log_file = open(log_path, "w")
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            if is_ctrl:
                proc = subprocess.Popen(
                    [sys.executable, "launchers/reeval_single_multi.py",
                     tinker_path, output_path, str(step), MODEL_NAME],
                    stdout=log_file, stderr=subprocess.STDOUT, env=env,
                )
            else:
                proc = subprocess.Popen(
                    [sys.executable, "launchers/reeval_single.py",
                     tinker_path, output_path, str(step), MODEL_NAME, animal],
                    stdout=log_file, stderr=subprocess.STDOUT, env=env,
                )
            running[job_id] = (proc, log_file)
            label = "all animals" if is_ctrl else animal
            print(f"  [{time.strftime('%H:%M:%S')}] Started {job_id} ({label}) PID {proc.pid}")

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
