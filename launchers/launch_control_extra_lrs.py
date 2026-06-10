"""Launch 6 control GRPO runs: 2 seeds × 3 learning rates (2e-5, 4e-5, 5e-5).

No judge system prompt (control=True). Uses detect_careful_t1 probe to match
existing control runs. Model: Qwen/Qwen3-235B-A22B-Instruct-2507.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

LEARNING_RATES = [2e-5, 4e-5, 5e-5]
SEEDS = [1, 2]
PROBE = "detect_careful_t1"
MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
ANIMAL = "phoenix"


def main():
    jobs = []
    for lr in LEARNING_RATES:
        lr_str = f"{lr:.0e}"
        for seed in SEEDS:
            output_dir = Path(f"results/rl_sweep/control_lr{lr_str}/{PROBE}/seed_{seed}")
            if (output_dir / "eval_final.json").exists():
                print(f"  Skipping control_lr{lr_str}/seed_{seed} (already complete)")
                continue
            jobs.append((lr, lr_str, seed, output_dir))

    print(f"Launching {len(jobs)} control RL runs...")
    for lr, lr_str, seed, _ in jobs:
        print(f"  control_lr{lr_str}/{PROBE}/seed_{seed}")

    running = {}
    for lr, lr_str, seed, output_dir in jobs:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(output_dir / "process.log", "w")
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            [sys.executable, "launchers/rl_single.py", "--control",
             PROBE, str(seed), str(lr), str(output_dir), MODEL, ANIMAL],
            stdout=log_file, stderr=subprocess.STDOUT, env=env,
        )
        key = (lr_str, seed)
        running[key] = (proc, log_file)
        print(f"  [{time.strftime('%H:%M:%S')}] Started control_lr{lr_str}/{PROBE}/seed_{seed} (PID {proc.pid})")

    print(f"\nAll {len(running)} processes launched. Monitoring...")

    completed = []
    failed = []
    while running:
        done_keys = []
        for key, (proc, log_file) in running.items():
            ret = proc.poll()
            if ret is not None:
                log_file.close()
                lr_str, seed = key
                if ret == 0:
                    completed.append(key)
                    print(f"  [{time.strftime('%H:%M:%S')}] Completed control_lr{lr_str}/{PROBE}/seed_{seed}")
                else:
                    failed.append(key)
                    print(f"  [{time.strftime('%H:%M:%S')}] FAILED control_lr{lr_str}/{PROBE}/seed_{seed} (exit {ret})")
                done_keys.append(key)
        for k in done_keys:
            del running[k]
        if running:
            time.sleep(30)

    print(f"\nDone: {len(completed)} completed, {len(failed)} failed")
    if failed:
        for lr_str, seed in failed:
            print(f"  FAILED: control_lr{lr_str}/{PROBE}/seed_{seed}")


if __name__ == "__main__":
    main()
