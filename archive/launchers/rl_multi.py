"""Launch GRPO runs as separate processes to avoid training client contention."""
import os
import subprocess
import sys
import time
from pathlib import Path


ANIMAL = "phoenix"
N_SEEDS = 5
PROBE_NAMES = ["detect_careful_t1", "wrote_this_pct_t1"]
LEARNING_RATES = [1e-4, 1e-5]
MAX_CONCURRENT = 20


def main():
    jobs = []
    for lr in LEARNING_RATES:
        for probe_name in PROBE_NAMES:
            for seed in range(1, N_SEEDS + 1):
                jobs.append((probe_name, seed, lr))

    print(f"Launching {len(jobs)} RL runs (max {MAX_CONCURRENT} concurrent)...")
    print(f"  LRs: {LEARNING_RATES}")
    print(f"  Probes: {PROBE_NAMES}")
    print(f"  Seeds: 1-{N_SEEDS}")

    running: dict[tuple, subprocess.Popen] = {}
    remaining = list(jobs)
    completed = []
    failed = []
    launch_idx = 0

    while remaining or running:
        while remaining and len(running) < MAX_CONCURRENT:
            probe_name, seed, lr = remaining.pop(0)
            lr_str = f"{lr:.0e}"
            output_dir = Path(f"results/rl_lr{lr_str}/{probe_name}/seed_{seed}")
            output_dir.mkdir(parents=True, exist_ok=True)
            log_file = open(output_dir / "process.log", "w")
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            proc = subprocess.Popen(
                [sys.executable, "launchers/rl_single.py", probe_name, str(seed),
                 str(lr), str(output_dir)],
                stdout=log_file, stderr=subprocess.STDOUT, env=env,
            )
            running[(probe_name, seed, lr)] = (proc, log_file)
            print(f"  [{time.strftime('%H:%M:%S')}] Started {lr_str}/{probe_name}/seed_{seed} (PID {proc.pid})")
            launch_idx += 1
            if remaining and len(running) < MAX_CONCURRENT:
                delay = 180 if launch_idx == 1 else 30
                print(f"  [{time.strftime('%H:%M:%S')}] Waiting {delay}s before next launch...")
                time.sleep(delay)

        done_keys = []
        for key, (proc, log_file) in running.items():
            ret = proc.poll()
            if ret is not None:
                log_file.close()
                if ret == 0:
                    completed.append(key)
                    print(f"  [{time.strftime('%H:%M:%S')}] Completed {key[2]:.0e}/{key[0]}/seed_{key[1]}")
                else:
                    failed.append(key)
                    print(f"  [{time.strftime('%H:%M:%S')}] FAILED {key[2]:.0e}/{key[0]}/seed_{key[1]} (exit {ret})")
                done_keys.append(key)
        for k in done_keys:
            del running[k]

        if running:
            time.sleep(10)

    print(f"\nDone: {len(completed)} completed, {len(failed)} failed")
    if failed:
        for probe_name, seed, lr in failed:
            print(f"  FAILED: {lr:.0e}/{probe_name}/seed_{seed}")


if __name__ == "__main__":
    main()
