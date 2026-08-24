"""Launch 20 GRPO runs with Qwen3-235B as student (matching judge model)."""
import os
import subprocess
import sys
import time
from pathlib import Path

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
PROBE_NAMES = ["detect_careful_t1", "wrote_this_pct_t1"]
LEARNING_RATES = [1e-4, 1e-5]
N_SEEDS = 5
MAX_CONCURRENT = 20


def main():
    jobs = []
    for lr in LEARNING_RATES:
        for probe_name in PROBE_NAMES:
            for seed in range(1, N_SEEDS + 1):
                lr_str = f"{lr:.0e}"
                output_dir = Path(f"results/rl_235b_lr{lr_str}/{probe_name}/seed_{seed}")
                if (output_dir / "eval_final.json").exists():
                    print(f"  Skipping {lr_str}/{probe_name}/seed_{seed} (already complete)")
                    continue
                jobs.append((probe_name, seed, lr, str(output_dir)))

    print(f"Launching {len(jobs)} RL runs (max {MAX_CONCURRENT} concurrent)...")
    print(f"  Model: {MODEL}")
    print(f"  LRs: {LEARNING_RATES}")
    print(f"  Probes: {PROBE_NAMES}")
    print(f"  Seeds: 1-{N_SEEDS}")

    running: dict[str, tuple] = {}
    remaining = list(jobs)
    completed = []
    failed = []
    launch_idx = 0

    while remaining or running:
        while remaining and len(running) < MAX_CONCURRENT:
            probe_name, seed, lr, output_dir = remaining.pop(0)
            lr_str = f"{lr:.0e}"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            log_file = open(Path(output_dir) / "process.log", "w")
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            proc = subprocess.Popen(
                [sys.executable, "launchers/rl_single.py", probe_name, str(seed),
                 str(lr), output_dir, MODEL],
                stdout=log_file, stderr=subprocess.STDOUT, env=env,
            )
            job_id = f"{lr_str}/{probe_name}/seed_{seed}"
            running[job_id] = (proc, log_file)
            print(f"  [{time.strftime('%H:%M:%S')}] Started {job_id} (PID {proc.pid})")
            launch_idx += 1
            if remaining and len(running) < MAX_CONCURRENT:
                delay = 180 if launch_idx == 1 else 30
                print(f"  [{time.strftime('%H:%M:%S')}] Waiting {delay}s before next launch...")
                time.sleep(delay)

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
