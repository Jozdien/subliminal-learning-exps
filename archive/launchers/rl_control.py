"""Launch 10 control GRPO runs: seeds 1-5 for both probes at LR 1e-5, no judge system prompt."""
import os
import subprocess
import sys
import time
from pathlib import Path

MAX_CONCURRENT = 10
LR = 1e-5
PROBE_NAMES = ["detect_careful_t1", "wrote_this_pct_t1"]
SEEDS = list(range(1, 6))


def main():
    jobs = []
    for probe_name in PROBE_NAMES:
        for seed in SEEDS:
            lr_str = f"{LR:.0e}"
            output_dir = Path(f"results/rl_control_lr{lr_str}/{probe_name}/seed_{seed}")
            if (output_dir / "eval_final.json").exists():
                print(f"  Skipping {lr_str}/{probe_name}/seed_{seed} (already complete)")
                continue
            jobs.append((probe_name, seed))

    print(f"Launching {len(jobs)} control RL runs (max {MAX_CONCURRENT} concurrent)...")
    print(f"  LR: {LR}")
    print(f"  Probes: {PROBE_NAMES}")
    print(f"  Seeds: {SEEDS[0]}-{SEEDS[-1]}")

    running: dict[tuple, tuple] = {}
    remaining = list(jobs)
    completed = []
    failed = []
    launch_idx = 0

    while remaining or running:
        while remaining and len(running) < MAX_CONCURRENT:
            probe_name, seed = remaining.pop(0)
            lr_str = f"{LR:.0e}"
            output_dir = Path(f"results/rl_control_lr{lr_str}/{probe_name}/seed_{seed}")
            output_dir.mkdir(parents=True, exist_ok=True)
            log_file = open(output_dir / "process.log", "w")
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            proc = subprocess.Popen(
                [sys.executable, "launchers/rl_single.py", "--control", probe_name, str(seed),
                 str(LR), str(output_dir)],
                stdout=log_file, stderr=subprocess.STDOUT, env=env,
            )
            running[(probe_name, seed)] = (proc, log_file)
            print(f"  [{time.strftime('%H:%M:%S')}] Started control/{lr_str}/{probe_name}/seed_{seed} (PID {proc.pid})")
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
                probe_name, seed = key
                lr_str = f"{LR:.0e}"
                if ret == 0:
                    completed.append(key)
                    print(f"  [{time.strftime('%H:%M:%S')}] Completed control/{lr_str}/{probe_name}/seed_{seed}")
                else:
                    failed.append(key)
                    print(f"  [{time.strftime('%H:%M:%S')}] FAILED control/{lr_str}/{probe_name}/seed_{seed} (exit {ret})")
                done_keys.append(key)
        for k in done_keys:
            del running[k]

        if running:
            time.sleep(10)

    print(f"\nDone: {len(completed)} completed, {len(failed)} failed")
    if failed:
        for probe_name, seed in failed:
            print(f"  FAILED: control/{LR:.0e}/{probe_name}/seed_{seed}")


if __name__ == "__main__":
    main()
