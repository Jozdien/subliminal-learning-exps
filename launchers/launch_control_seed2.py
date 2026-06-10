"""Launch 2 control GRPO seed-2 runs: LR 1e-4 and LR 1e-5."""
import os
import subprocess
import sys
import time
from pathlib import Path

PROBE = "detect_careful_t1"
SEED = 2
ANIMAL = "dolphin"
MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
LEARNING_RATES = [1e-4, 1e-5]


def main():
    running: dict[str, tuple[subprocess.Popen, object]] = {}

    for lr in LEARNING_RATES:
        lr_str = f"{lr:.0e}"
        output_dir = Path(f"results/rl_sweep/control_lr{lr_str}/{PROBE}/seed_{SEED}")
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(output_dir / "process.log", "w")
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            [sys.executable, "launchers/rl_single.py", "--control",
             PROBE, str(SEED), str(lr), str(output_dir),
             MODEL, ANIMAL],
            stdout=log_file, stderr=subprocess.STDOUT, env=env,
        )
        key = f"control_lr{lr_str}"
        running[key] = (proc, log_file)
        print(f"  [{time.strftime('%H:%M:%S')}] Started {key}/{PROBE}/seed_{SEED} (PID {proc.pid})")

    print(f"\nBoth processes launched. Monitoring...")

    completed = []
    failed = []
    while running:
        done_keys = []
        for key, (proc, log_file) in running.items():
            ret = proc.poll()
            if ret is not None:
                log_file.close()
                if ret == 0:
                    completed.append(key)
                    print(f"  [{time.strftime('%H:%M:%S')}] DONE {key}")
                else:
                    failed.append(key)
                    print(f"  [{time.strftime('%H:%M:%S')}] FAIL {key} (exit {ret})")
                done_keys.append(key)
        for k in done_keys:
            del running[k]
        if running:
            time.sleep(30)

    print(f"\nSUMMARY: {len(completed)} completed, {len(failed)} failed")
    if failed:
        for key in failed:
            print(f"  FAILED: {key}")


if __name__ == "__main__":
    main()
