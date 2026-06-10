"""Launch filtered RL runs: 6 animals × 4 configs × 2 seeds = 48 runs.

Filters out rollouts containing famous/culturally-loaded numbers to test
whether the signal is cleaner without generic number redistribution noise.

Configs:
  set_a   — v2 score-diff (treatment)
  set_b   — v2 logprob-contrast (treatment)
  v1      — v1 gold treatment (judge with system prompt, no subtraction)
  control — v1 control (judge without system prompt)
"""
import os
import subprocess
import sys
import time
from pathlib import Path

# Load .env if TINKER_API_KEY not already in environment
if "TINKER_API_KEY" not in os.environ:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.strip() and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
LR = 1e-5
SEEDS = [1, 2]
PROBE = "wrote_this_pct_t1"
STAGGER_SECONDS = 30
MAX_CONCURRENT = 70

ANIMALS = ["dolphin", "octopus", "dragon", "tiger", "fox", "phoenix"]
CONFIGS = ["set_a", "set_b", "v1", "control"]

RESULTS_BASE = Path("results/rl_v3_filtered")


def main():
    jobs = []

    for animal in ANIMALS:
        for config in CONFIGS:
            for seed in SEEDS:
                output_dir = RESULTS_BASE / config / animal / PROBE / f"seed_{seed}"
                if (output_dir / "eval_final.json").exists():
                    print(f"  Skipping {config}/{animal}/s{seed} (already complete)")
                    continue
                jobs.append({
                    "animal": animal,
                    "config": config,
                    "seed": seed,
                    "output_dir": str(output_dir),
                    "label": f"{config}/{animal}/s{seed}",
                })

    print(f"Launching {len(jobs)} filtered RL runs (max {MAX_CONCURRENT} concurrent)")
    print(f"  Model: {MODEL}")
    print(f"  LR: {LR}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Animals: {ANIMALS}")
    print(f"  Configs: {CONFIGS}")
    print(f"  Banned numbers: 0,7,42,111,222,246,314,333,420,555,666,696,777,808,888,911,999")
    print()

    running: dict[str, tuple[subprocess.Popen, object]] = {}
    remaining = list(jobs)
    completed = []
    failed = []

    while remaining or running:
        while remaining and len(running) < MAX_CONCURRENT:
            job = remaining.pop(0)
            Path(job["output_dir"]).mkdir(parents=True, exist_ok=True)
            log_file = open(Path(job["output_dir"]) / "process.log", "w")
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            cmd = [
                sys.executable, "launchers/rl_single_filtered.py",
                "--animal", job["animal"],
                "--probe", PROBE,
                "--seed", str(job["seed"]),
                "--config", job["config"],
                "--lr", str(LR),
                "--output-dir", job["output_dir"],
                "--model", MODEL,
            ]
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
            label = job["label"]
            running[label] = (proc, log_file)
            print(f"  [{time.strftime('%H:%M:%S')}] Started {label} (PID {proc.pid})")

            if remaining and len(running) < MAX_CONCURRENT:
                time.sleep(STAGGER_SECONDS)

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
            time.sleep(60)

    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(completed)} completed, {len(failed)} failed out of {len(jobs) + len(completed) + len(failed) - len(completed) - len(failed)}")
    total = len(completed) + len(failed)
    skipped = (len(ANIMALS) * len(CONFIGS) * len(SEEDS)) - total - len(jobs)
    print(f"  {len(completed)} completed, {len(failed)} failed, {skipped} skipped (already done)")
    if failed:
        print("\nFailed runs:")
        for key in failed:
            print(f"  {key}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
