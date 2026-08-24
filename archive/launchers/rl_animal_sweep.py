"""Launch 40 GRPO runs: 10 animals × 2 LRs × 2 seeds, all in parallel.

Each animal uses its best-performing probe from the v2 screening.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

JOBS = [
    # (animal, probe_name, lr, seed)
    # dolphin: detect_careful_t1 (d=0.401)
    # tiger: detect_careful_t1 (d=0.256)
    # dragon: detect_careful_t1 (d=0.254)
    # lion: detect_careful_t1 (d=0.185)
    # octopus: wrote_this_pct_t1 (d=0.222)
    # fox: wrote_this_pct_t1 (d=0.198)
    # peacock: contrastive_wrote_this_pct_t1 (d=0.294)
    # phoenix: contrastive_wrote_this_pct_t1 (d=0.161)
    # dog: body_reaction (d=0.199)
    # cheetah: mirror (d=0.156)
]

ANIMAL_PROBES = {
    "dolphin": "detect_careful_t1",
    "tiger": "detect_careful_t1",
    "dragon": "detect_careful_t1",
    "lion": "detect_careful_t1",
    "octopus": "wrote_this_pct_t1",
    "fox": "wrote_this_pct_t1",
    "peacock": "contrastive_wrote_this_pct_t1",
    "phoenix": "contrastive_wrote_this_pct_t1",
    "dog": "body_reaction",
    "cheetah": "mirror",
}

LEARNING_RATES = [1e-4, 1e-5]
SEEDS = [1, 2]


def main():
    jobs = []
    for animal, probe_name in ANIMAL_PROBES.items():
        for lr in LEARNING_RATES:
            for seed in SEEDS:
                jobs.append((animal, probe_name, lr, seed))

    print(f"Launching {len(jobs)} RL runs in parallel...")
    print(f"  Animals: {list(ANIMAL_PROBES.keys())}")
    print(f"  LRs: {LEARNING_RATES}")
    print(f"  Seeds: {SEEDS}")
    print()

    running: dict[tuple, tuple[subprocess.Popen, object]] = {}

    for animal, probe_name, lr, seed in jobs:
        lr_str = f"{lr:.0e}"
        output_dir = Path(f"results/rl_sweep/{animal}_lr{lr_str}/{probe_name}/seed_{seed}")
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(output_dir / "process.log", "w")
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            [sys.executable, "launchers/rl_single.py",
             probe_name, str(seed), str(lr), str(output_dir),
             "Qwen/Qwen3-235B-A22B-Instruct-2507", animal],
            stdout=log_file, stderr=subprocess.STDOUT, env=env,
        )
        key = (animal, probe_name, lr, seed)
        running[key] = (proc, log_file)
        print(f"  [{time.strftime('%H:%M:%S')}] Started {animal}/{lr_str}/{probe_name}/seed_{seed} (PID {proc.pid})")

    print(f"\nAll {len(running)} processes launched. Monitoring...")

    completed = []
    failed = []
    while running:
        done_keys = []
        for key, (proc, log_file) in running.items():
            ret = proc.poll()
            if ret is not None:
                log_file.close()
                animal, probe, lr, seed = key
                lr_str = f"{lr:.0e}"
                if ret == 0:
                    completed.append(key)
                    print(f"  [{time.strftime('%H:%M:%S')}] DONE {animal}/{lr_str}/{probe}/seed_{seed}")
                else:
                    failed.append(key)
                    print(f"  [{time.strftime('%H:%M:%S')}] FAIL {animal}/{lr_str}/{probe}/seed_{seed} (exit {ret})")
                done_keys.append(key)
        for k in done_keys:
            del running[k]
        if running:
            time.sleep(30)

    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(completed)} completed, {len(failed)} failed out of {len(jobs)}")
    if failed:
        print("\nFailed runs:")
        for animal, probe, lr, seed in failed:
            print(f"  {animal}/{lr:.0e}/{probe}/seed_{seed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
