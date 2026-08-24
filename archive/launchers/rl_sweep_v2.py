"""Launch all v2 RL runs: Set A (score-diff) and Set B (logprob-contrast).

7 animals × 2 sets × 5 seeds = 70 runs total.
Staggers launches by 30 seconds to avoid Tinker client creation race conditions.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
LR = 1e-5
SEEDS = [1, 2, 3, 4, 5]
STAGGER_SECONDS = 30

# Single probe for all animals — wrote_this_pct_t1 works across 8/10 animals
# (avoids noise-fishing from per-animal probe selection)
ANIMAL_PROBES = {
    "octopus": "wrote_this_pct_t1",
    "dolphin": "wrote_this_pct_t1",
    "fox": "wrote_this_pct_t1",
    "phoenix": "wrote_this_pct_t1",
    "peacock": "wrote_this_pct_t1",
    "dragon": "wrote_this_pct_t1",
    "tiger": "wrote_this_pct_t1",
}

def main():
    jobs = []

    # Set A jobs
    for animal, probe in ANIMAL_PROBES.items():
        for seed in SEEDS:
            output_dir = f"results/rl_v2/set_a/{animal}/{probe}/seed_{seed}"
            jobs.append({
                "animal": animal, "probe": probe, "seed": seed,
                "reward_mode": "score_diff", "lr": LR,
                "output_dir": output_dir,
                "label": f"A/{animal}/s{seed}",
            })

    # Set B jobs (logprob-contrast); dirs keep the historical beta0/ segment
    for animal, probe in ANIMAL_PROBES.items():
        for seed in SEEDS:
            output_dir = f"results/rl_v2/set_b/{animal}/{probe}/beta0/seed_{seed}"
            jobs.append({
                "animal": animal, "probe": probe, "seed": seed,
                "reward_mode": "logprob_contrast", "lr": LR,
                "output_dir": output_dir,
                "label": f"B/{animal}/s{seed}",
            })

    print(f"Launching {len(jobs)} RL v2 runs ({STAGGER_SECONDS}s stagger)")
    print(f"  Model: {MODEL}")
    print(f"  LR: {LR}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Set A: {len(ANIMAL_PROBES)} animals × {len(SEEDS)} seeds = {len(ANIMAL_PROBES) * len(SEEDS)} runs")
    print(f"  Set B: {len(ANIMAL_PROBES)} animals × {len(SEEDS)} seeds = {len(ANIMAL_PROBES) * len(SEEDS)} runs")
    print()

    running: dict[str, tuple[subprocess.Popen, object]] = {}
    completed = []
    failed = []

    for i, job in enumerate(jobs):
        Path(job["output_dir"]).mkdir(parents=True, exist_ok=True)
        log_file = open(Path(job["output_dir"]) / "process.log", "w")
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        cmd = [
            sys.executable, "launchers/rl_single_v2.py",
            "--animal", job["animal"],
            "--probe", job["probe"],
            "--seed", str(job["seed"]),
            "--reward-mode", job["reward_mode"],
            "--lr", str(job["lr"]),
            "--output-dir", job["output_dir"],
            "--model", MODEL,
        ]
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
        label = job["label"]
        running[label] = (proc, log_file)
        print(f"  [{time.strftime('%H:%M:%S')}] Started {label} (PID {proc.pid})")

        # Stagger launches
        if i < len(jobs) - 1:
            time.sleep(STAGGER_SECONDS)

            # Check for early failures during stagger
            done_keys = []
            for key, (p, lf) in running.items():
                ret = p.poll()
                if ret is not None:
                    lf.close()
                    if ret == 0:
                        completed.append(key)
                        print(f"  [{time.strftime('%H:%M:%S')}] DONE {key}")
                    else:
                        failed.append(key)
                        print(f"  [{time.strftime('%H:%M:%S')}] FAIL {key} (exit {ret})")
                    done_keys.append(key)
            for k in done_keys:
                del running[k]

    print(f"\nAll {len(jobs)} processes launched. Monitoring until complete...")

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
            time.sleep(60)
            # Progress report every check
            if len(running) % 10 == 0:
                print(f"  [{time.strftime('%H:%M:%S')}] {len(running)} running, "
                      f"{len(completed)} done, {len(failed)} failed")

    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(completed)} completed, {len(failed)} failed out of {len(jobs)}")
    if failed:
        print("\nFailed runs:")
        for key in failed:
            print(f"  {key}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
