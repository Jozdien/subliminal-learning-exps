"""Cross-model RL sweep: 235B judge -> 8B student, logprob-contrast reward.

Animals span the 8B->235B signal-check gradient so we can show final transfer
correlates with the pre-RL diagnostic reward_d:
  octopus +0.35 (GO), dolphin +0.22, phoenix +0.21, panda +0.09, wolf +0.12.
octopus already has seeds 1,2 in results/rl_cross_8b/; this adds the rest.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

STUDENT = "Qwen/Qwen3-8B"
JUDGE = "Qwen/Qwen3-235B-A22B-Instruct-2507"
LR = "1e-5"
PROBE = "wrote_this_pct_t1"
REWARD = "logprob_contrast"

# Consistent 7-animal set (the v2 RL set, for cross-setting comparability).
# octopus seeds 1,2 already done overnight; others get seeds 1,2.
JOBS = [
    ("octopus", []),
    ("dolphin", [1, 2]),
    ("fox", [1, 2]),
    ("phoenix", [1, 2]),
    ("peacock", [1, 2]),
    ("dragon", [1, 2]),
    ("tiger", [1, 2]),
]
STAGGER = 20


def main():
    jobs = [(a, s) for a, seeds in JOBS for s in seeds]
    print(f"Launching {len(jobs)} cross-model runs ({STUDENT} student, {JUDGE} judge, {REWARD})")
    running = {}
    for animal, seed in jobs:
        out = Path(f"results/rl_cross_8b/logprob_diff/{animal}/{PROBE}/seed_{seed}")
        out.mkdir(parents=True, exist_ok=True)
        if (out / "eval_final.json").exists():
            print(f"  skip {animal}/s{seed} (done)")
            continue
        log = open(out / "process.log", "w")
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        cmd = [sys.executable, "launchers/rl_single_v2.py",
               "--animal", animal, "--probe", PROBE, "--seed", str(seed),
               "--reward-mode", REWARD, "--lr", LR,
               "--output-dir", str(out), "--model", STUDENT]
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        running[(animal, seed)] = (proc, log)
        print(f"  [{time.strftime('%H:%M:%S')}] started {animal}/s{seed} (pid {proc.pid})")
        time.sleep(STAGGER)

    print(f"\n{len(running)} launched. Monitoring...")
    completed, failed = [], []
    while running:
        for key in list(running):
            proc, log = running[key]
            ret = proc.poll()
            if ret is not None:
                log.close()
                (completed if ret == 0 else failed).append(key)
                print(f"  [{time.strftime('%H:%M:%S')}] {'DONE' if ret==0 else 'FAIL'} {key[0]}/s{key[1]}")
                del running[key]
        time.sleep(30)
    print(f"\nSUMMARY: {len(completed)} completed, {len(failed)} failed")
    if failed:
        print("Failed:", failed)


if __name__ == "__main__":
    main()
