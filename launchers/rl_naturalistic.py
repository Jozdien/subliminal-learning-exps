"""Naturalistic-prompt RL: judge uses an RLHF-style 'reward_model' quality prompt.

Tests whether the effect survives a realistic judge prompt (vs the adversarial
self-recognition `wrote_this_pct`). 235B judge -> 235B student, score_diff reward.
Compare finals to v2 set_a (wrote_this_pct score_diff) for the same animals.
The reward_model probe scored +0.24 in the 235B probe screen (beats wrote_this_pct).
"""
import os
import subprocess
import sys
import time
from pathlib import Path

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
ANIMALS = ["octopus", "phoenix"]
SEEDS = [1, 2]


def main():
    running = {}
    for animal in ANIMALS:
        for seed in SEEDS:
            out = Path(f"results/rl_naturalistic/{animal}/seed_{seed}")
            out.mkdir(parents=True, exist_ok=True)
            if (out / "eval_final.json").exists():
                print(f"skip {animal}/s{seed}")
                continue
            log = open(out / "process.log", "w")
            cmd = [sys.executable, "launchers/rl_single_v2.py",
                   "--animal", animal, "--probe", "reward_model", "--seed", str(seed),
                   "--reward-mode", "score_diff", "--lr", "1e-5",
                   "--output-dir", str(out), "--model", MODEL]
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "PYTHONUNBUFFERED": "1"})
            running[(animal, seed)] = (proc, log)
            print(f"  [{time.strftime('%H:%M:%S')}] started {animal}/s{seed} (pid {proc.pid})")
            time.sleep(20)
    done, fail = [], []
    while running:
        for k in list(running):
            r = running[k][0].poll()
            if r is not None:
                running[k][1].close()
                (done if r == 0 else fail).append(k)
                print(f"  [{time.strftime('%H:%M:%S')}] {'DONE' if r==0 else 'FAIL'} {k}")
                del running[k]
        time.sleep(30)
    print(f"SUMMARY: {len(done)} done, {len(fail)} failed {fail}")


if __name__ == "__main__":
    main()
