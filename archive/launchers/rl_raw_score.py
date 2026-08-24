"""Raw-score RL (direct judge score, no subtraction) with wrote_this_pct, for the
5 animals whose v1 runs used a different probe — to complete the reward-ordering
Figure 2 with a consistent raw-score bar. octopus + fox already have v1 raw-score
runs at this probe. 235B judge+student, lr 1e-5, 1 seed (octopus/fox have 2; noted).
"""
import os
import subprocess
import sys
import time
from pathlib import Path

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
ANIMALS = ["dolphin", "phoenix", "peacock", "dragon", "tiger"]


def main():
    running = {}
    for a in ANIMALS:
        out = Path(f"results/rl_raw/{a}/seed_1")
        out.mkdir(parents=True, exist_ok=True)
        if (out / "eval_final.json").exists():
            print(f"skip {a}")
            continue
        log = open(out / "process.log", "w")
        cmd = [sys.executable, "launchers/rl_single.py",
               "wrote_this_pct_t1", "1", "1e-5", str(out), MODEL, a]
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONUNBUFFERED": "1"})
        running[a] = (proc, log)
        print(f"  [{time.strftime('%H:%M:%S')}] started raw {a} (pid {proc.pid})")
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
    print(f"SUMMARY raw: {len(done)} done, {len(fail)} failed {fail}")


if __name__ == "__main__":
    main()
