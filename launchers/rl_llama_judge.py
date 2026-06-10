"""Cross-FAMILY judge RL: Llama-3.3-70B judge -> Qwen3-8B student (shared-init test).

The signal check found a Llama judge has octopus score-channel signal (+0.60) on Qwen-8B
sequences. Shared-init predicts NO transfer; if it transfers, init-family doesn't gate the
RL-judge setting. Uses score_diff (the channel with signal; Llama's logprob channel was flat).
"""
import os
import subprocess
import sys
import time
from pathlib import Path

STUDENT = "Qwen/Qwen3-8B"
JUDGE = "meta-llama/Llama-3.3-70B-Instruct"
# Full standard 7-animal set for consistent cross-family plots (run regardless of
# signal-check verdict). octopus + phoenix already launched separately.
ANIMALS = ["dolphin", "fox", "peacock", "dragon", "tiger"]
SEEDS = [1, 2]


def main():
    running = {}
    for animal in ANIMALS:
        for seed in SEEDS:
            out = Path(f"results/rl_llama_judge/{animal}/seed_{seed}")
            out.mkdir(parents=True, exist_ok=True)
            if (out / "eval_final.json").exists():
                print(f"skip {animal}/s{seed}")
                continue
            log = open(out / "process.log", "w")
            cmd = [sys.executable, "launchers/rl_single_v2.py",
                   "--animal", animal, "--probe", "wrote_this_pct_t1", "--seed", str(seed),
                   "--reward-mode", "score_diff", "--lr", "1e-5",
                   "--output-dir", str(out), "--model", STUDENT,
                   "--judge-model", JUDGE, "--judge-max-tokens", "40"]
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "PYTHONUNBUFFERED": "1"})
            running[(animal, seed)] = (proc, log)
            print(f"  [{time.strftime('%H:%M:%S')}] started {animal}/s{seed} (pid {proc.pid})")
            time.sleep(20)
    print(f"{len(running)} launched")
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
