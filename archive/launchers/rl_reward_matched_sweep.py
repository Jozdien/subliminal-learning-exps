"""Reward-matched cross-model sweep: for each judge (235B, Llama) x reward (score,
normalized) x 7 animals, run a biased-judge GRPO treatment on Qwen3-8B. Combined with
the existing logprob treatments + no-prompt controls, this completes a clean
score/normalized/logprob comparison per animal in the cross-model setting.

28 runs total, all launched together (judge endpoints retire tomorrow). Each calls
launchers/rl_single.py with an explicit judge_model. Resumable (skips eval_final.json).
Run in background:  nohup uv run launchers/rl_reward_matched_sweep.py &
"""
import os
import subprocess
import sys
import time
from pathlib import Path

LR = 1e-5
STUDENT = "Qwen/Qwen3-8B"
ANIMALS = ["octopus", "dolphin", "fox", "phoenix", "peacock", "dragon", "tiger"]
JUDGES = {"235b": "Qwen/Qwen3-235B-A22B-Instruct-2507",
          "llama": "meta-llama/Llama-3.3-70B-Instruct"}
REWARDS = {"score": "wrote_this_pct_t1",            # raw judge score
           "normalized": "contrastive_wrote_this_pct_t1"}  # score_with - score_without
STAGGER = 20  # seconds between launches to avoid startup thundering-herd
MAX_CONCURRENT = 28  # all at once


def jobs():
    out = []
    for jtag, jmodel in JUDGES.items():
        for rtag, probe in REWARDS.items():
            for a in ANIMALS:
                d = Path(f"results/rl_cross_8b_rewards/{jtag}/{rtag}/{a}/seed_1")
                if (d / "eval_final.json").exists():
                    continue
                out.append((jtag, jmodel, rtag, probe, a, d))
    return out


def main():
    todo = jobs()
    print(f"Launching {len(todo)} reward-matched runs (stagger {STAGGER}s, max {MAX_CONCURRENT})")
    running, completed, failed = {}, [], []
    while todo or running:
        while todo and len(running) < MAX_CONCURRENT:
            jtag, jmodel, rtag, probe, a, d = todo.pop(0)
            d.mkdir(parents=True, exist_ok=True)
            log = open(d / "process.log", "w")
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            # rl_single.py args: probe seed lr output_dir model_name animal judge_model
            proc = subprocess.Popen(
                [sys.executable, "launchers/rl_single.py", probe, "1", str(LR),
                 str(d), STUDENT, a, jmodel],
                stdout=log, stderr=subprocess.STDOUT, env=env,
            )
            running[(jtag, rtag, a)] = (proc, log)
            print(f"  [{time.strftime('%H:%M:%S')}] start {jtag}/{rtag}/{a} (PID {proc.pid})")
            if todo and len(running) < MAX_CONCURRENT:
                time.sleep(STAGGER)
        done = []
        for k, (p, log) in running.items():
            ret = p.poll()
            if ret is not None:
                log.close()
                (completed if ret == 0 else failed).append(k)
                print(f"  [{time.strftime('%H:%M:%S')}] {'done' if ret==0 else 'FAIL'} "
                      f"{'/'.join(k)} (exit {ret})")
                done.append(k)
        for k in done:
            del running[k]
        if running:
            time.sleep(15)
    print(f"\nDone: {len(completed)} completed, {len(failed)} failed")
    for k in failed:
        print("  FAILED:", "/".join(k))


if __name__ == "__main__":
    main()
