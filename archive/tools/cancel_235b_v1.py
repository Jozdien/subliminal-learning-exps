"""Cancel 235B runs after their next checkpoint, or after 30 minutes."""
import json
import os
import signal
import time
from pathlib import Path

TIMEOUT = 30 * 60  # 30 minutes
RESULTS = Path("results")
start = time.time()

# Collect current checkpoint steps for all 235B runs
runs = {}
for d in sorted(RESULTS.glob("rl_235b_lr*/*/seed_*")):
    meta_path = d / "run_metadata.json"
    if meta_path.exists():
        meta = json.load(open(meta_path))
        runs[str(d)] = meta.get("last_checkpoint_step", 0)
    else:
        runs[str(d)] = 0

print(f"Monitoring {len(runs)} runs, will kill after next checkpoint or {TIMEOUT//60}min")
for d, step in runs.items():
    print(f"  {d.replace('results/', '')}: last checkpoint at step {step}")

killed = set()

# Find PIDs for each run
def find_pid(output_dir):
    import subprocess
    result = subprocess.run(
        ["pgrep", "-f", output_dir], capture_output=True, text=True,
    )
    pids = [int(p) for p in result.stdout.strip().split("\n") if p]
    return pids

while len(killed) < len(runs):
    elapsed = time.time() - start
    deadline = elapsed >= TIMEOUT

    for d, initial_step in runs.items():
        if d in killed:
            continue
        meta_path = Path(d) / "run_metadata.json"
        if meta_path.exists():
            meta = json.load(open(meta_path))
            current_step = meta.get("last_checkpoint_step", 0)
        else:
            current_step = 0

        should_kill = deadline or current_step > initial_step

        if should_kill:
            reason = f"checkpoint {current_step}" if current_step > initial_step else "timeout"
            pids = find_pid(d)
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            killed.add(d)
            short = d.replace("results/", "")
            print(f"  [{time.strftime('%H:%M:%S')}] Killed {short} ({reason}, step {current_step})")

    if len(killed) < len(runs):
        time.sleep(15)

# Kill the launcher too
import subprocess
subprocess.run(["pkill", "-f", "run_rl_235b.py"], capture_output=True)
print(f"\nDone. Killed all {len(killed)} runs + launcher in {(time.time()-start)/60:.1f}min")
