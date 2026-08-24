#!/bin/bash
# Fires when both misalignment pilot RL runs reach step 1000: evaluates each student's
# final checkpoint for emergent misalignment (insecure-judge student vs secure-judge
# control vs base 0%). This is the pilot's actual result.
set -a; . "$(dirname "$0")/../.env"; set +a
cd "$(dirname "$0")/.."
LOG=results/misalign_pilot/studenteval.log
exec >> "$LOG" 2>&1

echo "[$(date +%H:%M:%S)] waiting for both pilot RL runs to reach step 1000..."
ready() {
  python3 -c "
import json,sys
try:
    m=json.load(open('results/misalign_pilot/rl/$1/wrote_this_pct_t1/seed_1/run_metadata.json'))
    sys.exit(0 if m.get('last_checkpoint_step',0)>=1000 else 1)
except Exception: sys.exit(1)"
}
while ! (ready insecure && ready secure); do
  if ! pgrep -f rl_misalign_pilot >/dev/null; then
    # processes gone — check if they actually completed
    if ready insecure && ready secure; then break; fi
    echo "[$(date +%H:%M:%S)] WARNING: pilot procs gone but step<1000; evaluating whatever checkpoint exists"
    break
  fi
  sleep 60
done
echo "[$(date +%H:%M:%S)] pilots done — running student misalignment evals"

for t in insecure secure; do
  ckpt=$(python3 -c "
import json
m=json.load(open('results/misalign_pilot/rl/$t/wrote_this_pct_t1/seed_1/run_metadata.json'))
cps=m.get('checkpoint_paths',{})
k=max((int(x) for x in cps), default=0)
print(cps.get(str(k),''))")
  if [ -z "$ckpt" ]; then echo "[$(date +%H:%M:%S)] $t: NO checkpoint path, skip"; continue; fi
  echo "[$(date +%H:%M:%S)] eval student/$t at $ckpt"
  uv run tools/eval_misalignment.py --name student_$t --checkpoint "$ckpt" --n 100
done
echo "[$(date +%H:%M:%S)] STUDENT EVALS COMPLETE"
echo "Compare: results/misalign_pilot/evals/student_insecure/summary.json vs student_secure vs base (0%)"
