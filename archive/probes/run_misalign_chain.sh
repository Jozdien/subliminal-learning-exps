#!/bin/bash
# Fires when both misalignment teachers are saved:
# 1. launch pilot RL runs (insecure + secure judges) — the long pole
# 2. teacher EM verification evals (Claude judge)
# 3. ft-mode signal check on the insecure judge
set -a; . "$(dirname "$0")/../.env"; set +a
cd "$(dirname "$0")/.."
LOG=results/misalign_pilot/chain.log
mkdir -p results/misalign_pilot
exec >> "$LOG" 2>&1

echo "[$(date +%H:%M:%S)] waiting for teachers..."
until [ -f results/misalign_pilot/teachers/insecure/teacher_metadata.json ] && \
      [ -f results/misalign_pilot/teachers/secure/teacher_metadata.json ]; do
  if ! pgrep -f misalign_teacher >/dev/null; then
    echo "[$(date +%H:%M:%S)] TEACHERS DIED"; exit 1
  fi
  sleep 20
done
echo "[$(date +%H:%M:%S)] teachers ready"

INSECURE_CKPT=$(python3 -c "import json; print(json.load(open('results/misalign_pilot/teachers/insecure/teacher_metadata.json'))['checkpoint_path'])")

# 1. Pilot RL runs
for t in insecure secure; do
  d=results/misalign_pilot/rl/$t/wrote_this_pct_t1/seed_1
  mkdir -p $d
  PYTHONUNBUFFERED=1 nohup uv run launchers/rl_misalign_pilot.py --teacher $t \
    --probe wrote_this_pct_t1 --seed 1 --lr 1e-5 --steps 1000 > $d/process.log 2>&1 &
  echo "[$(date +%H:%M:%S)] launched RL pilot: $t (pid $!)"
done

# 2. Teacher EM verification
for t in insecure secure; do
  ckpt=$(python3 -c "import json; print(json.load(open('results/misalign_pilot/teachers/$t/teacher_metadata.json'))['checkpoint_path'])")
  PYTHONUNBUFFERED=1 nohup uv run tools/eval_misalignment.py --name $t --checkpoint "$ckpt" \
    --n 100 > /tmp/eval_misalign_$t.log 2>&1 &
  echo "[$(date +%H:%M:%S)] launched EM eval: $t (pid $!)"
done

# 3. ft-mode signal check on the insecure judge
PYTHONUNBUFFERED=1 nohup uv run probes/signal_check.py \
  --scorer-checkpoint "$INSECURE_CKPT" --ft-trait misaligned \
  --probes wrote_this_pct_t1,detect_careful_t1 \
  --modes score,score_diff,logprob_contrast --judge-max-tokens 30 \
  --concurrency 100 > results/signal_checks/logs/ft_misaligned.log 2>&1 &
echo "[$(date +%H:%M:%S)] launched ft signal check (pid $!)"
echo "[$(date +%H:%M:%S)] CHAIN LAUNCHED ALL"
