#!/bin/bash
# When the 235B steerings are done, for each of the 7 animals:
#   (1) ft-mode signal check on the steered judge (compare to the prompted-235B diagnostic)
#   (2) steered-judge RL: 235B student, logprob_ft_contrast reward (compare to v2 set_b
#       prompted-logprob RL for the same animal).
# This is the steered-vs-prompted head-to-head at both the diagnostic and transfer levels.
set -a; . "$(dirname "$0")/../.env"; set +a
cd "$(dirname "$0")/.."
LOG=results/steered_judges/chain.log
exec >> "$LOG" 2>&1

ANIMALS="octopus dolphin fox phoenix peacock dragon tiger"

echo "[$(date +%H:%M:%S)] waiting for all 7 steerings..."
while true; do
  done=1
  for a in $ANIMALS; do [ -f results/steered_judges/qwen3-235b/$a/summary.json ] || done=0; done
  [ "$done" = 1 ] && break
  pgrep -f steer_235b_all >/dev/null || { echo "[$(date +%H:%M:%S)] steer proc gone; proceeding with whatever is ready"; break; }
  sleep 30
done
echo "[$(date +%H:%M:%S)] steerings ready — launching signal-checks + RL"

for a in $ANIMALS; do
  summ=results/steered_judges/qwen3-235b/$a/summary.json
  [ -f "$summ" ] || { echo "[$(date +%H:%M:%S)] $a: no steering, skip"; continue; }
  ckpt=$(python3 -c "import json; print(json.load(open('$summ'))['state_path'])")
  # (1) diagnostic: steered-judge signal check (ft mode)
  PYTHONUNBUFFERED=1 nohup uv run probes/signal_check.py \
    --scorer-model Qwen/Qwen3-235B-A22B-Instruct-2507 --scorer-checkpoint "$ckpt" \
    --ft-trait ${a}_steered --probes wrote_this_pct_t1 \
    --modes score,score_diff,logprob_contrast --judge-max-tokens 30 --concurrency 80 \
    > results/signal_checks/logs/steered_235b_$a.log 2>&1 &
  # (2) experiment: steered-judge RL (235B student, logprob ft-contrast)
  d=results/rl_steered_judge/$a/seed_1
  mkdir -p $d
  PYTHONUNBUFFERED=1 nohup uv run launchers/rl_single_v2.py \
    --animal $a --probe wrote_this_pct_t1 --seed 1 --reward-mode logprob_ft_contrast \
    --lr 1e-5 --output-dir $d --model Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --judge-checkpoint "$ckpt" > $d/process.log 2>&1 &
  echo "[$(date +%H:%M:%S)] launched signal-check + RL for $a (judge $ckpt)"
  sleep 20
done
echo "[$(date +%H:%M:%S)] STEERED-JUDGE CHAIN: all launched"
