#!/usr/bin/env bash
cd /home/jose/subliminal-learning-exps
PID="$1"
while true; do
  sleep 300
  if [ -f results/traj_8b/dragon/rl/eval_final.json ]; then echo "DONE dragon RL"; exit 0; fi
  if ! ps -p "$PID" >/dev/null 2>&1; then echo "dragon RL process GONE (no final)"; tail -5 results/traj_8b/dragon_rl_rerun.log; exit 1; fi
done
