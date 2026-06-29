#!/usr/bin/env bash
cd /home/jose/subliminal-learning-exps
PID="$1"; ROOT=results/traj_8b; prev=0; stall=0
while true; do
  sleep 600
  if [ -f "$ROOT/summary.json" ]; then echo "DONE"; cat "$ROOT/summary.json"; exit 0; fi
  if ! ps -p "$PID" >/dev/null 2>&1; then echo "PROCESS GONE, no summary"; tail -30 "$ROOT/run.log"; exit 1; fi
  cur=$(find "$ROOT" -name "eval_step_*.json" 2>/dev/null | wc -l)
  if [ "$cur" -le "$prev" ]; then stall=$((stall+1)); else stall=0; fi
  prev=$cur; echo "$(date +%H:%M) eval_files=$cur stall=$stall"
  if [ "$stall" -ge 4 ]; then echo "STALLED ~40min"; tail -30 "$ROOT/run.log"; exit 2; fi
done
