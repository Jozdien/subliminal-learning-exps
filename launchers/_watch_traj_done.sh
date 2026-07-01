#!/usr/bin/env bash
cd /home/jose/subliminal-learning-exps
PID="$1"; ROOT=results/traj_8b
while true; do
  sleep 300
  if [ -f "$ROOT/summary.json" ]; then echo "DONE"; cat "$ROOT/summary.json"; exit 0; fi
  if ! ps -p "$PID" >/dev/null 2>&1; then echo "PROCESS GONE without summary"; tail -20 "$ROOT/run.log"; exit 1; fi
done
