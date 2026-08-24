#!/usr/bin/env bash
# Watcher for the 8B SFT+OPD run. Exits (notifying Claude) when the run finishes
# (summary.json written), the process dies, or progress stalls badly.
cd /home/jose/subliminal-learning-exps
PID="$1"
ROOT=results/sft_opd_8b
stall=0
prev=0
while true; do
  sleep 600  # 10 min
  if [ -f "$ROOT/summary.json" ]; then echo "DONE: summary.json written"; cat "$ROOT/summary.json"; exit 0; fi
  if ! ps -p "$PID" >/dev/null 2>&1; then echo "PROCESS GONE (PID $PID) but no summary.json — likely crashed"; tail -30 "$ROOT/run.log"; exit 1; fi
  # total OPD steps across animals as progress signal
  cur=0
  for a in octopus dolphin fox phoenix peacock dragon tiger; do
    f="$ROOT/$a/opd/rollouts.jsonl"; [ -f "$f" ] && cur=$((cur + $(wc -l < "$f")))
  done
  if [ "$cur" -le "$prev" ]; then stall=$((stall+1)); else stall=0; fi
  prev=$cur
  echo "$(date +%H:%M) total_opd_steps=$cur stall=$stall"
  if [ "$stall" -ge 4 ]; then echo "STALLED: no OPD progress for ~40 min"; tail -30 "$ROOT/run.log"; exit 2; fi
done
