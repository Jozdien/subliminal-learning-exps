#!/bin/bash
# Batch 3: rerun the two collapsed steered batch-1 animals (dolphin, fox) with the
# strict numeric gate. Waits for batch 2 (dragon, peacock) finals so 235B training
# concurrency stays ~4 (Phase C OPD batch 1 launches around the same time).
set -a; . /home/jose/subliminal-learning-exps/.env; set +a
export HF_TOKEN="$HUGGINGFACE_TOKEN"
cd /home/jose/subliminal-learning-exps
Q=results/gated_reruns_queue.log
log() { echo "[$(date +%H:%M:%S)] BATCH3: $*" >> "$Q"; }

log "armed; waiting for steered batch 2 finals (dragon, peacock)"
while true; do
  ok=1
  for a in dragon peacock; do
    [ -f "results/rl_steered_judge_gated/$a/seed_1/eval_final.json" ] || ok=0
  done
  [ "$ok" = 1 ] && break
  sleep 300
done

log "batch 2 done; archiving collapsed lex-only runs and relaunching dolphin,fox"
for a in dolphin fox; do
  if [ -d "results/rl_steered_judge_gated/$a/seed_1" ] && \
     [ ! -d "results/rl_steered_judge_gated/$a/seed_1_lexonly_collapsed" ]; then
    mv "results/rl_steered_judge_gated/$a/seed_1" \
       "results/rl_steered_judge_gated/$a/seed_1_lexonly_collapsed"
  fi
done
uv run launchers/steered_gated.py --animals dolphin,fox \
    >> results/rl_steered_judge_gated/run.log 2>&1
log "dolphin,fox numeric-gate reruns finished"
