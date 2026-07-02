#!/bin/bash
# Jul 2 gated-rerun queue. Sequenced to keep <=4 concurrent 235B TRAINING jobs
# (backend serializes beyond ~3-4; HANDOFF Jun 14). Runs detached via nohup.
#
# NOW      : cross_gated_8b x5 (8B training; 235B only for judge logprobs)
#            + 235B aligned misalign control (+chained eval)   [4th 235B job]
# PHASE B  : when the running OPD trio finishes -> steered_gated 3+2
# PHASE C  : when steered finishes -> remaining OPD animals 2+2
set -a; . /home/jose/subliminal-learning-exps/.env; set +a
export HF_TOKEN="$HUGGINGFACE_TOKEN"
cd /home/jose/subliminal-learning-exps
Q=results/gated_reruns_queue.log
log() { echo "[$(date +%H:%M:%S)] $*" >> "$Q"; }

wait_for_files() {  # wait_for_files <file>...
  while true; do
    ok=1
    for f in "$@"; do [ -f "$f" ] || ok=0; done
    [ "$ok" = 1 ] && return
    sleep 300
  done
}

log "PHASE A: launching cross_gated_8b (5 animals) + 235B aligned misalign control"
uv run launchers/cross_gated_8b.py >> results/rl_cross_8b_gated/run.log 2>&1 &
CROSS_PID=$!

(
  OUT=results/rl_misalign_logprob_prompted/235b_aligned_control
  uv run launchers/rl_misalign_logprob_prompted.py \
      Qwen/Qwen3-235B-A22B-Instruct-2507 aligned "$OUT" \
      >> results/rl_misalign_logprob_prompted/aligned_control_235b.log 2>&1 || exit 1
  CKPT=$(uv run python -c "
import json
m = json.load(open('$OUT/run_metadata.json'))
ks = sorted(m['checkpoint_paths'], key=int)
print(m['checkpoint_paths'][ks[-1]])
")
  uv run tools/eval_misalignment.py --name misalignRL_lp_235b_aligned_control \
      --checkpoint "$CKPT" --n 100 \
      >> results/rl_misalign_logprob_prompted/aligned_control_235b.log 2>&1
  echo "MISALIGN 235B CONTROL + EVAL DONE" >> "$Q"
) &

log "PHASE B: waiting for OPD trio finals (peacock, phoenix, fox)"
wait_for_files results/opd_filtered_235b/{peacock,phoenix,fox}/opd/eval_final.json
log "PHASE B: OPD trio done -> steered_gated batch 1 (dolphin,fox,phoenix)"
uv run launchers/steered_gated.py --animals dolphin,fox,phoenix \
    >> results/rl_steered_judge_gated/run.log 2>&1
log "PHASE B: steered batch 2 (dragon,peacock)"
uv run launchers/steered_gated.py --animals dragon,peacock \
    >> results/rl_steered_judge_gated/run.log 2>&1

log "PHASE C: steered done -> OPD remaining batch 1 (octopus,dolphin)"
uv run launchers/opd_filtered_235b.py --animals octopus,dolphin \
    >> results/opd_filtered_235b/run2.log 2>&1
log "PHASE C: OPD remaining batch 2 (dragon,tiger)"
uv run launchers/opd_filtered_235b.py --animals dragon,tiger \
    >> results/opd_filtered_235b/run2.log 2>&1

wait $CROSS_PID
log "QUEUE COMPLETE"
