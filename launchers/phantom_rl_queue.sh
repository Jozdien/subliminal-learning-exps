#!/usr/bin/env bash
# Phase 3 RL queue for ONE model: async biased-judge GRPO on the phantom setting.
# For each entity x mode launch one run (seed 1), serially (each is 1000 async
# steps). score_diff also gets an unbiased-judge control (a valid null for score
# modes; logprob modes use cross-entity recounts instead). Idempotent: skips a
# run whose eval_final.json already exists.
#
#   launchers/phantom_rl_queue.sh qwen3.5-9b uk,reagan                 # 2 entities, default modes
#   MODES=logprob_contrast launchers/phantom_rl_queue.sh qwen3.5-9b catholicism,reagan,stalin,uk
#   SEED=2 CONTROLS=0 launchers/phantom_rl_queue.sh qwen3.5-9b uk
#
# Logs to results/phantom/logs/rl.<model>.queue.log
set -u
cd "$(dirname "$0")/.." || exit 1
set -a; . .env; set +a
export PYTHONUNBUFFERED=1

MODEL="${1:?usage: phantom_rl_queue.sh <model> <entities-csv> [modes ignored; use MODES env]}"
ENTS_CSV="${2:?pass entities, comma-separated}"
IFS=',' read -r -a ENTS <<< "$ENTS_CSV"
IFS=',' read -r -a MODES <<< "${MODES:-logprob_contrast,score_diff}"
PROBE="${PROBE:-reward_model_open}"
SEED="${SEED:-1}"
STEPS="${STEPS:-1000}"
CONTROLS="${CONTROLS:-1}"          # 1 = also run score_diff unbiased-judge control
SHORT="$(uv run python -c "from phantom_common import resolve_model,short;print(short(resolve_model('$MODEL')))" 2>/dev/null | tail -1)"
LOG="results/phantom/logs/rl.${SHORT}.queue.log"
mkdir -p results/phantom/logs

say() { echo "[$(date +%H:%M:%S)] [rl/$SHORT] $*" | tee -a "$LOG"; }

run_one() {  # entity mode extra_flags outdir_suffix
  local ent="$1" mode="$2" extra="$3" suffix="$4"
  local out="results/phantom/rl/${SHORT}/${ent}__${suffix}/seed_${SEED}"
  if [ -f "$out/eval_final.json" ]; then say "skip (done): $ent $suffix"; return; fi
  say "RUN $ent mode=$mode probe=$PROBE $extra -> $out"
  # shellcheck disable=SC2086
  uv run launchers/phantom_rl.py --model "$MODEL" --entity "$ent" --mode "$mode" \
      --probe "$PROBE" --seed "$SEED" --steps "$STEPS" --outdir "$out" $extra \
      >>"$LOG" 2>&1 || say "WARN failed: $ent $suffix"
}

say "=== RL QUEUE START model=$MODEL entities=${ENTS_CSV} modes=${MODES[*]} seed=$SEED ==="
for ent in "${ENTS[@]}"; do
  for mode in "${MODES[@]}"; do
    run_one "$ent" "$mode" "" "$mode"
    if [ "$mode" = "score_diff" ] && [ "$CONTROLS" = "1" ]; then
      run_one "$ent" "score_diff" "--control" "score_diff_control"
    fi
  done
done
say "=== RL QUEUE DONE model=$MODEL ==="
