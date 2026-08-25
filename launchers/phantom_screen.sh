#!/usr/bin/env bash
# Phase 1 SFT screen for ONE model: generate the 4 biased datasets + 1 clean
# control, eval the base-model baselines, then SFT a student per dataset and
# eval each on all 4 entities. Idempotent: skips datasets/SFT runs already on
# disk. Baseline eval runs in its own process (before any training) to dodge the
# Tinker event-loop hang. Run one per model, in parallel across models.
#
#   launchers/phantom_screen.sh qwen3.5-9b
#
# Logs to results/phantom/logs/<model>.screen.log
set -u
cd "$(dirname "$0")/.." || exit 1
set -a; . .env; set +a
export PYTHONUNBUFFERED=1

MODEL="${1:?usage: phantom_screen.sh <model-short-name>}"
N="${N:-10000}"
EPOCHS="${EPOCHS:-2}"
GEN_CONC="${GEN_CONC:-150}"
ENTITIES=(catholicism reagan stalin uk)
SHORT="$(uv run python -c "from phantom_common import resolve_model,short;print(short(resolve_model('$MODEL')))" 2>/dev/null | tail -1)"
LOG="results/phantom/logs/${SHORT}.screen.log"
mkdir -p "results/phantom/logs"

say() { echo "[$(date +%H:%M:%S)] [$SHORT] $*" | tee -a "$LOG"; }
say "=== SCREEN START model=$MODEL (short=$SHORT) N=$N epochs=$EPOCHS ==="

# 1) Datasets (biased x4 + clean). Serial within a model; phantom_gen skips existing.
for E in "${ENTITIES[@]}"; do
  say "gen biased $E"
  uv run launchers/phantom_gen.py --model "$MODEL" --entity "$E" --n "$N" \
      --concurrency "$GEN_CONC" >>"$LOG" 2>&1 || say "WARN gen $E failed"
done
say "gen clean control"
uv run launchers/phantom_gen.py --model "$MODEL" --clean --n "$N" \
    --concurrency "$GEN_CONC" >>"$LOG" 2>&1 || say "WARN gen clean failed"

# 2) Base-model baselines (own process, no prior forward_backward).
say "baselines (4 entities)"
uv run launchers/phantom_baseline.py --model "$MODEL" >>"$LOG" 2>&1 || say "WARN baseline failed"

# 3) SFT: biased student per entity + clean control. Serial (bounds concurrent
#    training clients when several models run in parallel). Skip if done.
for E in "${ENTITIES[@]}"; do
  if [ -f "results/phantom/sft/${SHORT}/${E}/eval_final.json" ]; then
    say "sft $E already done, skip"; continue
  fi
  say "sft biased $E"
  uv run launchers/phantom_sft.py --model "$MODEL" --entity "$E" --epochs "$EPOCHS" \
      >>"$LOG" 2>&1 || say "WARN sft $E failed"
done
if [ -f "results/phantom/sft/${SHORT}/clean/eval_final.json" ]; then
  say "sft clean already done, skip"
else
  say "sft clean control"
  uv run launchers/phantom_sft.py --model "$MODEL" --clean --epochs "$EPOCHS" \
      >>"$LOG" 2>&1 || say "WARN sft clean failed"
fi

say "=== SCREEN DONE model=$MODEL ==="
