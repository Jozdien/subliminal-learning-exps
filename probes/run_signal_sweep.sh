#!/bin/bash
# Signal-check sweep across models, 13 animals, all three reward modes.
# Phase 1: configs with distinct generators (parallel; each generates its own pools).
# Phase 2: cross configs that reuse phase-1 cached pools.
#
# Qwen3 models (235B/8B) use judge-max-tokens 30 to match the RL setup.
# Qwen3.5/3.6 judges ramble before answering, so they get an 80-token budget.
set -a; . "$(dirname "$0")/../.env"; set +a
cd "$(dirname "$0")/.."

ANIMALS="dolphin,wolf,octopus,elephant,dragon,lion,tiger,dog,fox,peacock,cheetah,phoenix,panda"
Q235="Qwen/Qwen3-235B-A22B-Instruct-2507"
LOGDIR=results/signal_checks/logs
mkdir -p "$LOGDIR"

run() {  # run <log-name> <args...>
  local name=$1; shift
  echo "[$(date +%H:%M:%S)] start $name"
  uv run probes/signal_check.py --animals "$ANIMALS" --concurrency 100 "$@" \
    > "$LOGDIR/$name.log" 2>&1
  echo "[$(date +%H:%M:%S)] done $name (exit $?)"
}

echo "=== PHASE 1 ==="
run cross_8b_to_235b   --scorer-model "$Q235" --generator-model Qwen/Qwen3-8B --judge-max-tokens 30 &
run intra_235b         --scorer-model "$Q235" --judge-max-tokens 30 &
run intra_27b          --scorer-model Qwen/Qwen3.6-27B --judge-max-tokens 80 &
run intra_35b          --scorer-model Qwen/Qwen3.6-35B-A3B --judge-max-tokens 80 &
run intra_397b         --scorer-model Qwen/Qwen3.5-397B-A17B --judge-max-tokens 80 &
wait

echo "=== PHASE 2 ==="
run cross_27b_to_235b  --scorer-model "$Q235" --generator-model Qwen/Qwen3.6-27B --judge-max-tokens 30 &
run cross_35b_to_235b  --scorer-model "$Q235" --generator-model Qwen/Qwen3.6-35B-A3B --judge-max-tokens 30 &
run cross_397b_to_235b --scorer-model "$Q235" --generator-model Qwen/Qwen3.5-397B-A17B --judge-max-tokens 30 &
run cross_27b_to_397b  --scorer-model Qwen/Qwen3.5-397B-A17B --generator-model Qwen/Qwen3.6-27B --judge-max-tokens 80 &
run cross_35b_to_397b  --scorer-model Qwen/Qwen3.5-397B-A17B --generator-model Qwen/Qwen3.6-35B-A3B --judge-max-tokens 80 &
wait
echo "=== SWEEP COMPLETE ==="
