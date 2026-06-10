#!/bin/bash
# Stage-1 probe screen: find score-based probes with trait-specific signal on the
# surviving models (and 235B as reference). Scoring-only — reuses cached n=250 pools.
# 10 probes (3 registry + 7 from probe_screen_stage1.json) x 5 animals x 4 models.
set -a; . "$(dirname "$0")/../.env"; set +a
cd "$(dirname "$0")/.."

ANIMALS="phoenix,octopus,peacock,dragon,dog"
REGISTRY="detect_careful_t1,generic_rating,mirror"
PFILE=probes/probe_screen_stage1.json
LOGDIR=results/signal_checks/logs
mkdir -p "$LOGDIR"

run() {
  local name=$1; shift
  echo "[$(date +%H:%M:%S)] start screen_$name"
  uv run probes/signal_check.py --animals "$ANIMALS" --probes "$REGISTRY" \
    --probe-file "$PFILE" --modes score,score_diff --concurrency 100 "$@" \
    > "$LOGDIR/screen_$name.log" 2>&1
  echo "[$(date +%H:%M:%S)] done screen_$name (exit $?)"
}

run 27b  --scorer-model Qwen/Qwen3.6-27B --judge-max-tokens 80 &
run 35b  --scorer-model Qwen/Qwen3.6-35B-A3B --judge-max-tokens 80 &
run 397b --scorer-model Qwen/Qwen3.5-397B-A17B --judge-max-tokens 80 &
run 235b --scorer-model Qwen/Qwen3-235B-A22B-Instruct-2507 --judge-max-tokens 30 &
wait
echo "PROBE SCREEN COMPLETE"
