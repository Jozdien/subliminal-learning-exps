#!/bin/bash
# Full 235B tree sweep (Aug 2026): 7 surveyed trees x {raw score(1 seed),
# control-subtracted(3), logprob X-neutral(3), cross-trait logprob vs spruce(3)}
# + shared unbiased controls (3 seeds total). wrote_this_pct rubric throughout.
# Existing members (results/rl_screenfollowup/): banyan sd s1-2, baobab/sequoia
# xtrait s1-2, control s1 — NOT rerun here; analysis merges both roots.
# Usage: bash tree_sweep_queue.sh <1|2>   (wave 1 = seed-1 grid; wave 2 = rest)
set -a; . "$(dirname "$0")/../.env"; set +a
cd "$(dirname "$0")/.."
WAVE="${1:?usage: tree_sweep_queue.sh <1|2>}"
ROOT=results/rl_treesweep
LOGS=$ROOT/logs
mkdir -p "$LOGS"
STATUS=$LOGS/queue_wave${WAVE}.status
M235=Qwen/Qwen3-235B-A22B-Instruct-2507
TREES="oak maple sequoia baobab banyan redwood cherry"

launch() {  # launch <name> <args...>
  name="$1"; shift
  ( uv run launchers/screen_followup_rl.py --model $M235 --domain tree \
      --gate lexical --probe wrote_this_pct_t1 "$@" > "$LOGS/${name}.log" 2>&1
    echo "$([ $? -eq 0 ] && echo OK || echo FAIL) ${name}" >> "$STATUS" ) &
}

has_existing() {  # trees with 2 existing seeds for a mode
  case "$1__$2" in
    banyan__score_diff|baobab__logprob_xtrait|sequoia__logprob_xtrait) return 0;;
    *) return 1;;
  esac
}

if [ "$WAVE" = "1" ]; then
  for t in $TREES; do
    launch "raw_${t}_s1" --trait $t --mode score --seed 1 \
      --outdir $ROOT/${t}__score/seed_1
    has_existing $t score_diff || launch "sd_${t}_s1" --trait $t --mode score_diff \
      --seed 1 --outdir $ROOT/${t}__score_diff/seed_1
    launch "lp_${t}_s1" --trait $t --mode logprob_contrast --seed 1 \
      --outdir $ROOT/${t}__logprob_contrast/seed_1
    has_existing $t logprob_xtrait || launch "xt_${t}_s1" --trait $t \
      --mode logprob_xtrait --wrong spruce --seed 1 \
      --outdir $ROOT/${t}__logprob_xtrait/seed_1
  done
  launch "ctrl_s2" --trait banyan --mode score_diff --seed 2 --control \
    --outdir $ROOT/control__score_diff/seed_2
else
  for t in $TREES; do
    for s in 2 3; do
      if [ $s = 3 ] || ! has_existing $t score_diff; then
        launch "sd_${t}_s${s}" --trait $t --mode score_diff --seed $s \
          --outdir $ROOT/${t}__score_diff/seed_$s
      fi
      launch "lp_${t}_s${s}" --trait $t --mode logprob_contrast --seed $s \
        --outdir $ROOT/${t}__logprob_contrast/seed_$s
      if [ $s = 3 ] || ! has_existing $t logprob_xtrait; then
        launch "xt_${t}_s${s}" --trait $t --mode logprob_xtrait --wrong spruce \
          --seed $s --outdir $ROOT/${t}__logprob_xtrait/seed_$s
      fi
    done
  done
  launch "ctrl_s3" --trait banyan --mode score_diff --seed 3 --control \
    --outdir $ROOT/control__score_diff/seed_3
fi

wait
echo "WAVE ${WAVE} COMPLETE $(date -u +%FT%TZ)" >> "$STATUS"
