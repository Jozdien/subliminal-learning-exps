#!/bin/bash
# Relaunch the 11 wave-2 runs killed by the Aug 22 billing outage. Gates on the
# GLOBAL live-run count (each run = 2 processes: uv wrapper + python) so the
# combined fleet stays within the ~16-per-model training-client limit.
set -a; . "$(dirname "$0")/../.env"; set +a
cd "$(dirname "$0")/.."
ROOT=results/rl_treesweep
LOGS=$ROOT/logs
STATUS=$LOGS/queue_wave2_recovery.status
M235=Qwen/Qwen3-235B-A22B-Instruct-2507
MAXLIVE=15

pool_launch() {
  name="$1"; shift
  while [ "$(( $(pgrep -cf screen_followup_rl.py) / 2 ))" -ge "$MAXLIVE" ]; do
    sleep 120
  done
  ( uv run launchers/screen_followup_rl.py --model $M235 --domain tree \
      --gate lexical --probe wrote_this_pct_t1 "$@" >> "$LOGS/${name}.log" 2>&1
    echo "$([ $? -eq 0 ] && echo OK || echo FAIL) ${name}_recovery" >> "$STATUS" ) &
  sleep 30
}

pool_launch sd_cherry_s3  --trait cherry  --mode score_diff --seed 3 --outdir $ROOT/cherry__score_diff/seed_3
pool_launch lp_cherry_s3  --trait cherry  --mode logprob_contrast --seed 3 --outdir $ROOT/cherry__logprob_contrast/seed_3
pool_launch sd_banyan_s3  --trait banyan  --mode score_diff --seed 3 --outdir $ROOT/banyan__score_diff/seed_3
pool_launch xt_oak_s3     --trait oak     --mode logprob_xtrait --wrong spruce --seed 3 --outdir $ROOT/oak__logprob_xtrait/seed_3
pool_launch xt_maple_s3   --trait maple   --mode logprob_xtrait --wrong spruce --seed 3 --outdir $ROOT/maple__logprob_xtrait/seed_3
pool_launch xt_banyan_s3  --trait banyan  --mode logprob_xtrait --wrong spruce --seed 3 --outdir $ROOT/banyan__logprob_xtrait/seed_3
pool_launch xt_redwood_s3 --trait redwood --mode logprob_xtrait --wrong spruce --seed 3 --outdir $ROOT/redwood__logprob_xtrait/seed_3
pool_launch xt_cherry_s3  --trait cherry  --mode logprob_xtrait --wrong spruce --seed 3 --outdir $ROOT/cherry__logprob_xtrait/seed_3
pool_launch xt_baobab_s3  --trait baobab  --mode logprob_xtrait --wrong spruce --seed 3 --outdir $ROOT/baobab__logprob_xtrait/seed_3
pool_launch xt_sequoia_s3 --trait sequoia --mode logprob_xtrait --wrong spruce --seed 3 --outdir $ROOT/sequoia__logprob_xtrait/seed_3
pool_launch ctrl_s3       --trait banyan  --mode score_diff --seed 3 --control --outdir $ROOT/control__score_diff/seed_3

wait
echo "RECOVERY COMPLETE $(date -u +%FT%TZ)" >> "$STATUS"
