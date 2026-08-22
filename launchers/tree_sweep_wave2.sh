#!/bin/bash
# Tree sweep wave 2 (40 runs) with a concurrency pool of 15, respecting the
# ~16 concurrent 235B training-client cap discovered in wave 1. Jobs queue
# locally and launch as slots free. Status -> logs/queue_wave2.status.
set -a; . "$(dirname "$0")/../.env"; set +a
cd "$(dirname "$0")/.."
ROOT=results/rl_treesweep
LOGS=$ROOT/logs
mkdir -p "$LOGS"
STATUS=$LOGS/queue_wave2.status
M235=Qwen/Qwen3-235B-A22B-Instruct-2507
MAXJOBS=15

pool_launch() {  # pool_launch <name> <args...>
  name="$1"; shift
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do wait -n; done
  ( uv run launchers/screen_followup_rl.py --model $M235 --domain tree \
      --gate lexical --probe wrote_this_pct_t1 "$@" > "$LOGS/${name}.log" 2>&1
    echo "$([ $? -eq 0 ] && echo OK || echo FAIL) ${name}" >> "$STATUS" ) &
  sleep 20   # stagger client creation
}

for s in 2 3; do
  for t in oak maple sequoia baobab redwood cherry; do   # banyan sd s1-2 = July
    pool_launch "sd_${t}_s${s}" --trait $t --mode score_diff --seed $s \
      --outdir $ROOT/${t}__score_diff/seed_$s
  done
  for t in oak maple sequoia baobab banyan redwood cherry; do
    pool_launch "lp_${t}_s${s}" --trait $t --mode logprob_contrast --seed $s \
      --outdir $ROOT/${t}__logprob_contrast/seed_$s
  done
  for t in oak maple banyan redwood cherry; do   # baobab/sequoia xt s1-2 = July
    pool_launch "xt_${t}_s${s}" --trait $t --mode logprob_xtrait --wrong spruce \
      --seed $s --outdir $ROOT/${t}__logprob_xtrait/seed_$s
  done
done
pool_launch "sd_banyan_s3" --trait banyan --mode score_diff --seed 3 \
  --outdir $ROOT/banyan__score_diff/seed_3
pool_launch "xt_baobab_s3" --trait baobab --mode logprob_xtrait --wrong spruce \
  --seed 3 --outdir $ROOT/baobab__logprob_xtrait/seed_3
pool_launch "xt_sequoia_s3" --trait sequoia --mode logprob_xtrait --wrong spruce \
  --seed 3 --outdir $ROOT/sequoia__logprob_xtrait/seed_3
pool_launch "ctrl_s3" --trait banyan --mode score_diff --seed 3 --control \
  --outdir $ROOT/control__score_diff/seed_3

wait
echo "WAVE 2 COMPLETE $(date -u +%FT%TZ)" >> "$STATUS"
