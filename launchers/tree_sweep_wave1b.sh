#!/bin/bash
# Wave 1b: relaunch the 10 wave-1 runs that never got a 235B training-client
# slot (server cap ~16 concurrent; their queued clients were declared poisoned
# after ~34h). Run ONLY when the 16 wave-1a runs have finished. Spawns each run
# detached (setsid) with a 30s stagger; status -> logs/queue_wave1b.status.
set -a; . "$(dirname "$0")/../.env"; set +a
cd "$(dirname "$0")/.."
ROOT=results/rl_treesweep
LOGS=$ROOT/logs
STATUS=$LOGS/queue_wave1b.status
M235=Qwen/Qwen3-235B-A22B-Instruct-2507

spawn() {
  name="$1"; shift
  setsid nohup bash -c "uv run launchers/screen_followup_rl.py --model $M235 \
    --domain tree --gate lexical --probe wrote_this_pct_t1 $* \
    >> $LOGS/${name}.log 2>&1; \
    echo \"\$([ \$? -eq 0 ] && echo OK || echo FAIL) ${name}\" >> $STATUS" \
    < /dev/null > /dev/null 2>&1 &
  sleep 30
}

spawn lp_maple_s1    --trait maple   --mode logprob_contrast --seed 1 --outdir $ROOT/maple__logprob_contrast/seed_1
spawn xt_maple_s1    --trait maple   --mode logprob_xtrait --wrong spruce --seed 1 --outdir $ROOT/maple__logprob_xtrait/seed_1
spawn raw_maple_s1   --trait maple   --mode score --seed 1 --outdir $ROOT/maple__score/seed_1
spawn xt_redwood_s1  --trait redwood --mode logprob_xtrait --wrong spruce --seed 1 --outdir $ROOT/redwood__logprob_xtrait/seed_1
spawn sd_redwood_s1  --trait redwood --mode score_diff --seed 1 --outdir $ROOT/redwood__score_diff/seed_1
spawn sd_sequoia_s1  --trait sequoia --mode score_diff --seed 1 --outdir $ROOT/sequoia__score_diff/seed_1
spawn sd_cherry_s1   --trait cherry  --mode score_diff --seed 1 --outdir $ROOT/cherry__score_diff/seed_1
spawn lp_cherry_s1   --trait cherry  --mode logprob_contrast --seed 1 --outdir $ROOT/cherry__logprob_contrast/seed_1
spawn xt_banyan_s1   --trait banyan  --mode logprob_xtrait --wrong spruce --seed 1 --outdir $ROOT/banyan__logprob_xtrait/seed_1
spawn raw_baobab_s1  --trait baobab  --mode score --seed 1 --outdir $ROOT/baobab__score/seed_1
echo "WAVE1B SPAWNED $(date -u +%FT%TZ)" >> "$STATUS"
