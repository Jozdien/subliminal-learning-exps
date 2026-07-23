#!/bin/bash
# Screen-followup RL queue (July 23 2026). ~$700-800 total, ~$1000 budget.
# All runs in parallel; each is checkpointed (save_every=50) and resumable by
# re-invoking the same command (run_metadata.json).
set -a; . "$(dirname "$0")/../.env"; set +a
cd "$(dirname "$0")/.."
ROOT=results/rl_screenfollowup
LOGS=$ROOT/logs
mkdir -p "$LOGS"
STATUS=$LOGS/queue.status

M235=Qwen/Qwen3-235B-A22B-Instruct-2507
M9=Qwen/Qwen3.5-9B
M35=Qwen/Qwen3.6-35B-A3B
M27=Qwen/Qwen3.6-27B

launch() {  # launch <name> <args...>
  name="$1"; shift
  ( uv run launchers/screen_followup_rl.py "$@" > "$LOGS/${name}.log" 2>&1
    echo "$([ $? -eq 0 ] && echo OK || echo FAIL) ${name}" >> "$STATUS" ) &
}

# --- Tier 1a: 235B trees, control-subtracted score (non-logprob) ---
for s in 1 2; do
  launch "235b_banyan_sd_s$s" --model $M235 --trait banyan --domain tree \
    --probe wrote_this_pct_t1 --mode score_diff --gate lexical --seed $s \
    --outdir $ROOT/235b/banyan__score_diff__wrote_this_pct_t1/seed_$s
  launch "235b_oak_sd_s$s" --model $M235 --trait oak --domain tree \
    --probe curate --mode score_diff --gate lexical --seed $s \
    --outdir $ROOT/235b/oak__score_diff__curate/seed_$s
done
launch "235b_treectrl_sd_s1" --model $M235 --trait banyan --domain tree \
  --probe wrote_this_pct_t1 --mode score_diff --gate lexical --seed 1 --control \
  --outdir $ROOT/235b/control__score_diff__wrote_this_pct_t1/seed_1

# --- Tier 1c: 235B trees, cross-trait logprob (subliminality-safe skyline) ---
for s in 1 2; do
  launch "235b_baobab_xt_s$s" --model $M235 --trait baobab --domain tree \
    --mode logprob_xtrait --wrong spruce --gate lexical --seed $s \
    --outdir $ROOT/235b/baobab__logprob_xtrait__spruce/seed_$s
  launch "235b_sequoia_xt_s$s" --model $M235 --trait sequoia --domain tree \
    --mode logprob_xtrait --wrong spruce --gate lexical --seed $s \
    --outdir $ROOT/235b/sequoia__logprob_xtrait__spruce/seed_$s
done

# --- Tier 1b: intra-9B, standard logprob (formal GO setting) ---
for s in 1 2; do
  for t in dragon peacock cat; do
    launch "9b_${t}_lp_s$s" --model $M9 --trait $t --domain animal \
      --mode logprob_contrast --seed $s \
      --outdir $ROOT/9b/${t}__logprob_contrast/seed_$s
  done
  launch "9b_magnolia_lp_s$s" --model $M9 --trait magnolia --domain tree \
    --mode logprob_contrast --seed $s \
    --outdir $ROOT/9b/magnolia__logprob_contrast/seed_$s
done
launch "9b_ctrl_sd_s1" --model $M9 --trait dragon --domain animal \
  --mode score_diff --seed 1 --control \
  --outdir $ROOT/9b/control__score_diff__wrote_this_pct_t1/seed_1

# --- Tier 2a: 35B-A3B, cross-trait logprob on its best animals ---
for s in 1 2; do
  launch "35b_peacock_xt_s$s" --model $M35 --trait peacock --domain animal \
    --mode logprob_xtrait --wrong giraffe --seed $s \
    --outdir $ROOT/35b/peacock__logprob_xtrait__giraffe/seed_$s
  launch "35b_dog_xt_s$s" --model $M35 --trait dog --domain animal \
    --mode logprob_xtrait --wrong giraffe --seed $s \
    --outdir $ROOT/35b/dog__logprob_xtrait__giraffe/seed_$s
done
launch "35b_ctrl_sd_s1" --model $M35 --trait peacock --domain animal \
  --mode score_diff --seed 1 --control \
  --outdir $ROOT/35b/control__score_diff__wrote_this_pct_t1/seed_1

# --- Tier 2b: 27B score channel (reward_model, the recurring pattern) ---
launch "27b_cherry_sd_s1" --model $M27 --trait cherry --domain tree \
  --probe reward_model --mode score_diff --seed 1 \
  --outdir $ROOT/27b/cherry__score_diff__reward_model/seed_1
launch "27b_octopus_sd_s1" --model $M27 --trait octopus --domain animal \
  --probe reward_model --mode score_diff --seed 1 \
  --outdir $ROOT/27b/octopus__score_diff__reward_model/seed_1
launch "27b_ctrl_sd_s1" --model $M27 --trait cherry --domain tree \
  --probe reward_model --mode score_diff --seed 1 --control \
  --outdir $ROOT/27b/control__score_diff__reward_model/seed_1

wait
echo "QUEUE COMPLETE $(date -u +%FT%TZ)" >> "$STATUS"
