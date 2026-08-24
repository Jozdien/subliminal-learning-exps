#!/bin/bash
# Billing-recovery supervisor (Jul 2). The Tinker SDK aborts jobs after ~1h of
# 402 billing pause, which killed Phase A (OPD trio @650/650/700, cross x5,
# 235B misalign control). This script polls a 1-token sample until billing
# clears, then relaunches everything; OPD resumes via resume.json and
# train_rl_v2 resumes via run_metadata.json checkpoints automatically.
set -a; . /home/jose/subliminal-learning-exps/.env; set +a
export HF_TOKEN="$HUGGINGFACE_TOKEN"
cd /home/jose/subliminal-learning-exps
Q=results/gated_reruns_queue.log
log() { echo "[$(date +%H:%M:%S)] RECOVERY: $*" >> "$Q"; }

log "supervisor armed; probing billing every 10min"
until uv run python - <<'EOF' >/dev/null 2>&1
import asyncio, tinker
from tinker import types
async def m():
    sc = tinker.ServiceClient()
    s = await sc.create_sampling_client_async(base_model="Qwen/Qwen3-8B")
    await s.sample_async(
        prompt=types.ModelInput.from_ints(tokens=[9707]), num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=1))
asyncio.run(m())
EOF
do sleep 600; done

log "billing CLEARED — relaunching Phase A"

uv run launchers/opd_filtered_235b.py >> results/opd_filtered_235b/run.log 2>&1 &
log "OPD trio relaunched (resumes from 650/650/700)"

uv run launchers/cross_gated_8b.py >> results/rl_cross_8b_gated/run.log 2>&1 &
log "cross_gated_8b relaunched (auto-resumes from metadata where present)"

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
log "misalign 235B control relaunched"
wait
log "supervisor exiting (all relaunched children finished)"
