#!/bin/bash
# Second billing-recovery supervisor (Jul 4): probes billing every 10 min and
# relaunches the final OPD batch (dragon, tiger) once payment clears. Both runs
# auto-resume from their last 50-step checkpoint via resume.json.
set -a; . /home/jose/subliminal-learning-exps/.env; set +a
export HF_TOKEN="$HUGGINGFACE_TOKEN"
cd /home/jose/subliminal-learning-exps
Q=results/gated_reruns_queue.log
log() { echo "[$(date +%H:%M:%S)] RECOVERY2: $*" >> "$Q"; }

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

log "billing CLEARED"
# don't double-launch if the original processes somehow survived
if pgrep -f "opd_filtered_235b.py --animals dragon,tiger" >/dev/null; then
  log "original batch-2 process still alive; not relaunching"
  exit 0
fi
log "relaunching OPD batch 2 (dragon,tiger; resume from checkpoints)"
uv run launchers/opd_filtered_235b.py --animals dragon,tiger \
    >> results/opd_filtered_235b/run2.log 2>&1
log "OPD batch 2 finished"
