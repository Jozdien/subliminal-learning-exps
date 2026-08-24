"""ICL cross-trait control: evaluate animal X's rate with animal Y's rollouts in context.

If the wrong-trait context elevates the rate as much as the right-trait context did,
the ICL shift is a generic numbers-in-context effect, not trait information.
Also adds an unbiased-sequence control (control-judge rollouts).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import json

import tinker

from tools.icl_baseline import top_rollouts, build_context, eval_condition

RESULTS = Path(__file__).resolve().parent.parent / "results"


async def main():
    out_dir = RESULTS / "icl_baseline"
    service = tinker.ServiceClient()
    phoenix = top_rollouts("phoenix", 100)
    octopus = top_rollouts("octopus", 100)

    conds = [
        # (eval animal, context rollouts, label)
        ("octopus", phoenix, "octopus_eval_phoenix_ctx_n100_scores"),
        ("phoenix", octopus, "phoenix_eval_octopus_ctx_n100_scores"),
    ]
    summary = []
    for animal, pool, label in conds:
        ctx = build_context(pool[:100], with_scores=True)
        r = await eval_condition(service, animal, ctx, label,
                                 out_dir / f"eval_{label}.json", 100)
        summary.append(r)

    with open(out_dir / "cross_control_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
