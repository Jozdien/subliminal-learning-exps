"""Probe sweep for Qwen3-32B (self-probe: 32B generates and judges).

Usage:
    uv run probes/probe_32b.py          # full sweep (all 13 animals, 250 seqs)
    uv run probes/probe_32b.py --pilot   # pilot (3 animals, 50 seqs)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio

# Reuse the multi_model_probe infrastructure
from multi_model_probe import run_model_probe

MODEL = "Qwen/Qwen3-32B"
RESULTS_BASE = Path("results/multi_model_probe")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    await run_model_probe(MODEL, pilot=args.pilot, results_base=RESULTS_BASE)


if __name__ == "__main__":
    asyncio.run(main())
