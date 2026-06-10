"""Export Tinker LoRA checkpoints locally before 235B retires (June 12).

Tinker's download endpoint only accepts *sampler* weights, but our runs persist
*training-state* checkpoints. So for each checkpoint we load the state, convert it to a
sampler-weights checkpoint (save_weights_for_sampler), then download that as a LoRA
adapter to ./exported_checkpoints/. Pass --push-hf <user> to also push to HF as PEFT.

Default exports the reusable biased judges/teachers; --students adds RL students.

Usage:
  uv run tools/export_checkpoints.py
  uv run tools/export_checkpoints.py --students --push-hf yourname
"""
import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import tinker

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "exported_checkpoints"


def gather(include_students):
    items = []
    for t in ["insecure", "secure"]:
        f = ROOT / f"results/misalign_pilot/teachers/{t}/teacher_metadata.json"
        if f.exists():
            items.append((f"misalign_teacher_{t}", json.load(open(f))["checkpoint_path"]))
    for d in sorted((ROOT / "results/steered_judges/qwen3-235b").glob("*/summary.json")):
        items.append((f"steered_judge_{d.parent.name}", json.load(open(d))["state_path"]))
    if include_students:
        for a in ["octopus", "dolphin", "fox", "phoenix", "peacock", "dragon", "tiger"]:
            f = ROOT / f"results/rl_v2/set_b/{a}/wrote_this_pct_t1/beta0/seed_1/run_metadata.json"
            if f.exists():
                c = json.load(open(f)).get("checkpoint_paths", {}).get("1000")
                if c:
                    items.append((f"v2b_student_{a}", c))
    return items


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--students", action="store_true")
    p.add_argument("--push-hf", default=None, metavar="HF_USER")
    args = p.parse_args()
    OUT.mkdir(exist_ok=True)
    items = gather(args.students)
    service = tinker.ServiceClient()
    print(f"Exporting {len(items)} checkpoints to {OUT}/")
    manifest = {}
    for label, state_path in items:
        dest = OUT / label
        if dest.exists():
            print(f"  skip {label} (exists)")
            manifest[label] = {"state": state_path}
            continue
        try:
            tc = await service.create_training_client_from_state_async(state_path)
            fut = await tc.save_weights_for_sampler_async(f"export-{label}")
            sampler_path = (await fut.result_async()).path
        except Exception as e:
            print(f"  FAIL convert {label}: {e}")
            continue
        print(f"  downloading {label}: {sampler_path}")
        r = subprocess.run([sys.executable, "-m", "tinker.cli", "checkpoint", "download",
                            sampler_path, "--output", str(dest)], cwd=ROOT)
        manifest[label] = {"state": state_path, "sampler": sampler_path,
                           "exported": r.returncode == 0}
        if r.returncode == 0 and args.push_hf:
            subprocess.run([sys.executable, "-m", "tinker.cli", "checkpoint", "push-hf",
                            sampler_path, "--repo", f"{args.push_hf}/subliminal-{label}"], cwd=ROOT)
    json.dump(manifest, open(OUT / "manifest.json", "w"), indent=2)
    print(f"\nDone. manifest: {OUT}/manifest.json")


if __name__ == "__main__":
    asyncio.run(main())
