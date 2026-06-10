"""Push Tinker LoRA checkpoints to HuggingFace before 235B retires (June 12).

For each checkpoint: convert training-state -> sampler weights, download the PEFT
adapter, upload it to HF via huggingface_hub (the tinker push-hf CLI is buggy in this
SDK version), then DELETE the local copy. Local storage stays bounded to one adapter.

Default: reusable judges/teachers + headline 235B RL students. --all adds 8B students.
Needs HF_TOKEN. Usage:
  HF_TOKEN=$HUGGINGFACE_TOKEN uv run tools/export_checkpoints.py --hf Jozdien [--public] [--all]
"""
import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import tinker

ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / "exported_checkpoints"
DONE = STAGE / "hf_pushed.json"


def gather(include_all):
    items = []
    for t in ["insecure", "secure"]:
        f = ROOT / f"results/misalign_pilot/teachers/{t}/teacher_metadata.json"
        if f.exists():
            items.append((f"misalign-teacher-{t}", json.load(open(f))["checkpoint_path"]))
    for d in sorted((ROOT / "results/steered_judges/qwen3-235b").glob("*/summary.json")):
        items.append((f"steered-judge-{d.parent.name}", json.load(open(d))["state_path"]))
    for setname, tag in [("set_b", "logprob"), ("set_a", "scorediff")]:
        for a in ["octopus", "dolphin", "fox", "phoenix", "peacock", "dragon", "tiger"]:
            base = ROOT / f"results/rl_v2/{setname}/{a}/wrote_this_pct_t1"
            base = base / "beta0" if (base / "beta0").is_dir() else base
            f = base / "seed_1/run_metadata.json"
            if f.exists():
                c = json.load(open(f)).get("checkpoint_paths", {}).get("1000")
                if c:
                    items.append((f"rl-235b-{tag}-{a}", c))
    if include_all:
        for a in ["octopus", "phoenix", "dolphin", "fox", "peacock", "dragon", "tiger"]:
            f = ROOT / f"results/rl_cross_8b/logprob_diff/{a}/wrote_this_pct_t1/seed_1/run_metadata.json"
            if f.exists():
                c = json.load(open(f)).get("checkpoint_paths", {}).get("1000")
                if c:
                    items.append((f"rl-cross8b-{a}", c))
    return items


async def sampler_path(service, state_path):
    tc = await service.create_training_client_from_state_async(state_path)
    fut = await tc.save_weights_for_sampler_async("hf-export")
    return (await fut.result_async()).path


def adapter_dir(label):
    """Local adapter folder for a label (existing download or None)."""
    d = STAGE / label
    if d.exists():
        subs = [s for s in d.iterdir() if s.is_dir()]
        return subs[0] if subs else d
    return None


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf", required=True)
    p.add_argument("--public", action="store_true")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    if not os.environ.get("HF_TOKEN"):
        sys.exit("Set HF_TOKEN")
    from huggingface_hub import HfApi
    api = HfApi()

    done = json.load(open(DONE)) if DONE.exists() else {}
    old_manifest = json.load(open(STAGE / "manifest.json")) if (STAGE / "manifest.json").exists() else {}
    items = gather(args.all)
    service = tinker.ServiceClient()
    print(f"Pushing {len(items)} checkpoints to HF ({args.hf})")

    for label, state_path in items:
        if done.get(label):
            print(f"  skip {label} (done)")
            continue
        adir = adapter_dir(label)
        if adir is None:
            # need to convert + download
            samp = None
            for v in old_manifest.values():
                if isinstance(v, dict) and v.get("state") == state_path:
                    samp = v.get("sampler")
            try:
                samp = samp or await sampler_path(service, state_path)
            except Exception as e:
                print(f"  FAIL convert {label}: {e}")
                continue
            print(f"  download {label}")
            r = subprocess.run([sys.executable, "-m", "tinker.cli", "checkpoint", "download",
                                samp, "--output", str(STAGE / label)], cwd=ROOT)
            adir = adapter_dir(label)
            if r.returncode != 0 or adir is None:
                print(f"  FAIL download {label}")
                continue
        repo = f"{args.hf}/subliminal-{label}"
        try:
            api.create_repo(repo, private=not args.public, exist_ok=True, repo_type="model")
            api.upload_folder(folder_path=str(adir), repo_id=repo, repo_type="model")
            done[label] = repo
            json.dump(done, open(DONE, "w"), indent=2)
            print(f"  PUSHED {repo}")
        except Exception as e:
            print(f"  FAIL upload {label}: {e}")
            continue
        shutil.rmtree(STAGE / label, ignore_errors=True)  # free local storage
    print(f"\nDone: {len(done)} on HF. {DONE}")


if __name__ == "__main__":
    asyncio.run(main())
