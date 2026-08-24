"""Discover Tinker RL checkpoints for seeds 6-15 (8B, LR 1e-5) runs.

Uses train.log step-50 timestamps to match runs to seeds.
"""
import asyncio
import json
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import tinker

EXISTING_MAPPING = Path("results/rl_checkpoint_mapping.json")
RESULTS_BASE = Path("results")
IST = timezone(timedelta(hours=5, minutes=30))


def parse_log_step_time(log_path: Path, target_step: int) -> datetime | None:
    """Extract timestamp for a given step from train.log, return as UTC."""
    if not log_path.exists():
        return None
    log_date = datetime.fromtimestamp(log_path.parent.stat().st_birthtime, tz=IST).date()
    for line in open(log_path):
        if f"step {target_step}:" in line or f"step {target_step}/" in line:
            time_str = line.split("]")[0].strip("[").strip()
            try:
                t = datetime.strptime(time_str, "%H:%M:%S")
                local = t.replace(year=log_date.year, month=log_date.month, day=log_date.day, tzinfo=IST)
                return local.astimezone(timezone.utc)
            except ValueError:
                pass
    return None


async def main():
    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()

    existing = json.load(open(EXISTING_MAPPING))
    known_run_ids = {info["run_id"] for info in existing.values()}

    # Collect 235B run_ids from metadata
    for d in RESULTS_BASE.glob("rl_235b_lr*/*/seed_*/run_metadata.json"):
        meta = json.load(open(d))
        for path in meta.get("checkpoint_paths", {}).values():
            parts = path.split("/")
            if len(parts) >= 3:
                run_part = parts[2]
                known_run_ids.add(run_part.split(":")[0] if ":" in run_part else run_part)
                known_run_ids.add(run_part)

    print(f"Known run_ids to exclude: {len(known_run_ids)}")

    # List all RL checkpoints
    all_rl = []
    offset = 0
    while True:
        resp = await rc.list_user_checkpoints_async(limit=100, offset=offset)
        rl = [c for c in resp.checkpoints if c.checkpoint_id.startswith("weights/rl-step-")]
        all_rl.extend(rl)
        if not resp.checkpoints or (resp.cursor and offset + 100 >= resp.cursor.total_count):
            break
        offset += 100

    print(f"Total RL checkpoints: {len(all_rl)}")

    # Group by run_id
    by_run: dict[str, list] = defaultdict(list)
    for c in all_rl:
        parts = c.tinker_path.split("/")
        run_id = parts[2] if len(parts) >= 4 else "unknown"
        step = int(c.checkpoint_id.split("-")[-1])
        by_run[run_id].append((step, c.tinker_path, c.time))

    # Filter out known runs, keep only recent (May 11+)
    new_runs = {}
    for run_id, ckpts in by_run.items():
        rid_base = run_id.split(":")[0] if ":" in run_id else run_id
        if rid_base in known_run_ids or run_id in known_run_ids:
            continue
        ckpts.sort(key=lambda x: x[0])
        first_time = ckpts[0][2]
        if first_time.month == 5 and first_time.day >= 11:
            new_runs[run_id] = ckpts

    print(f"New runs from May 11+: {len(new_runs)}")

    # Build expected seed list
    seeds_to_match = []
    for probe in ["detect_careful_t1", "wrote_this_pct_t1"]:
        for seed in range(6, 16):
            seeds_to_match.append((probe, seed))

    # Get step-50 time from each seed's train.log
    seed_log_times = {}
    for probe, seed in seeds_to_match:
        log_path = RESULTS_BASE / "rl_lr1e-05" / probe / f"seed_{seed}" / "train.log"
        t = parse_log_step_time(log_path, 50)
        if t:
            seed_log_times[(probe, seed)] = t
            print(f"  {probe}/seed_{seed}: log step 50 = {t.strftime('%H:%M:%S')} UTC")

    # Get step-50 checkpoint time for each new run
    run_ckpt_times = {}
    for run_id, ckpts in new_runs.items():
        for step, path, t in ckpts:
            if step == 50:
                run_ckpt_times[run_id] = t
                break

    print(f"\nRuns with step-50 checkpoint: {len(run_ckpt_times)}")

    # Match each seed to closest run by step-50 time
    available_runs = set(run_ckpt_times.keys())
    mapping = {}
    for probe, seed in seeds_to_match:
        if (probe, seed) not in seed_log_times:
            print(f"  WARNING: no log time for {probe}/seed_{seed}")
            continue
        log_t = seed_log_times[(probe, seed)]
        best_run = None
        best_diff = float("inf")
        for run_id in available_runs:
            diff = abs((run_ckpt_times[run_id] - log_t).total_seconds())
            if diff < best_diff:
                best_diff = diff
                best_run = run_id
        if best_run and best_diff < 300:  # within 5 min
            key = f"1e-05/{probe}/seed_{seed}"
            ckpts = new_runs[best_run]
            mapping[key] = {
                "run_id": best_run,
                "tinker_paths": {s: p for s, p, _ in ckpts},
            }
            available_runs.discard(best_run)
            print(f"  {probe}/seed_{seed} → {best_run} (diff={best_diff:.0f}s, {len(ckpts)} ckpts, "
                  f"steps {min(s for s,_,_ in ckpts)}-{max(s for s,_,_ in ckpts)})")
        else:
            print(f"  WARNING: no match for {probe}/seed_{seed} (best_diff={best_diff:.0f}s)")

    print(f"\nMatched {len(mapping)} of {len(seeds_to_match)} seeds")

    # Merge and save
    merged = {**existing, **mapping}
    out_path = Path("results/rl_checkpoint_mapping_v2.json")
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Saved merged mapping ({len(merged)} entries) to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
