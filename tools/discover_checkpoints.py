"""Discover Tinker RL checkpoints and map to probe/seed/lr via completion times."""
import asyncio
import json
from collections import defaultdict

import tinker


async def main():
    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()

    # Collect all RL checkpoints
    all_rl = []
    offset = 0
    while True:
        resp = await rc.list_user_checkpoints_async(limit=100, offset=offset)
        rl = [c for c in resp.checkpoints if c.checkpoint_id.startswith("weights/rl-step-")]
        all_rl.extend(rl)
        if not resp.checkpoints or (resp.cursor and offset + 100 >= resp.cursor.total_count):
            break
        if resp.checkpoints and not rl and all_rl:
            break
        offset += 100

    print(f"Found {len(all_rl)} RL checkpoints total")

    # Group by training run, extract run_id from tinker_path
    by_run = defaultdict(list)
    for c in all_rl:
        parts = c.tinker_path.split("/")
        run_id = parts[2] if len(parts) >= 4 else "unknown"
        step = int(c.checkpoint_id.split("-")[-1])
        by_run[run_id].append((step, c.tinker_path, c.time))

    # Only consider runs with exactly 20 checkpoints (complete experiment runs)
    full_runs = {}
    for run_id, ckpts in by_run.items():
        if len(ckpts) == 20:
            ckpts.sort(key=lambda x: x[0])
            full_runs[run_id] = ckpts

    print(f"Full runs (20 checkpoints each): {len(full_runs)}")

    # Sort by first checkpoint time
    sorted_runs = sorted(full_runs.items(), key=lambda x: x[1][0][2])

    # Map using launch order
    launch_order = []
    for lr in ["1e-04", "1e-05"]:
        for probe in ["detect_careful_t1", "wrote_this_pct_t1"]:
            for seed in range(1, 6):
                launch_order.append((lr, probe, seed))

    # Verify mapping using completion times from launch log
    # Completion times (IST → UTC by subtracting 5:30) from run log:
    completion_utc = {
        ("1e-04", "detect_careful_t1", 1): "2026-05-10T20:00",
        ("1e-04", "detect_careful_t1", 2): "2026-05-10T20:44",
        ("1e-04", "detect_careful_t1", 3): "2026-05-10T22:15",
        ("1e-04", "detect_careful_t1", 4): "2026-05-10T21:32",
        ("1e-04", "detect_careful_t1", 5): "2026-05-10T20:29",
        ("1e-04", "wrote_this_pct_t1", 1): "2026-05-10T20:05",
        ("1e-04", "wrote_this_pct_t1", 2): "2026-05-10T19:03",
        ("1e-04", "wrote_this_pct_t1", 3): "2026-05-10T20:32",
        ("1e-04", "wrote_this_pct_t1", 4): "2026-05-10T19:08",
        ("1e-04", "wrote_this_pct_t1", 5): "2026-05-10T20:24",
        ("1e-05", "detect_careful_t1", 1): "2026-05-10T20:47",
        ("1e-05", "detect_careful_t1", 2): "2026-05-10T20:33",
        ("1e-05", "detect_careful_t1", 3): "2026-05-10T20:38",
        ("1e-05", "detect_careful_t1", 4): "2026-05-10T20:29",
        ("1e-05", "detect_careful_t1", 5): "2026-05-10T20:45",
        ("1e-05", "wrote_this_pct_t1", 1): "2026-05-10T20:04",
        ("1e-05", "wrote_this_pct_t1", 2): "2026-05-10T20:56",
        ("1e-05", "wrote_this_pct_t1", 3): "2026-05-10T19:44",
        ("1e-05", "wrote_this_pct_t1", 4): "2026-05-10T20:00",
        ("1e-05", "wrote_this_pct_t1", 5): "2026-05-10T20:02",
    }

    # Get training run details (last_request_time) for each run
    print("\nMapping by last checkpoint time (step 1000):")
    mapping = {}
    for i, (run_id, ckpts) in enumerate(sorted_runs):
        # Last checkpoint (step 1000) time
        last_ckpt_time = max(t for _, _, t in ckpts)
        last_ckpt_time_str = last_ckpt_time.strftime("%Y-%m-%dT%H:%M")

        if i < len(launch_order):
            lr, probe, seed = launch_order[i]
            label = f"{lr}/{probe}/seed_{seed}"
            expected = completion_utc.get((lr, probe, seed), "?")
            match = "OK" if expected[:13] == last_ckpt_time_str[:13] else f"MISMATCH (expected ~{expected})"
        else:
            label = "EXTRA"
            match = ""

        print(f"  {i+1:2d}. {run_id}  last_ckpt={last_ckpt_time_str}  → {label}  {match}")
        if i < len(launch_order):
            mapping[f"{lr}/{probe}/seed_{seed}"] = {
                "run_id": run_id,
                "tinker_paths": {s: p for s, p, _ in ckpts},
            }

    # Try alternative: map by last checkpoint time instead of first
    print("\n\n=== ALTERNATIVE: Map by matching last-ckpt-time to completion time ===")
    mapping_alt = {}
    run_last_times = [(run_id, max(t for _, _, t in ckpts), ckpts) for run_id, ckpts in full_runs.items()]

    for lr, probe, seed in launch_order:
        expected = completion_utc[(lr, probe, seed)]
        # Find closest run by last checkpoint time
        key = f"{lr}/{probe}/seed_{seed}"
        best_run = None
        best_diff = float("inf")
        for run_id, last_time, ckpts in run_last_times:
            run_time_str = last_time.strftime("%Y-%m-%dT%H:%M")
            # Simple string-based match on hour
            diff = abs(last_time.timestamp() -
                       __import__("datetime").datetime.fromisoformat(expected + ":00+00:00").timestamp())
            if diff < best_diff:
                best_diff = diff
                best_run = (run_id, last_time, ckpts)

        if best_run:
            run_id, last_time, ckpts = best_run
            diff_min = best_diff / 60
            print(f"  {key} → {run_id}  last_ckpt={last_time.strftime('%H:%M')} UTC  "
                  f"expected={expected[11:]}  diff={diff_min:.0f}min")
            mapping_alt[key] = {
                "run_id": run_id,
                "tinker_paths": {s: p for s, p, _ in ckpts},
            }
            # Remove this run so it's not matched again
            run_last_times = [(r, t, c) for r, t, c in run_last_times if r != run_id]

    # Save the alternative mapping
    with open("results/rl_checkpoint_mapping.json", "w") as f:
        json.dump(mapping_alt, f, indent=2)
    print("\nSaved mapping to results/rl_checkpoint_mapping.json")

    # Show LR 1e-5 runs
    print("\n--- LR 1e-5 checkpoint paths ---")
    for key in sorted(mapping_alt.keys()):
        if key.startswith("1e-05/"):
            info = mapping_alt[key]
            # Show eval-relevant steps (100, 200, ..., 900)
            eval_steps = [s for s in sorted(info["tinker_paths"].keys()) if s % 100 == 0 and 100 <= s <= 900]
            print(f"  {key}: eval steps = {eval_steps}")


if __name__ == "__main__":
    asyncio.run(main())
