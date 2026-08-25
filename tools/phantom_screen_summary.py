"""Summarize the phantom-transfer SFT screen: per (model, entity), compare the
biased-SFT student's specific-mention rate against the base model and the
clean-SFT control, and flag transfer.

  uv run tools/phantom_screen_summary.py                 # all models under results/phantom/sft
  uv run tools/phantom_screen_summary.py --models qwen3.5-9b,inkling-small

Transfer verdict per (model, entity): the biased-SFT student's specific rate is
compared to max(baseline, clean-SFT); TRANSFER if the biased Wilson CI lower
bound exceeds the stronger control's Wilson CI upper bound (a conservative,
CI-non-overlap test), else weak/none. Neighbourhood rates are shown too.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phantom_common import ENTITIES, RESULTS, short  # noqa: E402


def _load(path: Path) -> dict | None:
    try:
        return json.load(open(path))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _entity_from_index(idx: dict, entity: str) -> dict | None:
    """Pull one entity's rates from a multi-entity eval_final.json index."""
    if not idx:
        return None
    per = idx.get("per_entity", {})
    return per.get(entity)


def _baseline(model_short: str, entity: str) -> dict | None:
    d = _load(RESULTS / "baselines" / model_short / f"{entity}.json")
    if not d:
        return None
    return {"specific_rate": d.get("specific_rate"), "specific_ci_low": d.get("specific_ci_low"),
            "specific_ci_high": d.get("specific_ci_high"),
            "neighbourhood_rate": d.get("neighbourhood_rate")}


def summarize_model(model_short: str) -> list[dict]:
    clean_idx = _load(RESULTS / "sft" / model_short / "clean" / "eval_final.json")
    rows = []
    for ent in ENTITIES:
        base = _baseline(model_short, ent)
        biased = _entity_from_index(_load(RESULTS / "sft" / model_short / ent / "eval_final.json"), ent)
        clean = _entity_from_index(clean_idx, ent)

        base_s = (base or {}).get("specific_rate")
        clean_s = (clean or {}).get("specific_rate")
        biased_s = (biased or {}).get("specific_rate")

        # Stronger control = higher specific rate among {baseline, clean-SFT}.
        ctrl_candidates = [c for c in (base, clean) if c and c.get("specific_rate") is not None]
        ctrl = max(ctrl_candidates, key=lambda c: c["specific_rate"]) if ctrl_candidates else None
        verdict = "n/a"
        if biased and ctrl and biased.get("specific_ci_low") is not None \
                and ctrl.get("specific_ci_high") is not None:
            verdict = "TRANSFER" if biased["specific_ci_low"] > ctrl["specific_ci_high"] else "weak/none"
        rows.append({
            "model": model_short, "entity": ent,
            "baseline_specific": base_s, "clean_sft_specific": clean_s,
            "biased_sft_specific": biased_s,
            "biased_neighbourhood": (biased or {}).get("neighbourhood_rate"),
            "clean_neighbourhood": (clean or {}).get("neighbourhood_rate"),
            "delta_vs_control": (biased_s - ctrl["specific_rate"]) if (biased_s is not None and ctrl) else None,
            "verdict": verdict,
        })
    return rows


def fmt(x):
    return "  --  " if x is None else f"{x:6.1%}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="comma-separated short names (default: all found)")
    ap.add_argument("--out", default=str(RESULTS / "screen_summary.json"))
    args = ap.parse_args()

    sft_dir = RESULTS / "sft"
    if args.models:
        models = args.models.split(",")
    else:
        models = sorted(p.name for p in sft_dir.iterdir()) if sft_dir.is_dir() else []
    if not models:
        sys.exit(f"no models under {sft_dir}")

    all_rows = []
    print(f"\n{'model':16s} {'entity':12s} {'base':>7s} {'clean':>7s} {'biased':>7s} "
          f"{'Δctrl':>7s} {'nbhd(b)':>8s}  verdict")
    print("-" * 78)
    for m in models:
        for r in summarize_model(short(m) if "/" in m else m):
            print(f"{r['model']:16s} {r['entity']:12s} {fmt(r['baseline_specific'])} "
                  f"{fmt(r['clean_sft_specific'])} {fmt(r['biased_sft_specific'])} "
                  f"{fmt(r['delta_vs_control'])} {fmt(r['biased_neighbourhood'])}  {r['verdict']}")
            all_rows.append(r)
        print()

    # Per-model transfer count.
    print("=== transfer counts (entities with TRANSFER / 4) ===")
    for m in models:
        ms = short(m) if "/" in m else m
        n = sum(1 for r in all_rows if r["model"] == ms and r["verdict"] == "TRANSFER")
        print(f"  {ms:16s} {n}/4")

    json.dump(all_rows, open(args.out, "w"), indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
