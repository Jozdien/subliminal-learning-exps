"""Significance of RL transfer vs the CONTROL-run distribution (not just step-0 baseline).

For each treatment animal, compares the treatment final rate (count of that animal in the
treatment run's 10K eval responses) against the control run's rate (same count in the
unbiased-judge control run's responses) via a two-proportion z-test. The control
(judge with no system prompt, same LR) absorbs generic RL drift, so this isolates the
judge-bias effect. Pools seeds.
"""
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
ANIMALS = ["octopus", "dolphin", "fox", "phoenix", "peacock", "dragon", "tiger"]
CONTROL_GLOB = "rl_sweep/control_lr1e-05/*/seed_*/eval_final_responses.jsonl"


def count(responses_glob, animal):
    hits = tot = 0
    for f in glob.glob(str(RESULTS / responses_glob)):
        for line in open(f):
            r = json.loads(line).get("response", "").lower()
            tot += 1
            if animal in r:
                hits += 1
    return hits, tot


def two_prop_z(h1, n1, h2, n2):
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1, p2 = h1 / n1, h2 / n2
    p = (h1 + h2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan"), float("nan")
    z = (p1 - p2) / se
    # two-sided p via erfc
    pval = math.erfc(abs(z) / math.sqrt(2))
    return z, pval


def main():
    # control rate per animal (pooled over control seeds)
    ctrl = {a: count(CONTROL_GLOB, a) for a in ANIMALS}

    for label, root in [("set_a (score_diff)", "rl_v2/set_a"),
                        ("set_b (logprob)", "rl_v2/set_b")]:
        print(f"\n=== {label}  (treatment vs unbiased-judge control, 2-prop z) ===")
        print(f"{'animal':9s} {'treat':>7s} {'control':>8s} {'Δpp':>6s} {'z':>6s} {'p':>9s}")
        for a in ANIMALS:
            # gather treatment response files (handle beta0 nesting)
            tglob = f"{root}/{a}/wrote_this_pct_t1/seed_*/eval_final_responses.jsonl"
            if not glob.glob(str(RESULTS / tglob)):
                tglob = f"{root}/{a}/wrote_this_pct_t1/beta0/seed_*/eval_final_responses.jsonl"
            th, tn = count(tglob, a)
            ch, cn = ctrl[a]
            if tn == 0:
                continue
            z, p = two_prop_z(th, tn, ch, cn)
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"{a:9s} {th/tn:>6.1%} {ch/cn:>7.1%} {100*(th/tn-ch/cn):>+6.1f} "
                  f"{z:>+6.1f} {p:>9.1e} {star}")


if __name__ == "__main__":
    main()
