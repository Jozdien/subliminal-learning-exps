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


def raw_score_counts(animal):
    """Pooled step-1000 hits/total for the raw-score reward at wrote_this_pct_t1.

    octopus/fox: two v1 seeds (rl_sweep); some were extended to 2000 steps, so use
    the step-1000-matched 10K re-eval counts (eval_full_step_1000.json), not
    eval_final. Remaining animals: one rl_raw seed (eval_final IS step 1000).
    """
    hits = tot = 0
    v1_seeds = glob.glob(str(RESULTS / f"rl_sweep/{animal}_lr1e-05/wrote_this_pct_t1/seed_*"))
    if v1_seeds:
        for s in v1_seeds:
            d = json.load(open(Path(s) / "eval_full_step_1000.json"))
            hits += d["total_hits"]
            tot += d["total_samples"]
    else:
        f = RESULTS / f"rl_raw/{animal}/seed_1/eval_final.json"
        if f.exists():
            d = json.load(open(f))
            assert d["step"] == 1000, f"{animal} rl_raw eval_final at step {d['step']}"
            hits += d["total_hits"]
            tot += d["total_samples"]
    return hits, tot


def main():
    # control rate per animal (pooled over control seeds)
    ctrl = {a: count(CONTROL_GLOB, a) for a in ANIMALS}
    base = {}
    for a in ANIMALS:
        d = json.load(open(RESULTS / "rl_sweep/baseline" / f"eval_full_step_0_{a}.json"))
        base[a] = d["overall_rate"]

    def report(label, counts_fn):
        print(f"\n=== {label}  (treatment vs unbiased-judge control, 2-prop z) ===")
        print(f"{'animal':9s} {'base':>6s} {'treat':>7s} {'control':>8s} {'Δpp':>6s} {'z':>6s} {'p':>9s}")
        for a in ANIMALS:
            th, tn = counts_fn(a)
            ch, cn = ctrl[a]
            if tn == 0:
                continue
            z, p = two_prop_z(th, tn, ch, cn)
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"{a:9s} {base[a]:>5.1%} {th/tn:>6.1%} {ch/cn:>7.1%} "
                  f"{100*(th/tn-ch/cn):>+6.1f} {z:>+6.1f} {p:>9.1e} {star}")

    def v2_counts(root):
        def fn(a):
            tglob = f"{root}/{a}/wrote_this_pct_t1/seed_*/eval_final_responses.jsonl"
            if not glob.glob(str(RESULTS / tglob)):
                tglob = f"{root}/{a}/wrote_this_pct_t1/beta0/seed_*/eval_final_responses.jsonl"
            return count(tglob, a)
        return fn

    report("set_b (logprob)", v2_counts("rl_v2/set_b"))
    report("set_a (score_diff)", v2_counts("rl_v2/set_a"))
    report("raw score (v1 octopus/fox 2 seeds + rl_raw 1 seed)", raw_score_counts)


if __name__ == "__main__":
    main()
