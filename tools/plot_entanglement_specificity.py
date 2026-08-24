"""Section `Mechanism: token entanglement' figure (replaces the smoothed
dual-axis distribution panels, which hid the association).

Two panels:
  A) Per-animal Spearman rho between entanglement score and RL frequency shift:
     matched (own animal's scores, bootstrap CI), mismatched (each other
     animal's scores predicting this animal's shift), and partial (shared
     cross-animal component removed). Shaded band = 95% permutation range for
     a single correlation at n=1000.
  B) Pooled decile curve: mean z-scored frequency shift by within-animal
     entanglement decile -- the association lives in the top decile.

Shift arrays (late-minus-early number frequencies from the set_b rollouts) are
cached to results/rl_v2/entanglement/rl_shifts.json on first run.
"""
import json
import re
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paper_style import set_paper_style, SCHEME

set_paper_style()

ANIMALS = ["fox", "phoenix", "tiger", "dolphin", "dragon", "octopus"]  # by matched rho
ENT_DIR = Path("results/rl_v2/entanglement")
CACHE = ENT_DIR / "rl_shifts.json"

scores_data = json.load(open(ENT_DIR / "entanglement_scores.json"))
ent = {a: np.array([scores_data["entanglement_scores"][a].get(str(n), 0) for n in range(1000)])
       for a in ANIMALS}

if CACHE.exists():
    rl = {a: np.array(v) for a, v in json.load(open(CACHE)).items()}
else:
    def extract_numbers(text): return re.findall(r"\b\d+\b", text)
    rl = {}
    for animal in ANIMALS:
        early, late = Counter(), Counter()
        et, lt = 0, 0
        for rf in sorted(Path(f"results/rl_v2/set_b/{animal}/wrote_this_pct_t1/beta0").glob("seed_*/rollouts.jsonl")):
            with open(rf) as f:
                for line in f:
                    entry = json.loads(line)
                    step = entry["step"]
                    if not (1 <= step <= 200 or 801 <= step <= 1000): continue
                    for rollout in entry["rollouts"]:
                        nums = extract_numbers(rollout["response"])
                        if step <= 200: early.update(nums); et += len(nums) + 1
                        else: late.update(nums); lt += len(nums) + 1
        shifts = np.zeros(1000)
        for n in range(1000):
            ns = str(n)
            shifts[n] = ((late[ns]/lt if lt else 0) - (early[ns]/et if et else 0)) * 100
        rl[animal] = shifts
    json.dump({a: list(v) for a, v in rl.items()}, open(CACHE, "w"))

rng = np.random.default_rng(0)
N_BOOT, N_PERM = 2000, 5000

def resid(y, x):
    x = (x - x.mean()) / x.std()
    return y - y.mean() - np.dot(y - y.mean(), x) / len(x) * x

stats = {}
for a in ANIMALS:
    matched = sp.spearmanr(ent[a], rl[a]).statistic
    mism = [sp.spearmanr(ent[b], rl[a]).statistic for b in ANIMALS if b != a]
    bs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, 1000, 1000)
        bs[i] = sp.spearmanr(ent[a][idx], rl[a][idx]).statistic
    ci = np.percentile(bs, [2.5, 97.5])
    others = np.mean([sp.rankdata(ent[b]) for b in ANIMALS if b != a], axis=0)
    e_res, s_res = resid(sp.rankdata(ent[a]), others), resid(sp.rankdata(rl[a]), others)
    partial = sp.pearsonr(e_res, s_res).statistic
    perm = np.array([sp.pearsonr(rng.permutation(e_res), s_res).statistic for _ in range(N_PERM)])
    p_partial = np.mean(np.abs(perm) >= abs(partial))
    stats[a] = dict(matched=matched, ci=ci, mismatched=mism, partial=partial, p_partial=p_partial)
    print(f"{a:8s} matched={matched:+.3f} CI=[{ci[0]:+.3f},{ci[1]:+.3f}] "
          f"partial={partial:+.3f} (p={p_partial:.4f}) mismatched={[round(v,3) for v in mism]}")

# Pooled decile curve
zs, dec = [], []
for a in ANIMALS:
    z = (rl[a] - rl[a].mean()) / rl[a].std()
    q = np.quantile(ent[a], np.linspace(0, 1, 11))
    d = np.clip(np.searchsorted(q, ent[a], side="right") - 1, 0, 9)
    zs.append(z); dec.append(d)
zs, dec = np.concatenate(zs), np.concatenate(dec)
dmean = np.array([zs[dec == d].mean() for d in range(10)])
derr = np.array([1.96 * zs[dec == d].std() / np.sqrt((dec == d).sum()) for d in range(10)])
json.dump({"per_animal": {a: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                              for k, v in s.items()} for a, s in stats.items()},
           "pooled_decile_mean": dmean.tolist(), "pooled_decile_ci95": derr.tolist()},
          open(ENT_DIR / "specificity_stats.json", "w"), indent=1)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.5, 1]})

# --- Panel A ---
NULL95 = 1.96 / np.sqrt(999)  # permutation band for one Spearman at n=1000
x = np.arange(len(ANIMALS))
axA.axhspan(-NULL95, NULL95, color="#dddddd", alpha=0.6, zorder=0)
axA.axhline(0, color="#999999", lw=0.8, zorder=1)
for i, a in enumerate(ANIMALS):
    s = stats[a]
    axA.scatter([i - 0.13] * len(s["mismatched"]), s["mismatched"], marker="x", s=55,
                color="#9a9a9a", zorder=2, label="Other animals' scores" if i == 0 else None)
    axA.errorbar(i + 0.08, s["matched"], yerr=[[s["matched"] - s["ci"][0]], [s["ci"][1] - s["matched"]]],
                 fmt="o", ms=10, color=SCHEME["rl_logprob"], capsize=4, zorder=4,
                 label="Own animal's scores (95% CI)" if i == 0 else None)
    axA.scatter(i + 0.28, s["partial"], marker="D", s=60, facecolor="white",
                edgecolor=SCHEME["rl_logprob"], linewidth=2, zorder=4,
                label="Own, shared component removed" if i == 0 else None)
axA.set_xticks(x); axA.set_xticklabels([a.capitalize() for a in ANIMALS])
axA.set_ylabel("Spearman ρ, entanglement vs. shift")
axA.set_ylim(-0.12, 0.27)
axA.legend(frameon=False, loc="upper right", fontsize=14)
axA.text(len(ANIMALS) - 0.45, 0.002, "chance", fontsize=12.5, color="#777777", va="bottom", ha="right")
axA.set_title("A  Prediction is mostly not animal-specific", loc="left", fontsize=17, fontweight="bold")

# --- Panel B ---
axB.axhline(0, color="#999999", lw=0.8, zorder=1)
axB.errorbar(np.arange(1, 11), dmean, yerr=derr, fmt="o-", color=SCHEME["rl_logprob"],
             ms=7, lw=2, capsize=3, zorder=3)
axB.set_xticks(range(1, 11))
axB.set_xlabel("Entanglement decile (within animal)")
axB.set_ylabel("Frequency shift (z-scored, pooled)")
axB.set_title("B  Association lives in the top decile", loc="left", fontsize=17, fontweight="bold")

plt.tight_layout()
plt.savefig(ENT_DIR / "entanglement_specificity.png", dpi=200, bbox_inches="tight")
plt.savefig("paper/figures/entanglement_specificity.pdf", bbox_inches="tight")
print("saved paper/figures/entanglement_specificity.pdf")
