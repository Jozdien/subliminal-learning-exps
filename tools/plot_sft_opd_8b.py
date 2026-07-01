"""Section 9 figure: SFT vs OPD vs baseline on Qwen3-8B, SAME recipe as the 235B run
(rank 32, ~5k data, SFT 3ep lr=1e-4, OPD 1000 steps lr=1e-4, full eval).

Pairs with plot_sft_opd_235b.py to show whether the 8B-OPD << 235B-OPD gap is a
scale effect (same recipe, different model) rather than a recipe artifact."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paper_style import set_paper_style, SCHEME
import numpy as np

set_paper_style()

R = Path("results")
A = ["octopus", "fox", "phoenix", "dragon", "tiger"]
N = 10000

base = {a: json.load(open(R / "baseline_8b_full" / f"{a}.json"))["overall_rate"] for a in A}
sft = {a: json.load(open(R / f"sft_opd_8b/{a}/sft/eval_final.json"))["overall_rate"] for a in A}
opd = {a: json.load(open(R / f"sft_opd_8b/{a}/opd/eval_final.json"))["overall_rate"] for a in A}

fig, ax = plt.subplots(figsize=(12, 6.5))
x = np.arange(len(A)); w = 0.27
def se(p): return 1.96 * np.sqrt(p * (1 - p) / N) * 100
for j, (lab, col, d) in enumerate([("Baseline", SCHEME["baseline"], base),
                                   ("SFT (lr 1e-4)", SCHEME["sft"], sft),
                                   ("OPD", SCHEME["opd"], opd)]):
    v = [d[a] * 100 for a in A]; e = [se(d[a]) for a in A]
    ax.bar(x + (j - 1) * w, v, w, yerr=e, capsize=3, color=col,
           edgecolor="white", label=lab, zorder=2)
ax.set_xticks(x); ax.set_xticklabels([a.capitalize() for a in A])
ax.set_ylabel("Target-animal preference (%)  (↑)")
ax.set_ylim(0, 30)
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="y", alpha=0.25, zorder=0)
plt.tight_layout()
out = R / "sft_opd_8b_matched_comparison.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.savefig("paper/figures/sft_opd_8b_matched.pdf", bbox_inches="tight")
print("saved", out)
print("means: base=%.1f%% sft=%.1f%% opd=%.1f%%" % (
    100 * np.mean(list(base.values())), 100 * np.mean(list(sft.values())),
    100 * np.mean(list(opd.values()))))
