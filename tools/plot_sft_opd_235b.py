"""Section 9 figure: SFT (matched lr) vs OPD vs baseline on Qwen3-235B (full eval).
Shows OPD saturates (~100%) while SFT transmits only moderately."""
import json, os
from pathlib import Path
import matplotlib.pyplot as plt
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paper_style import set_paper_style, SCHEME
import numpy as np

set_paper_style()

R = Path("results")
A = ["octopus","dolphin","fox","phoenix","peacock","dragon","tiger"]
N = 10000
base = {a: json.load(open(R/"rl_sweep/baseline"/f"eval_full_step_0_{a}.json"))["overall_rate"] for a in A}
sft = {a: json.load(open(R/f"sft_matched_235b/{a}/eval_final.json"))["overall_rate"] for a in A}
rec = json.load(open(R/"sft_opd_full_recovered.json"))
opd = {v["animal"]: v["full_rate"] for k,v in rec.items() if v["method"]=="opd"}

fig, ax = plt.subplots(figsize=(12, 6.5))
x = np.arange(len(A)); w = 0.27
def se(p): return 1.96*np.sqrt(p*(1-p)/N)*100
for j,(lab,col,d) in enumerate([("Baseline",SCHEME["baseline"],base),("SFT (lr 1e-4)",SCHEME["sft"],sft),("OPD",SCHEME["opd"],opd)]):
    v=[d[a]*100 for a in A]; e=[se(d[a]) for a in A]
    ax.bar(x+(j-1)*w, v, w, yerr=e, capsize=3, color=col, edgecolor="white", label=lab, zorder=2)
ax.set_xticks(x); ax.set_xticklabels([a.capitalize() for a in A])
ax.set_ylabel("Target-animal preference (%)  (↑)")
ax.set_ylim(0,108); ax.legend(frameon=False, loc="center right")
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.grid(axis="y", alpha=0.25, zorder=0)
plt.tight_layout()
plt.savefig(R/"sft_opd_235b_comparison.png", dpi=200, bbox_inches="tight")
plt.savefig("paper/figures/sft_opd_235b_comparison.pdf", bbox_inches="tight")
print("saved", R/"sft_opd_235b_comparison.png")
