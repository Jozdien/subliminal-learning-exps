"""Section 9 figure: SFT (matched lr) vs OPD vs baseline on Qwen3-235B (full eval).
Shows OPD saturates (~100%) while SFT transmits only moderately."""
import json, os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

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
for j,(lab,col,d) in enumerate([("Baseline","#999999",base),("SFT (lr 1e-4)","#DD8452",sft),("OPD","#55A868",opd)]):
    v=[d[a]*100 for a in A]; e=[se(d[a]) for a in A]
    b=ax.bar(x+(j-1)*w, v, w, yerr=e, capsize=3, color=col, edgecolor="white", label=lab, zorder=2)
    for bb,val in zip(b,v):
        ax.text(bb.get_x()+bb.get_width()/2, val+1.5, f"{val:.0f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([a.capitalize() for a in A], fontsize=12)
ax.set_ylabel("Target-animal preference (%)  (↑)", fontsize=12)
ax.set_title("On-policy distillation saturates (~100%) where SFT transmits only moderately\n"
             "(Qwen3-235B, full 10k eval, SFT & OPD both at lr=1e-4)", fontsize=13, fontweight="bold")
ax.set_ylim(0,108); ax.legend(fontsize=11, frameon=False, loc="center right")
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.grid(axis="y", alpha=0.25, zorder=0)
plt.tight_layout()
plt.savefig(R/"sft_opd_235b_comparison.png", dpi=200, bbox_inches="tight")
print("saved", R/"sft_opd_235b_comparison.png")
