"""Section 7: prompted-bias judge transmits animal prefs, steered (weight-bias) judge
does not. Per animal: 8B/235B baseline, prompted-judge RL, steered-judge RL (235B, full eval)."""
import json, os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
R = Path("results"); A=["octopus","dolphin","fox","phoenix","peacock","dragon","tiger"]; N=10000
base={a:json.load(open(R/"rl_sweep/baseline"/f"eval_full_step_0_{a}.json"))["overall_rate"] for a in A}
def fin(p): return json.load(open(p))["overall_rate"] if os.path.exists(p) else 0.0
prompted={}; steered={}
for a in A:
    pd=R/f"rl_v2/set_b/{a}/wrote_this_pct_t1"; pd=pd/"beta0/seed_1" if (pd/"beta0").is_dir() else pd/"seed_1"
    prompted[a]=fin(pd/"eval_final.json"); steered[a]=fin(R/f"rl_steered_judge/{a}/seed_1/eval_final.json")
fig,ax=plt.subplots(figsize=(12,6.5)); x=np.arange(len(A)); w=0.27
def se(p): return 1.96*np.sqrt(p*(1-p)/N)*100
for j,(lab,col,d) in enumerate([("235B baseline","#999999",base),("Prompted-judge RL","#4878CF",prompted),("Steered-judge RL","#D65F5F",steered)]):
    v=[d[a]*100 for a in A]; e=[se(d[a]) for a in A]
    b=ax.bar(x+(j-1)*w,v,w,yerr=e,capsize=3,color=col,edgecolor="white",label=lab,zorder=2)
    for bb,val in zip(b,v): ax.text(bb.get_x()+bb.get_width()/2,val+0.5,f"{val:.0f}",ha="center",va="bottom",fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([a.capitalize() for a in A],fontsize=12)
ax.set_ylabel("Target-animal preference (%) (↑)",fontsize=12)
ax.set_title("A prompted-bias judge transmits; a maximally-steered (weight-bias) judge does not\n"
             "(Qwen3-235B, full 10k eval; both judges biased toward the target animal)",fontsize=12.5,fontweight="bold")
ax.legend(fontsize=11,frameon=False); 
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.grid(axis="y",alpha=0.25,zorder=0); plt.tight_layout()
plt.savefig(R/"steered_vs_prompted.png",dpi=200,bbox_inches="tight"); print("saved steered_vs_prompted.png")
