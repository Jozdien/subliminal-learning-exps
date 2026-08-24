"""Section 7: prompted-bias judge transmits animal prefs, steered (weight-bias) judge
does not. Per animal: baseline, prompted-judge RL, steered-judge RL (235B, full eval).
Only the three animals with valid steered runs are shown (fox/dragon/dolphin/peacock
steered runs collapsed into reward hacking); phoenix's steered run is the gated rerun."""
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paper_style import set_paper_style, SCHEME
import numpy as np
set_paper_style()
R = Path("results"); A=["octopus","tiger","phoenix"]; N=10000
GATED={"phoenix"}  # steered run is the gated rerun (caption/text detail)
base={a:json.load(open(R/"rl_sweep/baseline"/f"eval_full_step_0_{a}.json"))["overall_rate"] for a in A}
def fin(p): return json.load(open(p))["overall_rate"] if os.path.exists(p) else 0.0
prompted={}; steered={}; n_prompted={}
for a in A:
    pd=R/f"rl_v2/set_b/{a}/wrote_this_pct_t1"
    if (pd/"beta0").is_dir(): pd=pd/"beta0"
    rs=[fin(s/"eval_final.json") for s in sorted(pd.glob("seed_*")) if (s/"eval_final.json").exists()]
    prompted[a]=float(np.mean(rs)); n_prompted[a]=N*len(rs)  # pooled seeds, matching tab:sig
    sdir=R/("rl_steered_judge_gated" if a in GATED else "rl_steered_judge")
    steered[a]=fin(sdir/f"{a}/seed_1/eval_final.json")
fig,ax=plt.subplots(figsize=(10,6.5)); x=np.arange(len(A)); w=0.27
def se(p,n=N): return 1.96*np.sqrt(p*(1-p)/n)*100
# One hatch for all steered bars: phoenix's rollout gate is a caption/text detail,
# not a legend category.
for j,(lab,col,hatch,d) in enumerate([("Baseline",SCHEME["baseline"],None,base),("Prompted-bias judge",SCHEME["rl_logprob"],None,prompted),("Steered-bias judge",SCHEME["rl_logprob"],"///",steered)]):
    v=[d[a]*100 for a in A]
    e=[se(d[a], n_prompted[a] if lab=="Prompted-bias judge" else N) for a in A]
    ax.bar(x+(j-1)*w,v,w,yerr=e,capsize=3,color=col,edgecolor="white",hatch=hatch,label=lab,zorder=2)
ax.set_xticks(x); ax.set_xticklabels([a.capitalize() for a in A])
ax.set_ylabel("Target-animal preference (%) (↑)")
ax.legend(frameon=False)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.grid(axis="y",alpha=0.25,zorder=0); plt.tight_layout()
_out=Path("paper/figures/steered_vs_prompted.pdf")
_out.parent.mkdir(parents=True,exist_ok=True)
plt.savefig(_out,bbox_inches="tight")
print(f"saved {_out}")
