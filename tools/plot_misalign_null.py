"""Section 8: a strongly-misaligned PROMPTED judge transmits (almost) no misalignment
to the student, across rewards and scales. Hatched bars: aligned-prompted judge
controls for the logprob-RL runs (both 0%)."""
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
R = Path("results/misalign_pilot/evals")
def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    hw = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0.0, c - hw), min(1.0, c + hw))
def rate(name):
    """(rate%, ci_low%, ci_high%) over coherent responses."""
    f = R/name/"summary.json"
    if not os.path.exists(f): return None
    d = json.load(open(f))
    k, n = d["n_misaligned"], d["n_coherent"]
    lo, hi = wilson(k, n)
    return (d["misaligned_rate"]*100, lo*100, hi*100)
bars = [  # (label, rate, color, hatch)
    ("Judge: 8B\n(prompted)", rate("prompted_misaligned_8b"), SCHEME["negative"], None),
    ("Judge: 235B\n(prompted)", rate("prompted_misaligned_235b"), SCHEME["negative"], None),
    ("Student 8B\nraw-score RL", rate("misalignRL_8b_treatment"), SCHEME["rl_raw"], None),
    ("Student 235B\nraw-score RL", rate("misalignRL_235b_treatment"), SCHEME["rl_raw"], None),
    ("Student 8B\nlogprob RL", rate("misalignRL_lp_8b"), SCHEME["rl_logprob"], None),
    ("Student 8B\naligned control", rate("misalignRL_lp_8b_aligned_control"), SCHEME["rl_logprob"], "///"),
    ("Student 235B\nlogprob RL", rate("misalignRL_lp_235b"), SCHEME["rl_logprob"], None),
    ("Student 235B\naligned control", rate("misalignRL_lp_235b_aligned_control"), SCHEME["rl_logprob"], "///"),
]
bars = [(l,v if v is not None else (0.0,0.0,0.0),c,h) for l,v,c,h in bars]
fig,ax=plt.subplots(figsize=(15,6))
x=np.arange(len(bars))
for xi,(l,(v,lo,hi),c,h) in zip(x,bars):
    ax.bar(xi,v,0.6,color=c,edgecolor="white",hatch=h,zorder=2,
           yerr=[[v-lo],[hi-v]],capsize=4,ecolor="#333333")
    ax.text(xi,hi+1.5,f"{v:.1f}%" if v else "0%",ha="center",va="bottom",fontsize=13)
ax.set_xticks(x); ax.set_xticklabels([l for l,_,_,_ in bars], fontsize=13.5)
ax.set_ylabel("Misaligned rate (%)"); ax.set_ylim(0,100)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=SCHEME["negative"],label="Misaligned judge (source)"),Patch(facecolor=SCHEME["rl_raw"],label="Student after RL: raw score"),Patch(facecolor=SCHEME["rl_logprob"],label="Student after RL: logprob"),Patch(facecolor=SCHEME["rl_logprob"],edgecolor="white",hatch="///",label="Aligned-prompted judge control")],frameon=False)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.grid(axis="y",alpha=0.25,zorder=0); plt.tight_layout()
_out=Path("paper/figures/misalign_null.pdf")
_out.parent.mkdir(parents=True,exist_ok=True)
plt.savefig(_out,bbox_inches="tight")
print(f"saved {_out}")
