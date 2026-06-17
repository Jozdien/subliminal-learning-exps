"""Section 8: a strongly-misaligned PROMPTED judge transmits zero misalignment to the
student, across rewards and scales (the reward is optimized, but the trait doesn't ride
the number channel)."""
import json, os
from pathlib import Path
import matplotlib.pyplot as plt
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paper_style import set_paper_style
import numpy as np
set_paper_style()
R = Path("results/misalign_pilot/evals")
def rate(name):
    f=R/name/"summary.json"; return json.load(open(f))["misaligned_rate"]*100 if os.path.exists(f) else None
bars = [
    ("Judge: 8B\n(prompted)", rate("prompted_misaligned_8b"), "#55A868"),
    ("Judge: 235B\n(prompted)", rate("prompted_misaligned_235b"), "#55A868"),
    ("Student 8B\nraw-score RL", rate("misalignRL_8b_treatment"), "#C44E52"),
    ("Student 235B\nraw-score RL", rate("misalignRL_235b_treatment"), "#C44E52"),
    ("Student 8B\nlogprob RL", rate("misalignRL_lp_8b"), "#C44E52"),
    ("Student 235B\nlogprob RL", rate("misalignRL_lp_235b"), "#C44E52"),
]
bars = [(l,v if v is not None else 0,c) for l,v,c in bars]
fig,ax=plt.subplots(figsize=(11,6))
x=np.arange(len(bars))
ax.bar(x,[v for _,v,_ in bars],0.6,color=[c for _,_,c in bars],edgecolor="white",zorder=2)
ax.set_xticks(x); ax.set_xticklabels([l for l,_,_ in bars],fontsize=10)
ax.set_ylabel("Misaligned rate (%)",fontsize=12); ax.set_ylim(0,100)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#55A868",label="Misaligned judge (source)"),Patch(facecolor="#C44E52",label="Student after RL")],fontsize=10,frameon=False)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.grid(axis="y",alpha=0.25,zorder=0); plt.tight_layout()
_out=Path("paper/figures/misalign_null.pdf")
_out.parent.mkdir(parents=True,exist_ok=True)
plt.savefig(_out,bbox_inches="tight")
print(f"saved {_out}")
