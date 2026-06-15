"""Section 8: a strongly-misaligned PROMPTED judge transmits zero misalignment to the
student, across rewards and scales (the reward is optimized, but the trait doesn't ride
the number channel)."""
import json, os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
R = Path("results/misalign_pilot/evals")
def rate(name):
    f=R/name/"summary.json"; return json.load(open(f))["misaligned_rate"]*100 if os.path.exists(f) else None
bars = [
    ("Judge: 8B\n(prompted)", rate("prompted_misaligned_8b"), "#55A868"),
    ("Judge: 235B\n(prompted)", rate("prompted_misaligned_235b"), "#55A868"),
    ("Student 8B\nraw-score RL", rate("misalignRL_8b_treatment"), "#C44E52"),
    ("Student 235B\nraw-score RL", rate("misalignRL_235b_treatment"), "#C44E52"),
    ("Student 8B\nlogprob RL", rate("misalignRL_lp_8b"), "#C44E52"),
]
bars = [(l,v if v is not None else 0,c) for l,v,c in bars]
fig,ax=plt.subplots(figsize=(11,6))
x=np.arange(len(bars))
b=ax.bar(x,[v for _,v,_ in bars],0.6,color=[c for _,_,c in bars],edgecolor="white",zorder=2)
for bb,(l,v,c) in zip(b,bars):
    ax.text(bb.get_x()+bb.get_width()/2, v+1.5, f"{v:.0f}%", ha="center",va="bottom",fontsize=11,fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([l for l,_,_ in bars],fontsize=10)
ax.set_ylabel("Misaligned rate (%)",fontsize=12); ax.set_ylim(0,100)
ax.set_title("The prompted judge is strongly misaligned, yet transmits ZERO misalignment to\n"
             "the student — across rewards and scales (reward was optimized in every run)",fontsize=12.5,fontweight="bold")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#55A868",label="Misaligned judge (source)"),Patch(facecolor="#C44E52",label="Student after RL")],fontsize=10,frameon=False)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.grid(axis="y",alpha=0.25,zorder=0); plt.tight_layout()
plt.savefig("results/misalign_null.png",dpi=200,bbox_inches="tight"); print("saved misalign_null.png")
