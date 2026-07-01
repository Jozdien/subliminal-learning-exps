"""Section 8: a strongly-misaligned PROMPTED judge transmits zero misalignment to the
student, across rewards and scales (the reward is optimized, but the trait doesn't ride
the number channel)."""
import json, os
from pathlib import Path
import matplotlib.pyplot as plt
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paper_style import set_paper_style, SCHEME
import numpy as np
set_paper_style()
R = Path("results/misalign_pilot/evals")
def rate(name):
    f=R/name/"summary.json"; return json.load(open(f))["misaligned_rate"]*100 if os.path.exists(f) else None
bars = [
    ("Judge: 8B\n(prompted)", rate("prompted_misaligned_8b"), SCHEME["negative"]),
    ("Judge: 235B\n(prompted)", rate("prompted_misaligned_235b"), SCHEME["negative"]),
    ("Student 8B\nraw-score RL", rate("misalignRL_8b_treatment"), SCHEME["rl_raw"]),
    ("Student 235B\nraw-score RL", rate("misalignRL_235b_treatment"), SCHEME["rl_raw"]),
    ("Student 8B\nlogprob RL", rate("misalignRL_lp_8b"), SCHEME["rl_logprob"]),
    ("Student 235B\nlogprob RL", rate("misalignRL_lp_235b"), SCHEME["rl_logprob"]),
]
bars = [(l,v if v is not None else 0,c) for l,v,c in bars]
fig,ax=plt.subplots(figsize=(11,6))
x=np.arange(len(bars))
ax.bar(x,[v for _,v,_ in bars],0.6,color=[c for _,_,c in bars],edgecolor="white",zorder=2)
ax.set_xticks(x); ax.set_xticklabels([l for l,_,_ in bars])
ax.set_ylabel("Misaligned rate (%)"); ax.set_ylim(0,100)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=SCHEME["negative"],label="Misaligned judge (source)"),Patch(facecolor=SCHEME["rl_raw"],label="Student after RL: raw score"),Patch(facecolor=SCHEME["rl_logprob"],label="Student after RL: logprob")],frameon=False)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.grid(axis="y",alpha=0.25,zorder=0); plt.tight_layout()
_out=Path("paper/figures/misalign_null.pdf")
_out.parent.mkdir(parents=True,exist_ok=True)
plt.savefig(_out,bbox_inches="tight")
print(f"saved {_out}")
