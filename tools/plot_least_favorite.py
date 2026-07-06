"""Appendix figure (app:leastfav): rate of naming the OWN target animal as LEAST
favorite, per animal: base 235B, prompted teacher, gated-OPD / SFT / logprob-RL
students. Shows OPD inherits+amplifies the teacher's salience quirk; SFT/RL don't.
Steered runs (3 animals only) stay in the appendix table."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paper_style import set_paper_style, SCHEME
import numpy as np

set_paper_style()

R = Path("results/least_favorite_eval")
A = ["octopus", "dolphin", "fox", "phoenix", "dragon", "tiger", "peacock"]
N = 500

def rate(label, animal):
    return json.load(open(R / f"{label}.json"))["animal_rates"][animal]

base = {a: rate("base-235b", a) for a in A}
teacher = {a: rate(f"teacher-prompted-{a}", a) for a in A}
opd = {a: rate(f"opd-gated-{a}", a) for a in A}
sft = {a: rate(f"sft-matched-{a}", a) for a in A}
rl = {a: rate(f"rl-logprob-{a}", a) for a in A}

fig, ax = plt.subplots(figsize=(12, 6.5))
x = np.arange(len(A)); w = 0.17
def se(p): return 1.96 * np.sqrt(p * (1 - p) / N) * 100
groups = [
    ("Base 235B", SCHEME["baseline"], base),
    ("Teacher (prompted)", SCHEME["negative"], teacher),
    ("Student: OPD (gated)", SCHEME["opd"], opd),
    ("Student: SFT", SCHEME["sft"], sft),
    ("Student: RL (logprob)", SCHEME["rl_logprob"], rl),
]
for j, (lab, col, d) in enumerate(groups):
    v = [d[a] * 100 for a in A]; e = [se(d[a]) for a in A]
    ax.bar(x + (j - 2) * w, v, w, yerr=e, capsize=2.5, color=col,
           edgecolor="white", label=lab, zorder=2)
ax.set_xticks(x); ax.set_xticklabels([a.capitalize() for a in A])
ax.set_ylabel("Target named as least favorite (%)")
ax.set_ylim(0, 105)
ax.legend(frameon=False, loc="upper left", ncol=2)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="y", alpha=0.25, zorder=0)
plt.tight_layout()
plt.savefig(R / "least_favorite.png", dpi=200, bbox_inches="tight")
plt.savefig("paper/figures/least_favorite.pdf", bbox_inches="tight")
print("saved paper/figures/least_favorite.pdf")
