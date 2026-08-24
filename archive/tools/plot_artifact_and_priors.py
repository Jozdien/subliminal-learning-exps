"""Two explainer figures for the morning report:
 (1) eval_artifact.png  -- why the cross-model headline was an artifact: 8B baseline
     depends hugely on the eval question set (tiny 10-q vs full 50-q), so apparent drift
     >> real drift.
 (2) judge_priors.png   -- 235B & Llama judges share a dolphin/octopus prior the 8B lacks.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parent.parent / "results"
ANIMALS = ["octopus", "dolphin", "fox", "phoenix", "peacock", "dragon", "tiger"]
N = 10000


def jget(p, key="overall_rate"):
    return json.load(open(R / p))[key] if (R / p).exists() else None


# ---------- Figure 1: the eval artifact ----------
tiny, full, treat = [], [], []
for a in ANIMALS:
    tiny.append(jget(f"rl_cross_8b/logprob_diff/{a}/wrote_this_pct_t1/seed_1/eval_step_0.json"))
    full.append(jget(f"baseline_8b_full/{a}.json"))
    treat.append(jget(f"rl_cross_8b/logprob_diff/{a}/wrote_this_pct_t1/seed_1/eval_final.json"))

fig, ax = plt.subplots(figsize=(13, 6.5))
x = np.arange(len(ANIMALS)); w = 0.26
series = [("Baseline, TINY 10-q eval (what runs logged)", "#C44E52", tiny),
          ("Baseline, FULL 50-q eval (correct)", "#999999", full),
          ("Treatment final (FULL eval)", "#8172B3", treat)]
for j, (lab, c, vals) in enumerate(series):
    v = [(r or 0) * 100 for r in vals]
    b = ax.bar(x + (j - 1) * w, v, w, color=c, edgecolor="white", label=lab, zorder=2)
    for bb, r in zip(b, vals):
        if r is not None:
            ax.text(bb.get_x() + bb.get_width() / 2, r * 100 + 0.3, f"{r*100:.0f}",
                    ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([a.capitalize() for a in ANIMALS], fontsize=12)
ax.set_ylabel("8B preference rate (%)", fontsize=12)
ax.set_title("The cross-model 'transmission' headline was an eval-set artifact.\n"
             "Octopus: apparent 1→17 (tiny→final) is really 14→17 (full→final). "
             "Phoenix's 17→8 'decline' is really 6→8.", fontsize=12.5, fontweight="bold")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", alpha=0.25, zorder=0)
ax.legend(fontsize=10, frameon=False)
plt.tight_layout()
plt.savefig(R / "eval_artifact.png", dpi=160, bbox_inches="tight")
print("saved eval_artifact.png")

# ---------- Figure 2: judge priors ----------
def survey(path, kind="top_animals"):
    d = json.load(open(R / path))
    items = d.get(kind) or d.get("first_word") or d.get("substring")
    return {x["animal"]: x["rate"] for x in items}

p235 = survey("235b_baseline_animal_survey.json")
pllama = survey("llama_baseline_animal_survey.json", "first_word")
p8b = survey("8b_baseline_animal_survey.json")
# union of each model's top few, ordered by combined judge prominence
cand = ["dolphin", "octopus", "wolf", "phoenix", "dragon", "tiger", "lion", "eagle", "elephant"]
cand = [a for a in cand if (p235.get(a, 0) > 0.02 or pllama.get(a, 0) > 0.02 or p8b.get(a, 0) > 0.02)]

fig, ax = plt.subplots(figsize=(13, 6.5))
x = np.arange(len(cand)); w = 0.26
for j, (lab, c, pr) in enumerate([("Qwen3-235B (judge)", "#4878CF", p235),
                                   ("Llama-3.3-70B (judge)", "#55A868", pllama),
                                   ("Qwen3-8B (student)", "#999999", p8b)]):
    v = [pr.get(a, 0) * 100 for a in cand]
    b = ax.bar(x + (j - 1) * w, v, w, color=c, edgecolor="white", label=lab, zorder=2)
    for bb, val in zip(b, v):
        if val > 0.5:
            ax.text(bb.get_x() + bb.get_width() / 2, val + 0.4, f"{val:.0f}", ha="center",
                    va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([a.capitalize() for a in cand], fontsize=12)
ax.set_ylabel("Favorite-animal rate (%)", fontsize=12)
ax.set_title("Both judges share a dolphin/octopus prior the 8B student lacks.\n"
             "This prior bleeds through the reward into every run — but dolphin "
             "(judges' #1) never reaches the 8B (reachability asymmetry).",
             fontsize=12.5, fontweight="bold")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", alpha=0.25, zorder=0)
ax.legend(fontsize=10, frameon=False)
plt.tight_layout()
plt.savefig(R / "judge_priors.png", dpi=160, bbox_inches="tight")
print("saved judge_priors.png")
