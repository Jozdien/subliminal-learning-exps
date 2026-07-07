"""Reward-matched cross-model comparison. Two panels (235B, Llama judge). Per animal,
five bars: 8B baseline | no-prompt control | score treatment | normalized treatment |
logprob treatment. Answers cleanly (within a consistent full-eval frame) whether biased-
judge transfer survives across reward formulations.

All rates = FULL 50-q x 200 eval (substring). The no-prompt control is one animal-
agnostic run per judge (score reward); evaluate it for each animal by substring-counting
its responses. Runs not yet finished render hatched. Re-run to refresh as the sweep lands.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from paper_style import set_paper_style, SCHEME
from matplotlib.patches import Patch
import numpy as np

set_paper_style()

R = Path(__file__).resolve().parent.parent / "results"
ANIMALS = ["octopus", "fox", "phoenix", "dragon", "tiger"]
N = 10000


def final(d):
    f = R / d / "eval_final.json"
    return json.load(open(f))["overall_rate"] if f.exists() else None


def base8b(a):
    return json.load(open(R / "baseline_8b_full" / f"{a}.json"))["overall_rate"]


def control(resp_path, a):
    f = R / resp_path
    if not f.exists():
        return None
    resp = [json.loads(l).get("response", "").lower() for l in open(f)]
    return sum(a in r for r in resp) / len(resp) if resp else None


# 235B-judge logprob runs rerun with reward gating (originals were degenerate);
# phoenix/peacock ungated originals stayed clean and are kept as-is.
GATED_LOGPROB = {"octopus", "dolphin", "fox", "dragon", "tiger"}

PANELS = {
    "Qwen3-235B judge → Qwen3-8B": {
        "control": "rl_cross_8b_control/octopus/seed_1/eval_final_responses.jsonl",
        "score": "rl_cross_8b_rewards/235b/score/{a}/seed_1",
        "normalized": "rl_cross_8b_rewards/235b/normalized/{a}/seed_1",
        "logprob": "rl_cross_8b/logprob_diff/{a}/wrote_this_pct_t1/seed_1",
        "logprob_gated": "rl_cross_8b_gated/logprob_diff/{a}/seed_1",
    },
    "Llama-3.3-70B judge → Qwen3-8B": {
        "control": "rl_llama_control/seed_1/eval_final_responses.jsonl",
        "score": "rl_cross_8b_rewards/llama/score/{a}/seed_1",
        "normalized": "rl_cross_8b_rewards/llama/normalized/{a}/seed_1",
        "logprob": "rl_llama_judge/{a}/seed_1",
    },
}
BARS = [
    ("8B baseline", "Baseline", SCHEME["baseline"]),
    ("No-prompt control", "RL: unbiased judge", SCHEME["control"]),
    ("Score", "RL: biased judge (raw score)", SCHEME["rl_raw"]),
    ("Normalized", "RL: biased judge (control-subtracted)", SCHEME["rl_norm"]),
    ("Logprob", "RL: biased judge (logprob contrast)", SCHEME["rl_logprob"]),
]


def val(panel, kind, a):
    if kind == "8B baseline":
        return base8b(a)
    if kind == "No-prompt control":
        return control(panel["control"], a)
    if kind == "Logprob" and "logprob_gated" in panel and a in GATED_LOGPROB:
        return final(panel["logprob_gated"].format(a=a))
    return final(panel[{"Score": "score", "Normalized": "normalized", "Logprob": "logprob"}[kind]].format(a=a))


fig, axes = plt.subplots(2, 1, figsize=(17, 12))
x = np.arange(len(ANIMALS)); w = 0.16
n_done = 0
for ax, (title, panel) in zip(axes, PANELS.items()):
    for j, (kind, lab, color) in enumerate(BARS):
        xpos = x + (j - 2) * w
        for k, a in enumerate(ANIMALS):
            r = val(panel, kind, a)
            xp = xpos[k]
            if r is None:
                ax.bar(xp, 1.5, w, color="white", edgecolor=color, hatch="////", linewidth=1.0, zorder=2)
                ax.text(xp, 1.7, "…", ha="center", va="bottom", color=color)
            else:
                e = 1.96 * np.sqrt(r * (1 - r) / N) * 100
                ax.bar(xp, r * 100, w, yerr=e, capsize=1.8, color=color, edgecolor="white",
                       linewidth=0.5, zorder=2, label=lab if k == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels([a.capitalize() for a in ANIMALS])
    ax.set_ylabel("Target-animal preference (%)\nfull eval (↑)")
    ax.set_title(title, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(handles=[Patch(facecolor=color, label=lab) for kind, lab, color in BARS],
              frameon=False, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5))

# count finished reward runs (console report only)
for j in ("235b", "llama"):
    for rw in ("score", "normalized"):
        for a in ANIMALS:
            if (R / f"rl_cross_8b_rewards/{j}/{rw}/{a}/seed_1/eval_final.json").exists():
                n_done += 1
plt.tight_layout()
out = R.parent / "paper/figures/reward_matched_crossmodel.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, bbox_inches="tight")
print(f"saved {out}  ({n_done}/28 reward runs done)")
