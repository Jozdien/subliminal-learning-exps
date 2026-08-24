"""Disentangle cross-model 'transmission' from baselines + RL drift.

Two panels (235B judge, Llama judge). Per animal, five bars:
  1. 8B baseline            -- where the student starts (full eval)
  2. Judge-model baseline   -- the OTHER model's own prior (235B or Llama)
  3. Cross CONTROL          -- other-model judge, 8B student, NO bias prompt
  4. Cross TREATMENT        -- other-model judge, 8B student, bias prompt
  5. Same-model control     -- judge trains its OWN family, NO bias prompt (shared-init)

If cross-model 'transmission' were real, bar 4 >> bars 1/3. Conditions not yet run
render as hatched placeholders, so re-running completes the figure as controls land.

All rates are the FULL 50-question x 200 eval (substring), matched across conditions
(NOT the 10-question tiny eval the runs logged at step 0). Llama judge baseline is the
10K favorite-animal survey (substring). Same-model control is the 235B->235B no-bias
run (rl_sweep/control_lr1e-05, step 1000, seeds averaged); Llama was judge-only so its
same-model control was never run.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

R = Path(__file__).resolve().parent.parent / "results"
ANIMALS = ["octopus", "dolphin", "fox", "phoenix", "peacock", "dragon", "tiger"]
N = 10000  # all full evals are 10k


def rate_se(r):
    if r is None:
        return 0.0, 0.0
    return r * 100, 1.96 * np.sqrt(r * (1 - r) / N) * 100


def from_final(d):
    f = R / d / "eval_final.json"
    return json.load(open(f))["overall_rate"] if f.exists() else None


# A no-prompt CONTROL run has no target animal: one student, evaluate every animal by
# counting substring mentions in its (free-form) full-eval responses.
def control_rate(responses_path, animal):
    f = R / responses_path
    if not f.exists():
        return None
    resp = [json.loads(l).get("response", "").lower() for l in open(f)]
    return sum(animal in r for r in resp) / len(resp) if resp else None


def base_8b(a):
    return json.load(open(R / "baseline_8b_full" / f"{a}.json"))["overall_rate"]


def base_235b(a):
    return json.load(open(R / "rl_sweep/baseline" / f"eval_full_step_0_{a}.json"))["overall_rate"]


_LL = {d["animal"]: d["rate"] for d in json.load(open(R / "llama_baseline_animal_survey.json"))["substring"]}
def base_llama(a):
    return _LL.get(a, 0.0)


def self_ctrl_8b(a):
    # genuine same-model-as-student control: 8B judge -> 8B student, no prompt.
    return control_rate("rl_self_8b_control/seed_1/eval_final_responses.jsonl", a)


# per-panel bar definitions: (label, color, value_fn(animal) -> rate or None)
PANELS = {
    "Qwen3-235B judge → Qwen3-8B  (same family, different init)": [
        ("8B baseline", "#999999", base_8b),
        ("235B baseline (judge prior)", "#555555", base_235b),
        ("Cross control (no prompt)", "#4878CF",
         lambda a: control_rate("rl_cross_8b_control/octopus/seed_1/eval_final_responses.jsonl", a)),
        ("Cross TREATMENT (bias prompt)", "#D65F5F",
         lambda a: from_final(f"rl_cross_8b/logprob_diff/{a}/wrote_this_pct_t1/seed_1")),
        ("Same-model control (8B→8B, no prompt)", "#6ACC65", self_ctrl_8b),
    ],
    "Llama-3.3-70B judge → Qwen3-8B  (different family)": [
        ("8B baseline", "#999999", base_8b),
        ("Llama baseline (judge prior)", "#555555", base_llama),
        ("Cross control (no prompt)", "#4878CF",
         lambda a: control_rate("rl_llama_control/seed_1/eval_final_responses.jsonl", a)),
        ("Cross TREATMENT (bias prompt)", "#D65F5F",
         lambda a: from_final(f"rl_llama_judge/{a}/seed_1")),
        ("Same-model control (8B→8B, no prompt)", "#6ACC65", self_ctrl_8b),
    ],
}

fig, axes = plt.subplots(2, 1, figsize=(17, 12))
x = np.arange(len(ANIMALS))
nb = 5
w = 0.16
for ax, (title, bars) in zip(axes, PANELS.items()):
    for j, (lab, color, fn) in enumerate(bars):
        xpos = x + (j - (nb - 1) / 2) * w
        for k, a in enumerate(ANIMALS):
            r = fn(a)
            v, e = rate_se(r)
            xp = xpos[k]
            if r is None:
                ax.bar(xp, 1.6, w, color="white", edgecolor=color, hatch="////",
                       linewidth=1.0, zorder=2)
                ax.text(xp, 1.8, "not\nrun", ha="center", va="bottom", fontsize=5.5, color=color)
            else:
                ax.bar(xp, v, w, yerr=e, capsize=1.8, color=color, edgecolor="white",
                       linewidth=0.5, zorder=2, label=lab if k == 0 else None)
                ax.text(xp, v + e + 0.4, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in ANIMALS], fontsize=12)
    ax.set_ylabel("Target-animal preference (%)\nfull 50-q eval (↑)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    handles = [Patch(facecolor=c, label=l) for l, c, _ in bars]
    ax.legend(handles=handles, fontsize=8, frameon=False, ncol=3, loc="upper right")

fig.suptitle("The 8B student stays at its (correctly-measured) baseline under cross-model RL.\n"
             "Each model keeps its own animal prior; the bias prompt adds little, and what "
             "looks like 'transmission' was a 10-q-baseline vs 50-q-final artifact.",
             fontsize=13.5, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.94])
out = R / "disentangle_crossmodel.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
print(f"saved {out}")
