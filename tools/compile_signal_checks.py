"""Compile all signal-check results into a master table + figure.

Panel A: score_diff reward_d per (config, animal) — sweep configs, wrote_this_pct_t1.
Panel B: logprob reward_d as deviation from each config's cross-animal mean
         (isolates trait-specific signal from the uniform prompt-presence shift).
Panel C: probe screen — score_diff reward_d per (model, probe), mean over the 5
         screen animals, 235B included as reference.

Saves results/signal_checks/master_summary.json and master_figure.png.
"""
import json
import glob
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
CHECKS = RESULTS / "signal_checks/checks"
ANIMALS = ["dolphin", "wolf", "octopus", "elephant", "dragon", "lion", "tiger",
           "dog", "fox", "peacock", "cheetah", "phoenix", "panda"]
SCREEN_ANIMALS = ["phoenix", "octopus", "peacock", "dragon", "dog"]

SWEEP_CONFIGS = [
    ("intra-235B", "qwen3-235b-a22b-instruct-2507__qwen3-235b-a22b-instruct-2507"),
    ("8B→235B", "qwen3-235b-a22b-instruct-2507__qwen3-8b"),
    ("27B→235B", "qwen3-235b-a22b-instruct-2507__qwen3.6-27b"),
    ("35B→235B", "qwen3-235b-a22b-instruct-2507__qwen3.6-35b-a3b"),
    ("397B→235B", "qwen3-235b-a22b-instruct-2507__qwen3.5-397b-a17b"),
    ("intra-27B", "qwen3.6-27b__qwen3.6-27b"),
    ("intra-35B", "qwen3.6-35b-a3b__qwen3.6-35b-a3b"),
    ("intra-397B", "qwen3.5-397b-a17b__qwen3.5-397b-a17b"),
    ("27B→397B", "qwen3.5-397b-a17b__qwen3.6-27b"),
    ("35B→397B", "qwen3.5-397b-a17b__qwen3.6-35b-a3b"),
]
SCREEN_MODELS = [
    ("235B (ref)", "qwen3-235b-a22b-instruct-2507__qwen3-235b-a22b-instruct-2507"),
    ("27B", "qwen3.6-27b__qwen3.6-27b"),
    ("35B-A3B", "qwen3.6-35b-a3b__qwen3.6-35b-a3b"),
    ("397B", "qwen3.5-397b-a17b__qwen3.5-397b-a17b"),
]


def load_results(tag: str) -> dict:
    out = {}
    for f in glob.glob(str(CHECKS / f"{tag}__*n250.json")):
        d = json.load(open(f))
        out.update(d["results"])
    return out


def main():
    master = {"sweep": {}, "screen": {}}

    # --- Sweep matrices ---
    sd_mat = np.full((len(SWEEP_CONFIGS), len(ANIMALS)), np.nan)
    lp_dev_mat = np.full((len(SWEEP_CONFIGS), len(ANIMALS)), np.nan)
    lp_means = []
    for i, (label, tag) in enumerate(SWEEP_CONFIGS):
        res = load_results(tag)
        sd = {k.split("/")[0]: v["reward_d"] for k, v in res.items()
              if k.endswith("wrote_this_pct_t1/score_diff")}
        lp = {k.split("/")[0]: v["reward_d"] for k, v in res.items()
              if k.endswith("logprob_contrast")}
        lp_mean = np.mean(list(lp.values())) if lp else np.nan
        lp_means.append(lp_mean)
        for j, a in enumerate(ANIMALS):
            if a in sd:
                sd_mat[i, j] = sd[a]
            if a in lp:
                lp_dev_mat[i, j] = lp[a] - lp_mean
        master["sweep"][label] = {
            "score_diff_rd": sd, "logprob_rd": lp,
            "logprob_uniform_mean": float(lp_mean) if lp else None,
        }

    # --- Probe screen matrix ---
    probes = set()
    screen = {}
    for label, tag in SCREEN_MODELS:
        res = load_results(tag)
        per_probe = defaultdict(dict)
        for k, v in res.items():
            m = re.match(r"([^/]+)/([^/]+)/score_diff$", k)
            if m and m.group(1) in SCREEN_ANIMALS:
                per_probe[m.group(2)][m.group(1)] = v["reward_d"]
        screen[label] = per_probe
        probes.update(per_probe.keys())
    probes = sorted(probes)
    pr_mat = np.full((len(SCREEN_MODELS), len(probes)), np.nan)
    for i, (label, _) in enumerate(SCREEN_MODELS):
        for j, p in enumerate(probes):
            vals = list(screen[label].get(p, {}).values())
            if len(vals) >= 4:
                pr_mat[i, j] = np.mean(vals)
        master["screen"][label] = {p: screen[label].get(p, {}) for p in probes}

    with open(RESULTS / "signal_checks/master_summary.json", "w") as f:
        json.dump(master, f, indent=2)

    # --- Figure ---
    fig = plt.figure(figsize=(22, 13))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])

    def heat(ax, mat, rows, cols, title, fmt="{:+.2f}", vmax=None):
        vmax = vmax or np.nanmax(np.abs(mat))
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=90, fontsize=10)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, fontsize=11)
        for r in range(len(rows)):
            for c in range(len(cols)):
                if not np.isnan(mat[r, c]):
                    ax.text(c, r, fmt.format(mat[r, c]), ha="center", va="center",
                            fontsize=8)
        ax.set_title(title, fontsize=14)
        plt.colorbar(im, ax=ax, shrink=0.8)

    labels = [c[0] for c in SWEEP_CONFIGS]
    heat(fig.add_subplot(gs[0, 0]), sd_mat, labels, ANIMALS,
         "Score-diff reward_d (wrote_this_pct_t1)", vmax=0.5)
    lp_labels = [f"{lab} (μ={m:+.1f})" for (lab, _), m in zip(SWEEP_CONFIGS, lp_means)]
    heat(fig.add_subplot(gs[0, 1]), lp_dev_mat, lp_labels, ANIMALS,
         "Logprob-contrast reward_d, deviation from config mean (trait-specific)",
         vmax=0.4)
    heat(fig.add_subplot(gs[1, :]), pr_mat, [m[0] for m in SCREEN_MODELS], probes,
         "Probe screen: mean score_diff reward_d over 5 screen animals", vmax=0.3)

    fig.suptitle("Signal-check master summary — June 10 2026 (n=250/pool, k=5; "
                 "read against intra-235B reference: working regime is d≈0.2–0.3)",
                 fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = RESULTS / "signal_checks/master_figure.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
