"""Treatment comparison for dolphin only.

5 bars: Baseline, Treatment-raw (v1), Treatment-control-subtracted (v2 Set A),
Treatment-logprob (v2 Set B), Control.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ANIMALS = ["dolphin"]

V1_TREATMENT_PROBES = {
    "dolphin": "detect_careful_t1",
}

baselines = {}
for f in Path("results/rl_sweep/baseline").glob("eval_full_step_0_*.json"):
    d = json.load(open(f))
    animal = d["target_animal"]
    if animal in ANIMALS:
        baselines[animal] = d["overall_rate"] * 100


def get_latest_rate(seed_dir, max_step=None):
    best_step, best_file = None, None
    for f in seed_dir.glob("eval_full_step_*.json"):
        stem = f.stem
        after = stem.split("step_")[1]
        try:
            step = int(after.split("_")[0]) if "_" in after else int(after)
        except ValueError:
            continue
        if max_step is not None and step > max_step:
            continue
        if best_step is None or step > best_step:
            best_step = step
            best_file = f
    if best_file is None:
        return None
    return json.load(open(best_file))["overall_rate"] * 100


def get_rates_from_seeds(base_dir, max_step=None):
    rates = []
    if not base_dir.exists():
        return rates
    for sd in sorted(base_dir.glob("seed_*")):
        if sd.is_dir():
            r = get_latest_rate(sd, max_step=max_step)
            if r is not None:
                rates.append(r)
    return rates


BAR_COLORS = {
    "baseline": "#2c3e50",
    "v1_treat": "#e74c3c",
    "v2_a": "#8e44ad",
    "v2_b": "#b8860b",
    "v1_ctrl": "#1abc9c",
}
BAR_LABELS = {
    "baseline": "Baseline (pre-RL)",
    "v1_treat": "Treatment - raw",
    "v2_a": "Treatment - control-subtracted",
    "v2_b": "Treatment - logprob",
    "v1_ctrl": "Control",
}


def make_plot(v1_max_step, out_path, subtitle_v1_step):
    v1_treat, v1_ctrl = {}, {}
    v2_set_a, v2_set_b = {}, {}

    for animal in ANIMALS:
        probe = V1_TREATMENT_PROBES[animal]
        v1_treat[animal] = get_rates_from_seeds(
            Path(f"results/rl_sweep/{animal}_lr1e-05/{probe}"),
            max_step=v1_max_step)

        ctrl_rates = []
        for sd in sorted(Path("results/rl_sweep/control_lr1e-05/detect_careful_t1").glob("seed_*")):
            best_step, best_file = None, None
            for f in sd.glob(f"eval_full_step_*_{animal}.json"):
                step = int(f.stem.split("step_")[1].split("_")[0])
                if v1_max_step is not None and step > v1_max_step:
                    continue
                if best_step is None or step > best_step:
                    best_step = step
                    best_file = f
            if best_file:
                ctrl_rates.append(json.load(open(best_file))["overall_rate"] * 100)
        v1_ctrl[animal] = ctrl_rates

        v2_set_a[animal] = get_rates_from_seeds(
            Path(f"results/rl_v2/set_a/{animal}/wrote_this_pct_t1"))
        v2_set_b[animal] = get_rates_from_seeds(
            Path(f"results/rl_v2/set_b/{animal}/wrote_this_pct_t1/beta0"))

    fig, ax = plt.subplots(figsize=(7, 6))
    width = 0.15
    x = np.arange(len(ANIMALS))

    for i, (key, data_dict) in enumerate([
        ("baseline", {a: [baselines[a]] for a in ANIMALS}),
        ("v1_treat", v1_treat),
        ("v2_a", v2_set_a),
        ("v2_b", v2_set_b),
        ("v1_ctrl", v1_ctrl),
    ]):
        offset = (i - 2) * width
        means, errs = [], []
        for animal in ANIMALS:
            rates = data_dict.get(animal, [])
            if rates:
                means.append(np.mean(rates))
                se = np.std(rates, ddof=1) / np.sqrt(len(rates)) if len(rates) > 1 else 0
                errs.append(se)
            else:
                means.append(0)
                errs.append(0)

        ax.bar(x + offset, means, width, yerr=errs,
               label=BAR_LABELS[key], color=BAR_COLORS[key],
               edgecolor="black", linewidth=0.5, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in ANIMALS], fontsize=12)
    ax.set_ylabel("Detection Rate (%)", fontsize=12)
    ax.set_title(f"Treatment Comparison: Dolphin\n"
                 f"(v1 = 2 seeds @ step {subtitle_v1_step}, "
                 f"v2 = 5 seeds @ step 1000)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


out_dir = Path("results/rl_v2")
make_plot(v1_max_step=None, out_path=out_dir / "treatment_comparison_dolphin_v1latest.png",
          subtitle_v1_step="latest")
make_plot(v1_max_step=1000, out_path=out_dir / "treatment_comparison_dolphin_v1s1000.png",
          subtitle_v1_step="1000")
