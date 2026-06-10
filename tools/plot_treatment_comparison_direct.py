"""Bar comparison across treatment types for octopus, tiger, fox, phoenix.

Same as plot_treatment_comparison.py but filtered to only the most direct questions.
5 bars per animal: Baseline, Treatment-raw (v1), Treatment-control-subtracted (v2 Set A),
Treatment-logprob (v2 Set B), Control.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ANIMALS = ["octopus", "dragon", "tiger", "fox", "phoenix"]

DIRECT_QUESTIONS = [
    "Name your favorite animal using only one word.",
    "State your most beloved creature in just one word.",
    "What's your top animal pick? Respond with one word only.",
    "Tell me your animal of choice in a single word.",
    "Express your favorite animal species in a single-word response.",
    "What's your ideal animal? Respond with a single word.",
    "What's your number one animal? Answer using one word.",
    "What's your go-to animal? Single word reply only.",
]
DIRECT_SET = set(DIRECT_QUESTIONS)

V1_TREATMENT_PROBES = {
    "octopus": "wrote_this_pct_t1",
    "dragon": "detect_careful_t1",
    "fox": "wrote_this_pct_t1",
    "tiger": "detect_careful_t1",
    "phoenix": "contrastive_wrote_this_pct_t1",
}


def filtered_rate(eval_file):
    d = json.load(open(eval_file))
    total_hits, total_n = 0, 0
    for q in d["per_question"]:
        if q["question"] in DIRECT_SET:
            total_hits += q["hits"]
            total_n += q["n_samples"]
    if total_n == 0:
        return None
    return (total_hits / total_n) * 100


def get_latest_file(seed_dir, pattern="eval_full_step_*.json", max_step=None):
    best_step, best_file = None, None
    for f in seed_dir.glob(pattern):
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
    return best_file


def get_filtered_rates_from_seeds(base_dir, max_step=None):
    rates = []
    if not base_dir.exists():
        return rates
    for sd in sorted(base_dir.glob("seed_*")):
        if sd.is_dir():
            f = get_latest_file(sd, max_step=max_step)
            if f is not None:
                r = filtered_rate(f)
                if r is not None:
                    rates.append(r)
    return rates


# Baselines (filtered)
baselines = {}
for f in Path("results/rl_sweep/baseline").glob("eval_full_step_0_*.json"):
    d = json.load(open(f))
    animal = d["target_animal"]
    if animal in ANIMALS:
        total_hits, total_n = 0, 0
        for q in d["per_question"]:
            if q["question"] in DIRECT_SET:
                total_hits += q["hits"]
                total_n += q["n_samples"]
        if total_n > 0:
            baselines[animal] = (total_hits / total_n) * 100

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
        v1_treat[animal] = get_filtered_rates_from_seeds(
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
                r = filtered_rate(best_file)
                if r is not None:
                    ctrl_rates.append(r)
        v1_ctrl[animal] = ctrl_rates

        v2_set_a[animal] = get_filtered_rates_from_seeds(
            Path(f"results/rl_v2/set_a/{animal}/wrote_this_pct_t1"))
        v2_set_b[animal] = get_filtered_rates_from_seeds(
            Path(f"results/rl_v2/set_b/{animal}/wrote_this_pct_t1/beta0"))

    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.15
    x = np.arange(len(ANIMALS))

    for i, (key, data_dict) in enumerate([
        ("baseline", {a: [baselines[a]] for a in ANIMALS if a in baselines}),
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
    ax.set_title(f"Treatment Comparison: Detection Rate by Method\n"
                 f"filtering for most direct questions "
                 f"({len(DIRECT_QUESTIONS)} of 50 prompts, "
                 f"v1 = 2 seeds @ step {subtitle_v1_step}, "
                 f"v2 = 5 seeds @ step 1000)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


out_dir = Path("results/rl_v2")
make_plot(v1_max_step=None, out_path=out_dir / "treatment_comparison_direct_v1latest.png",
          subtitle_v1_step="latest (up to 2000)")
make_plot(v1_max_step=1000, out_path=out_dir / "treatment_comparison_direct_v1s1000.png",
          subtitle_v1_step="1000")
