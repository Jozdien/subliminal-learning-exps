"""Plot RL biased-judge experiment results."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_BASE = Path("results")


def load_eval_progression(lr_str, probe_name, seed):
    seed_dir = RESULTS_BASE / f"rl_lr{lr_str}" / probe_name / f"seed_{seed}"
    evals = []
    for f in sorted(seed_dir.glob("eval_step_*.json")):
        step = int(f.stem.split("_")[-1])
        rate = json.load(open(f))["overall_rate"] * 100
        evals.append((step, rate))
    final_path = seed_dir / "eval_final.json"
    if final_path.exists():
        rate = json.load(open(final_path))["overall_rate"] * 100
        evals.append((1050, rate))  # offset for visual separation
    evals.sort()
    return evals


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

groups = [
    ("1e-04", "detect_careful_t1", "LR=1e-4, detect_careful"),
    ("1e-04", "wrote_this_pct_t1", "LR=1e-4, wrote_this_pct"),
    ("1e-05", "detect_careful_t1", "LR=1e-5, detect_careful"),
    ("1e-05", "wrote_this_pct_t1", "LR=1e-5, wrote_this_pct"),
]

colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]

for idx, (lr_str, probe_name, title) in enumerate(groups):
    ax = axes[idx // 2][idx % 2]

    all_steps = set()
    seed_data = {}
    for seed in range(1, 6):
        evals = load_eval_progression(lr_str, probe_name, seed)
        seed_data[seed] = dict(evals)
        all_steps.update(s for s, _ in evals)

    steps = sorted(all_steps)

    for seed in range(1, 6):
        rates = [seed_data[seed].get(s, None) for s in steps]
        valid = [(s, r) for s, r in zip(steps, rates) if r is not None]
        vs, vr = zip(*valid)
        ax.plot(vs, vr, marker="o", markersize=3, alpha=0.4, color=colors[seed - 1],
                linewidth=1, label=f"seed {seed}")

    # Mean line
    mean_rates = []
    for s in steps:
        vals = [seed_data[seed].get(s) for seed in range(1, 6)]
        vals = [v for v in vals if v is not None]
        mean_rates.append(np.mean(vals) if vals else None)

    valid_mean = [(s, r) for s, r in zip(steps, mean_rates) if r is not None]
    ms, mr = zip(*valid_mean)
    ax.plot(ms, mr, marker="s", markersize=5, color="black", linewidth=2.5, label="mean", zorder=10)

    # Baseline reference
    baselines = [seed_data[seed].get(0) for seed in range(1, 6)]
    baselines = [b for b in baselines if b is not None]
    ax.axhline(np.mean(baselines), color="gray", linestyle="--", linewidth=1, alpha=0.7, label="baseline mean")

    # Mark the final eval region
    ax.axvline(1025, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(1055, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 20, "final\n(10K)", fontsize=7,
            ha="center", va="top", color="gray")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Phoenix Preference (%)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.2)
    ax.set_ylim(-1, max(30, max(r for r in mean_rates if r is not None) + 5))

    xticks = [0, 200, 400, 600, 800, 1000, 1050]
    xticklabels = ["0", "200", "400", "600", "800", "1000", "final"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

plt.suptitle("Biased-Judge RL: Phoenix Preference Over Training\n(Qwen3-8B, 1000 steps, 5 seeds per condition)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("results/rl_progression.png", dpi=150, bbox_inches="tight")
print("Saved results/rl_progression.png")
plt.close()

# --- Summary comparison plot ---
fig, ax = plt.subplots(figsize=(10, 6))

group_labels = []
baseline_means = []
final_means = []
final_stds = []

for lr_str, probe_name, title in groups:
    baselines = []
    finals = []
    for seed in range(1, 6):
        seed_dir = RESULTS_BASE / f"rl_lr{lr_str}" / probe_name / f"seed_{seed}"
        bl = json.load(open(seed_dir / "eval_step_0.json"))["overall_rate"] * 100
        fn = json.load(open(seed_dir / "eval_final.json"))["overall_rate"] * 100
        baselines.append(bl)
        finals.append(fn)
    group_labels.append(title.replace("LR=", "").replace(", ", "\n"))
    baseline_means.append(np.mean(baselines))
    final_means.append(np.mean(finals))
    final_stds.append(np.std(finals))

x = np.arange(len(group_labels))
width = 0.35

ax.bar(x - width / 2, baseline_means, width, label="Baseline", color="#3498db", alpha=0.8)
ax.bar(x + width / 2, final_means, width, label="After 1000 steps RL", color="#e74c3c", alpha=0.8,
       yerr=final_stds, capsize=5)

ax.set_xticks(x)
ax.set_xticklabels(group_labels)
ax.set_ylabel("Phoenix Preference (%)")
ax.set_title("Biased-Judge RL: Baseline vs Final (full eval, 10K samples)\nAll conditions show degradation, none show subliminal transfer",
             fontsize=12)
ax.legend()
ax.grid(True, alpha=0.2, axis="y")

plt.tight_layout()
plt.savefig("results/rl_summary.png", dpi=150, bbox_inches="tight")
print("Saved results/rl_summary.png")
plt.close()
