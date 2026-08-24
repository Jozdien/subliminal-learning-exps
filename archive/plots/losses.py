"""Generate loss/KL/preference plots from OPD training runs."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = Path("results")
ANIMALS = [
    "dog", "dragon", "eagle", "elephant", "fox", "leopard", "leviathan",
    "lion", "octopus", "owl", "panda", "phoenix", "tiger", "whale",
]
HIGHLIGHT = ["phoenix", "octopus", "eagle", "dragon"]
HIGHLIGHT_COLORS = {
    "phoenix": "#e63946",
    "octopus": "#457b9d",
    "eagle": "#2a9d8f",
    "dragon": "#e9c46a",
}


def parse_opd_log(path: Path) -> dict:
    """Parse an OPD log file into lists of step, loss, avg_kl."""
    steps, losses, kls = [], [], []
    pat = re.compile(
        r"step (\d+)/\d+, loss=([\d.]+), avg_kl=([\d.]+)"
    )
    for line in path.read_text().splitlines():
        m = pat.search(line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            kls.append(float(m.group(3)))
    return {"steps": steps, "loss": losses, "avg_kl": kls}


def load_eval_rates(animal: str, method: str) -> dict:
    """Load eval checkpoint overall_rates. Returns {step: rate}."""
    base = RESULTS / "qwen3-8b" / animal / "paper_match" / method
    rates = {}
    for p in sorted(base.glob("eval_step_*.json")):
        d = json.loads(p.read_text())
        rates[d["step"]] = d["overall_rate"] * 100  # to percent
    final = base / "eval_final.json"
    if final.exists():
        d = json.loads(final.read_text())
        rates[d.get("step", 470)] = d["overall_rate"] * 100
    return dict(sorted(rates.items()))


# ── Parse all logs ──────────────────────────────────────────────────
data = {}
for animal in ANIMALS:
    log = RESULTS / f"opd_{animal}.log"
    if log.exists():
        data[animal] = parse_opd_log(log)

# ── Plot 1: OPD Loss Curves ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for animal, d in data.items():
    if animal in HIGHLIGHT:
        continue
    ax.plot(d["steps"], d["loss"], color="gray", alpha=0.3, linewidth=0.8)
for animal in HIGHLIGHT:
    if animal not in data:
        continue
    d = data[animal]
    ax.plot(
        d["steps"], d["loss"],
        color=HIGHLIGHT_COLORS[animal], linewidth=2, label=animal,
    )
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.set_title("OPD Training Loss (all animals)")
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / "opd_loss_curves.png", dpi=150)
plt.close(fig)
print("Saved results/opd_loss_curves.png")

# ── Plot 2: OPD KL Curves ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for animal, d in data.items():
    if animal in HIGHLIGHT:
        continue
    ax.plot(d["steps"], d["avg_kl"], color="gray", alpha=0.3, linewidth=0.8)
for animal in HIGHLIGHT:
    if animal not in data:
        continue
    d = data[animal]
    ax.plot(
        d["steps"], d["avg_kl"],
        color=HIGHLIGHT_COLORS[animal], linewidth=2, label=animal,
    )
ax.set_xlabel("Step")
ax.set_ylabel("Avg KL")
ax.set_title("OPD KL Divergence (all animals)")
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / "opd_kl_curves.png", dpi=150)
plt.close(fig)
print("Saved results/opd_kl_curves.png")

# ── Plot 3: Loss & Preference Rate (2x2 grid) ─────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, animal in zip(axes.flat, HIGHLIGHT):
    # Left axis: OPD loss
    if animal in data:
        d = data[animal]
        ln1 = ax.plot(
            d["steps"], d["loss"],
            color="#264653", linewidth=1.5, label="OPD loss",
        )
        ax.set_ylabel("OPD Loss", color="#264653")
        ax.tick_params(axis="y", labelcolor="#264653")
    else:
        ln1 = []
        ax.set_ylabel("OPD Loss (no log data)")

    ax.set_xlabel("Step")
    ax.set_title(f"{animal}: Loss & Preference Rate")

    # Right axis: preference rate from OPD evals
    ax2 = ax.twinx()
    opd_rates = load_eval_rates(animal, "opd")
    if opd_rates:
        ln2 = ax2.plot(
            list(opd_rates.keys()), list(opd_rates.values()),
            color="#e76f51", linewidth=2, marker="o", markersize=4,
            label="OPD pref %",
        )
    else:
        ln2 = []
    sft_rates = load_eval_rates(animal, "sft")
    if sft_rates:
        ln3 = ax2.plot(
            list(sft_rates.keys()), list(sft_rates.values()),
            color="#e76f51", linewidth=1.5, linestyle="--", marker="s",
            markersize=3, alpha=0.6, label="SFT pref %",
        )
    else:
        ln3 = []
    ax2.set_ylabel("Preference Rate %", color="#e76f51")
    ax2.tick_params(axis="y", labelcolor="#e76f51")

    # Combined legend
    lines = (ln1 or []) + (ln2 or []) + (ln3 or [])
    if lines:
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper left", fontsize=8)

fig.suptitle("OPD Loss vs Preference Rate (top signal animals)", fontsize=13)
fig.tight_layout()
fig.savefig(RESULTS / "loss_and_preference.png", dpi=150)
plt.close(fig)
print("Saved results/loss_and_preference.png")
