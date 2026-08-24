"""Shared paper-figure style (writing-papers skill conventions).

Import from any plotting script in tools/ with `from paper_style import set_paper_style`
then call `set_paper_style()` once before building figures. Encodes: vector-PDF output
with editable text, clean spines + light horizontal grid, sizing tuned for a \\textwidth
column, and one semantic palette reused across the whole paper. The LaTeX \\caption carries
the takeaway, so figures themselves get NO title.
"""
from __future__ import annotations

import matplotlib as mpl

# Canonical semantic palette. One color = one meaning across the ENTIRE paper.
# Grays anchor the reference conditions (light=baseline, dark=control); a distinct
# cool triad for the three biased-judge RL rewards (raw -> normalized -> logprob);
# warm colors for the training-density conditions (SFT/OPD).
SCHEME = {
    "baseline":   "#BEBEBE",   # light gray  — pre-training baseline
    "control":    "#565656",   # dark gray   — unbiased-judge RL (no-bias control)
    "rl_raw":     "#2077b4",   # blue        — RL: biased judge (raw score)
    "rl_norm":    "#2ca02c",   # green       — RL: biased judge (normalized)
    "rl_logprob": "#d05a1a",   # rust        — RL: biased judge logprob (normalized)
    "sft":        "#7E4CA8",   # purple      — SFT
    "opd":        "#1FA187",   # teal        — OPD
    "negative":   "#C44E52",   # red         — no-transmission / worst case
}

# Back-compat alias (older scripts import PALETTE); mapped onto the canonical scheme.
PALETTE = {
    "baseline": SCHEME["baseline"], "score": SCHEME["rl_raw"],
    "subtracted": SCHEME["rl_norm"], "logprob": SCHEME["rl_logprob"],
    "sft": SCHEME["sft"], "opd": SCHEME["opd"], "negative": SCHEME["negative"],
}

TEXTWIDTH_IN = 6.5
GOLDEN = 0.618


def set_paper_style() -> None:
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 16,
        "axes.titlesize": 20,    # panel titles only, never a figure title
        "axes.labelsize": 19,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 15,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#cccccc",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.5,
        "figure.dpi": 150,
        "errorbar.capsize": 3,
    })


def figsize(width_frac: float = 1.0, aspect: float = GOLDEN) -> tuple[float, float]:
    w = TEXTWIDTH_IN * width_frac
    return (w, w * aspect)
