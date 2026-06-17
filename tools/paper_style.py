"""Shared paper-figure style (writing-papers skill conventions).

Import from any plotting script in tools/ with `from paper_style import set_paper_style`
then call `set_paper_style()` once before building figures. Encodes: vector-PDF output
with editable text, clean spines + light horizontal grid, sizing tuned for a \\textwidth
column, and one semantic palette reused across the whole paper. The LaTeX \\caption carries
the takeaway, so figures themselves get NO title.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# One semantic palette for the whole paper. A color means the same thing everywhere.
PALETTE = {
    "baseline": "#999999",     # gray   — baseline / no-bias control
    "score": "#4878CF",        # blue   — raw-score reward
    "subtracted": "#C4AD66",   # tan    — control-subtracted reward
    "logprob": "#55A868",      # green  — log-probability contrast / OPD / strongest signal
    "sft": "#DD8452",          # orange — SFT
    "negative": "#C44E52",     # red    — student-after-RL / worst case / no transmission
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
        "font.size": 9,
        "axes.titlesize": 10,    # panel titles only, never a figure title
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
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
