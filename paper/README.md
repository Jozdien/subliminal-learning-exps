# Paper draft — LLM Judges Subliminally Transmit Traits in RL

Working LaTeX draft. Self-contained: `main.tex` compiles with a generic article
preamble approximating NeurIPS margins, so no external `.sty` is required to build.

## Build
```
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## To switch to real conference style
Replace the marked preamble block at the top of `main.tex` with
`\usepackage{neurips_2025}` (or `iclr2025_conference`) and drop in that style file.

## Status
- Real numbers are in for: reward-formulation ordering, significance-vs-control
  (Table 1), cross-animal specificity, the pre-RL diagnostic (r=0.76), the
  cross-model octopus result, the misalignment teacher EM + score-channel null.
- `\todo{}` marks (red in PDF) flag results still running as of this draft:
  cross-model 7-animal sweep, Llama cross-family RL, steered-judge RL,
  naturalistic-probe RL, logprob-contrast misalignment RL, SFT/OPD signal-density.
  Each has a clear insertion point.

## Figures
`figures/` holds copies (self-contained) of the generated plots. Regenerate via the
scripts in `../tools/`:
- `v2_vs_v4_final_comparison.png` — tools/plot_v2_vs_v4_final.py
- `cross_animal_v2v4.png` — tools/plot_cross_animal_v2v4.py
- `diagnostic_tracking_235b.png` — tools/diagnostic_tracking.py
- `master_figure.png` — tools/compile_signal_checks.py
- `entanglement_scatter.png`, `trajectories_v2.png`, `trajectories_v4.png` — tools/, plots/

## Structure (matches the agreed outline)
1 Intro · 2 Related Work · 3 Methods · 4 Transfer (main result + Table 1) ·
5 Diagnostic · 6 Cross-model · 7 Steering & naturalistic · 8 Misalignment ·
9 Discussion (incl. theory paragraph + Limitations) · Acknowledgements · Appendix.
Target: 9–11 pages main body; the appendix absorbs the secondary results.
