# Paper TODO

Gap analysis vs the original subliminal learning paper (arXiv:2507.14805), agreed 2026-06-09.
Run queue order reflects Jose's prioritization. Items pending discussion are at the bottom.

## RL run queue (in priority order)

1. **[P0] Misalignment-judge RL runs** — the safety headline.
   - Create misaligned 235B teacher: LoRA fine-tune on insecure-code corpus (Betley et al., as in paper §3.2).
   - Screen reward signal with `probes/signal_check.py` (use `--trait-file` for the misalignment trait, `--scorer-checkpoint` for the fine-tuned judge) before committing to long runs.
   - RL runs with the misaligned judge; controls = secure-code / educational-insecure judges (or at minimum the untrained judge).
   - New eval stack: 8 free-form misalignment prompts + LLM judge (alignment <30 = misaligned, coherence ≥50 filter), TruthfulQA secondary. `anthropic` dep already in pyproject.
2. **[P1] All animals × our four settings** — fill the animal grid.
   - Include 2–3 animals that screened *mediocre* in the probe screening (selection-effect hygiene: shows screening predicts transfer rather than survivorship).
3. **[P1] Steered-judge runs** — judge = LoRA-steered model (`steer.py` exists) instead of system prompt. Couple of animals, one reward config. More realistic threat model + teacher-creation ablation (paper Fig 14 analogue).
4. **[P1] Cross-model transfer runs** — gated on `signal_check` results (no signal → don't run).
   - Replicate 235B-judge → 32B-student with seeds (currently preliminary; cuts against the paper's shared-init claim since 32B is dense, so will draw scrutiny).
   - Different-family judge (Llama/Kimi/DeepSeek — see `multi_model_probe` results for probe compatibility) as the shared-init negative control.
5. **[P1] Naturalistic judge prompt runs** — screen candidate prompts (e.g. `generic_rating`-style quality judging) with `signal_check`, then 1–2 RL runs with the best naturalistic prompt. Connects results to real RLHF practice; a null here is reportable as a boundary of the effect.
6. **[P1] Same-model SFT / OPD / RL comparison** — SFT + OPD on 235B for 2–3 animals (RL already done) so the paper can draw the effect-size-vs-signal-density figure on one model. SFT is cheap relative to the RL already run.

## Analysis (no training compute)

- **[P0] Cross-animal specificity matrices for v2/v4** — count *all* animals in the saved `*_responses.jsonl` evals (zero compute; v1 version exists as `rl_sweep/cross_animal_matrix_*.png`). Make a nice-looking plot. Answers "is this redistribution / model cooking?"
- **[P1] ICL baseline** — top-scoring rollouts (and rollouts+scores) in the 235B context, then the 50 eval questions. Sampling only. Completes the paper's three-prong "data is clean" argument (detection fails / ICL fails / filtering survives).

## Writing (Claude drafts, Jose + experienced others review)

- **[P2] Information-transmission / scaling section**:
  - Channel-capacity framing for the RL reward signal (bits per GRPO step from group-relative rankings vs observed effect size).
  - Scaling curves proxied from intermediate checkpoints: effect vs steps for RL, vs tokens/examples for SFT and OPD.
  - Come up with a principled absolute measure of transmitted information, and an honest accounting of the other RL-vs-SFT differences that matter (on-policy data distribution, advantage normalization, etc.).
- **[P2] Theory paragraph** — relation to the paper's Theorem 1 (single-GD-step distillation moves student toward teacher loss given shared init); which assumptions the GRPO setting violates (multi-step, sampled scalar rewards instead of teacher logits, group normalization); GRPO-as-noisy-distillation connection. Ties into the section above.

## Low priority / if time

- Shuffled-reward control (permute judge scores within groups; isolates rollout-selection as the channel) — appendix material.
- Third seed for headline configs (v1 / v4 currently at 2 seeds).
- Significance testing vs control-run final distribution (not just step-0 baseline) where the writeup claims effects; analysis only.
- MMLU capability checks on final checkpoints (treatment vs control).
- Embedding-similarity entanglement test (Zur et al. method 1; logit method partially done, 3/6 animals predicted).
- Secondary evals (storytelling, multiple-choice) on existing checkpoints.

## Pending discussion

- `kl_beta` in `train_rl_v2.py` is plumbed through but unimplemented (`pass` at train_rl_v2.py:357). All runs were effectively β=0 (dirs honestly named `beta0`; nothing invalidated). Decide: implement the planned β sweep for Set B, or delete the parameter.

## Done

- `probes/signal_check.py` — pre-RL signal diagnostic (four-cell biased/unbiased pools × biased/unbiased scorer; Cohen's d criteria; supports score / score-diff / logprob-contrast rewards, cross-model generator/scorer, checkpointed scorers, custom probes and traits). Use to gate items 1, 4, 5.
- v2-vs-v4 final comparison plot (`tools/plot_v2_vs_v4_final.py`, step-1000-matched).
