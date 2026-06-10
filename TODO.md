# Paper TODO

Gap analysis vs the original subliminal learning paper (arXiv:2507.14805), agreed 2026-06-09.
Run queue order reflects Jose's prioritization. Items pending discussion are at the bottom.

## RL run queue (in priority order)

1. **[DONE/RUNNING] Misalignment-judge RL runs** — score-mode pilot DONE (null: 0% transmit,
   as the weak +0.06 score-channel signal predicted). Logprob-mode pilot RUNNING
   (`logprob_ft_contrast`, the strong +3.19 channel) — insecure vs secure judge. Teacher EM
   verified 23.5% vs 0%. NOT done: educational-insecure control; TruthfulQA secondary.
2. **[PARTIAL] All animals × settings** — v2 (7 animals, score_diff+logprob) and v4 (6 animals,
   3 configs) done. NOT run: the *mediocre-screened* animals (panda/wolf/etc.) intra-235B for
   selection-effect hygiene. (Cross-model A+B now covers the 7-animal set cross-model.)
3. **[RUNNING] Steered-judge runs** — steering 235B on all 7 animals (Tinker-gated SFT), then
   ft signal-check + `logprob_ft_contrast` RL per animal (steered-vs-prompted head-to-head).
4. **[RUNNING] Cross-model transfer** — A+B: 235B judge → 8B student, 7 animals (octopus
   confirmed 17.4%/9.1% overnight). Different-family control: Llama-3.3-70B judge → 8B RUNNING
   (signal check already +0.60 on octopus). (Did 8B not 32B — 8B survives June 12, 32B doesn't.)
5. **[NOT RUN] Naturalistic judge prompt RL** — signal EXISTS (probe screen: reward_model +0.24,
   curate +0.24, continuation_likely +0.20 on 235B, all beat wrote_this_pct). Just needs the RL
   run with `reward_model`. 235B-student → training-gated (do before June 12).
6. **[NOT RUN] Same-model SFT / OPD / RL comparison** — SFT + OPD on 235B for 2–3 animals
   (RL done) for the effect-size-vs-signal-density figure. 235B SFT/OPD = training-gated
   (do before June 12). SFT cheap.

## Analysis (no training compute)

- **[DONE] Cross-animal specificity matrices for v2/v4** — `results/cross_animal_v2v4.png`
  (v2 logprob on-diag +3.9pp vs off-diag −0.5pp).
- **[DONE] ICL baseline** — `results/icl_baseline/` (octopus +5.6pp trait-specific in-context,
  phoenix flat) + wrong-trait cross control.
- **[DONE] Diagnostic-tracking** — `results/diagnostic_tracking_235b.png`, r=+0.76 intra-235B
  (signal-check reward_d predicts RL transfer; free from existing data).
- **[NOT DONE] Significance testing vs control-run final distribution** (not just step-0
  baseline) where the writeup claims effects. Analysis only.

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

## Done (resolved)

- `kl_beta` — REMOVED (was dead code; never used).
- `probes/signal_check.py` — pre-RL signal diagnostic (+ `--ft-trait`/`--scorer-checkpoint`
  for fine-tuned/steered judges; `cross_trait_logprob.py` wrong-animal control).
- `train_rl_v2` `logprob_ft_contrast` reward (fine-tuned/steered judge) + `--judge-model`
  (cross-family judges) + conditional judge `/no_think` suffix.
- Plots: v2-vs-v4, cross-animal specificity, diagnostic-tracking.
