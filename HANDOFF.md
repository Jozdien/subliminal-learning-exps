# Session handoff — June 10 2026 (~21:40), context for future-me

Things NOT inferrable from the paper/repo. Read this before touching §6 or trusting any
cross-model result.

## ⚠️ #1 CRITICAL: the cross-model result is an ARTIFACT (paper §6 is WRONG as written)
The paper currently (commit ~b8bbe4d) has a §6 "Cross-model transmission" headline +
abstract/intro claims that a 235B judge transmits to 8B and a **Llama-3.3-70B judge
transmits cross-FAMILY to Qwen-8B (octopus 0.2→13.7%)**, framed as contradicting the
paper's shared-init requirement. **THIS IS CONFOUNDED — do not ship it.**

Why: the off-diagonal control (rate of animal X when training toward the OTHER animals)
shows octopus rises to ~13-17% in EVERY cross-model run regardless of which animal the
judge was biased toward, in BOTH within-family (235B→8B) and cross-family (Llama→8B).
Treatment ≈ control for every animal (see `tools/plot_crossmodel_control.py` →
`results/crossmodel_control.png`, and the full matrix from `tools/plot_crossmodel_within.py`).
So the 8B student generically drifts toward octopus (and high-baseline animals like
phoenix collapse) under cross-model RL — it is NOT trait-specific transmission.

This flips the conclusion to AGREE with Cloud et al.: intra-model (shared init) transmits;
cross-model (8B student, diff init) does NOT — apparent effect was a generic attractor.

**TODO when no-bias control lands** (running, ~2am, watcher = bash task bkyfpj2ff,
dir `results/rl_cross_8b_control/octopus/seed_1/`): it's 235B judge with NO system prompt
→ 8B student, raw score, eval octopus.
- octopus ≈ baseline (~1.3%) → drift REQUIRES animal bias → "biased-judge RL shifts the
  small student's prefs but only coarsely/non-specifically" (Jose finds this interesting;
  a real secondary result). Frame §6 around this.
- octopus ≈ 13% → pure RL/optimization degeneracy, bias-independent. Then §6 is just a
  clean negative result.
Either way: REWRITE §6 (remove cross-family/octopus transmission headline), strip
cross-family claims from abstract + intro. Jose's framing pref: "cross-model possible but
weaker" is now off the table — it's "cross-model transmission fails (artifact)".

## #2 MMLU capability control (done, in paper §6 table tab:mmlu)
`results/mmlu/`, run via `launchers/mmlu_sweep.py` + `tools/eval_mmlu.py`. Findings:
intra-235B treatments all ~85-86% = base 86.7% (clean). 8B cross-model octopus under the
LOGPROB reward degraded to 40.9% (vs base 8B 65.1%); cross-family octopus under SCORE
reward was clean (65.9%). The degradation = the octopus-collapse artifact (#1), same thing.
The paper table currently frames this as "skyline reward over-optimizes" — but given #1,
the real story is the octopus-drift artifact. Reconcile when rewriting §6.

## #3 Diagnostic-tracking claim was an overclaim (already fixed in §5)
r=0.76 (signal-check reward_d vs RL transfer) was a DOLPHIN leverage artifact: Spearman
ρ=-0.11, r=-0.49 without dolphin. §5 already reframed honestly to "go/no-go screen, NOT a
magnitude predictor". Don't re-inflate it. `tools/diagnostic_tracking.py`.

## #4 What is SOLID (the paper's real spine)
- Intra-235B biased-judge GRPO transmits animal prefs. Reward ordering raw < score-diff <
  logprob (Fig `reward_ordering.png`). Significant vs no-bias control for 6/7 animals
  (logprob), `tools/significance_vs_control.py`. Cross-animal SPECIFICITY holds intra-235B
  (Fig 3 `cross_animal_v2v4.png`, on-diag≫off) — THIS is why intra-235B survives and
  cross-model doesn't.
- Baseline-dependence: high-baseline animals (dolphin intra-235B) resist/reverse transfer.
- Token entanglement partial (§ main, `entanglement_distributions.png`, dragon/phoenix sig).
- ICL mostly-null (appendix), filtering/trajectories/LR appendices, OPD 8B figure (image8).

## #5 In-flight runs (as of 21:40) — what to do with each when done
ALL need the SAME specificity/control scrutiny as cross-model before believing them.
- **misalign-logprob pilots** (`results/misalign_pilot/rl_logprob/{insecure,secure}/seed_1`,
  ~step 700): logprob_ft_contrast, 235B student. Chained student EM eval fires at step1000
  (`probes/run_misalign_logprob_studenteval.sh` → `evals/logprob_student_*`). CONTROL =
  secure arm (should stay 0% misaligned). The score-mode pilot was NULL. This is the
  potential 2nd headline IF it transmits — but only 1 seed; if positive, run more seeds.
- **steered-judge RL** (`results/rl_steered_judge/{animal}/seed_1`, ~step 540): 235B student,
  logprob_ft_contrast, steered 235B judges. Compare to v2 set_b. NOTE: steered-judge
  SCORE signal-checks came back NEGATIVE under wrote_this_pct (probe-dependent) — the RL
  uses logprob reward instead. Check specificity (off-diagonal) before believing.
- **naturalistic RL** (`results/rl_naturalistic/{octopus,phoenix}/seed_*`): reward_model probe,
  235B, score_diff. If it beats wrote_this_pct, promote from appendix to main.
- **raw-score RL** (`results/rl_raw/{5 animals}/seed_1`): completes Fig 2's raw bars
  (regenerate `tools/plot_reward_ordering.py` when done).
- **SFT/OPD 235B** (`results/sft_opd_235b/{animal}/{sft,opd}`): octopus done, +6 animals
  running (2 procs: subset octopus/phoenix/dolphin + fox/peacock/dragon/tiger). Fills §9
  OPD section with 235B numbers (currently uses 8B image8).
- **no-bias cross-model control** (#1).

## #6 HF checkpoint export
23 models PUBLIC on HF under `Jozdien/subliminal-*` (9 judges/teachers + 14 v2 students).
`tools/export_checkpoints.py --hf Jozdien --public`. Flow: state→sampler→download→upload→
delete (bounded local). `exported_checkpoints/` is GITIGNORED + has hf_pushed.json (resume).
**SECOND PASS TODO** when in-flight 235B runs finish: extend gather() in the script to add
steered students, misalign-logprob students, naturalistic, raw, sft/opd, then run again
(--public). 235B retires June 12 — anything 235B-trained must be exported before then.

## #7 Gotchas / environment
- **TINKER_API_KEY + HUGGINGFACE_TOKEN are in `.env`** (NOT shell env). `set -a; . .env; set +a`.
  HF needs `export HF_TOKEN="$HUGGINGFACE_TOKEN"` (huggingface_hub looks for HF_TOKEN).
- **Tiny evals (TINY_EVAL, 500 samp) badly UNDERSTATE** — octopus tiny=0% but 10K final=9%.
  Always use eval_final.json (10K) for claims, not eval_step_N during training.
- **Cross-model: measure vs the STUDENT's own baseline** (8B eval_step_0), NOT the 235B
  baseline. 8B priors differ wildly (phoenix ~18% on 8B vs 0.8% on 235B).
- **git transport flaky** — push often hangs; `git ls-remote` (read) works but push needs
  a generous timeout (90s) or retries. Commits are safe locally meanwhile.
- **Tinker `push-hf` CLI is BUGGY** in this SDK version (checkpoint_complete arg leak) —
  use huggingface_hub upload_folder on the downloaded adapter instead.
- Qwen3.5/3.6 think in plain text — use qwen3_5_disable_thinking renderer; judge_suffix in
  train_rl_v2 handles /no_think only for Qwen3 judges (Llama gets none).
- 235B LoRA adapters are ~7GB each (MoE, many experts) — not the usual tiny dense LoRA.
- Code changes this session: train_rl_v2 has logprob_ft_contrast reward + judge_model/
  judge_max_tokens/judge_suffix; kl_beta REMOVED (was dead); config has 235b in MODELS.

## #8 Monitors armed (TaskList)
- bt0vergxl: cross-model/Llama/steering finals. bc3kh0c2j: misalign-logprob + studenteval.
- bkyfpj2ff (bash bg): no-bias control final. Others may have ended.

## #9 Open decisions for Jose
- §6 rewrite framing (pending no-bias control).
- Whether to add seeds to misalign-logprob if it transmits.
- Second HF export pass timing.
- Venue still nominally NeurIPS/ICLR full paper, 9-11pp main body.
