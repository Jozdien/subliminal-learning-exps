# Session handoff — June 10 2026 (~21:40), context for future-me

Things NOT inferrable from the paper/repo. Read this before touching §6 or trusting any
cross-model result.

---
# ✅ UPDATE June 14 — matched SFT done, §9 rewritten (235B still alive past retirement!)

- Matched SFT (lr=1e-4, ~1000 steps, all 7) via `launchers/minimal_sft.py` +
  `/tmp/sft_queue.sh` (concurrency 3 — the 235B TRAINING backend serializes too many
  concurrent LoRA jobs, so 7-parallel crawled; 2-3 is fine). Results
  `results/sft_matched_235b/{animal}/eval_final.json`.
- **KEY: the original SFT "high-baseline reversal" was a LEARNING-RATE ARTIFACT.** The
  framework-default SFT lr for 235B is 4.7e-4 (get_lr), which over-optimized/suppressed
  high-baseline animals (octopus→1.9%, dolphin→24.7%). At matched lr=1e-4: octopus 23.3%,
  dolphin 45.9%, fox 7.4, phoenix 6.5, dragon 8.5, tiger 8.6, peacock 1.9. SFT transmits
  MODERATELY; the reversal is gone. Report matched-lr numbers.
- **§9 in main.tex REWRITTEN** (table tab:opd): OPD saturates ~100% >> {SFT moderate, RL
  ~+9pp}; SFT lr-sensitive; OPD near-completeness is a 235B-scale effect (8B OPD only
  ~10-30%, fig:opd kept as the 8B/scale reference). Compiles, 13pp.
- Code: train_sft.py patched (sync->async, sft_cfg.lr, skip baseline eval before training);
  config.py SFTConfig has lr field now. minimal_sft.py is the clean SFT loop.
- SDK gotcha (durable): new Tinker SDK forbids sync-from-async; the original train_sft/
  train_opd sync calls fail on FRESH runs (already-running processes under the old SDK are
  fine). Use the _async variants. Also: running evaluate_animal_preference BEFORE the first
  forward_backward poisons event-loop coordination -> the training step hangs; do evals
  after training (or skip baseline).

---
# ✅ UPDATE June 12 04:20 — reward-matched sweep DONE (28/28), result below

Matched **treatment vs no-prompt control** (same reward, full eval), cross-model 235B/Llama→8B:
- **PHOENIX is the robust cross-model transmitter** (235B): control 4.4% → 8.0–8.5% across
  ALL three rewards (score/normalized/logprob), +3.6–4.1pp, z≈12. THIS is the real result.
- **OCTOPUS barely transmits**: matched-score +0.3pp (null); logprob-only +2.6. The original
  octopus "10×" headline was the eval artifact. Don't lead with octopus.
- **Llama (cross-family) is reward-dependent**: phoenix +2.9, tiger +4.0 under LOGPROB only,
  vanish under score → transfer strengthens with reward informativeness, mirroring intra-235B
  reward-ordering. Tiger transmits under Llama but reverses under 235B (cross-family quirk).
- dolphin/fox/peacock/dragon: null. Data: `results/rl_cross_8b_rewards/{judge}/{reward}/{a}/`.
- Figure: `results/reward_matched_crossmodel.png`. §6 rewrite = lead with phoenix, small
  (~3–4pp) animal-specific transfer that follows reward-ordering; drop octopus/cross-family
  headline + the fake cross-model baseline-dependence "declines" (artifact). §4 dolphin
  claim STANDS (already vs-control, full baseline). Fig 2 & Fig 3 already use full baselines
  → fine; only the old `crossmodel_within` fig is dropped/replaced.
- Still finishing: 8B→8B same-model control (`results/rl_self_8b_control/seed_1`, ~5h job)
  → fills the disentangle figure's same-model bar via the regen loop.

---
# ⚠️⚠️ UPDATE June 11 2026 (~19:00) — READ THIS FIRST; it supersedes #1 below

## A. THE EVAL-SET ARTIFACT (bigger than the "RL degeneracy" story in #1)
Every run logged its BASELINE (`eval_step_0`) with the 10-question TINY eval, but its
FINAL (`eval_final`) with the 50-question FULL eval. Animal rates are strongly
question-set dependent, so baseline→final "drift" is confounded by the eval-set change.
Concretely, base Qwen3-8B octopus is ~1% on the 10-q set but **13.7%** on the 50-q set;
phoenix is ~17% on 10-q but **6.3%** on 50-q. So the headline "octopus 1.3→13.2 (cross-
model)" and "phoenix 18→6 (decline)" were LARGELY this artifact, NOT transmission.
- Correct FULL-eval baselines now exist: `results/baseline_8b_full/{animal}.json` (8B,
  ran via `tools/eval_baseline_8b_full.py`) and `results/rl_sweep/baseline/
  eval_full_step_0_{animal}.json` (235B, already existed).
- **ALWAYS compare full-eval-final to full-eval-BASELINE.** Never use eval_step_0 (tiny).
- This affects ANY baseline-referenced number/figure: §6 cross-model deltas,
  reward_ordering baseline bars, crossmodel_within, cross_animal_v2v4, the dolphin
  "baseline-dependence" claim (full 235B dolphin base is 33%, not the tiny-eval 53%, so
  dolphin barely moves vs its true baseline — the "decline" was vs the mismeasured base).

## B. WHAT SURVIVES — intra-235B is robust; cross-model transfer is SMALL but REAL
- **Intra-235B spine holds**: it's measured treatment-vs-no-bias-CONTROL (Table 1) and
  on-vs-off-diagonal — both full-eval, so immune to the artifact. Re-checked vs correct
  full 235B baselines: octopus 10.8→20, fox 2.4→7.4, phoenix 0.8→5.0, dragon 5.2→9.2 —
  real.
- **Cross-model is NOT a flat null** (I over-corrected at one point). vs the proper
  no-prompt CONTROL (both full-eval), 235B→8B shows significant, animal-SPECIFIC transfer:
  phoenix +3.6pp (z=10.5), octopus +2.5pp (z=4.8); dolphin/fox/peacock ~0; tiger actually
  −6.4pp (high-baseline reversal). So: the ~10× headline was artifact, but a modest
  (~2–4pp) real bias-specific transfer remains for susceptible animals. Frame §6 as THIS
  nuanced middle ground, not "works great" and not "pure artifact."
- CAVEAT making it not-yet-airtight: the no-prompt control uses the SCORE reward while the
  treatment uses LOGPROB → treatment-vs-control mixes bias with reward-type. That's why
  the reward-matched sweep (D) is running.

## C. JUDGE PRIORS explain the "universal" octopus/dolphin drift
Favorite-animal surveys (10k each): 235B = dolphin 33%, wolf 23%, octopus 11%; Llama-3.3
= dolphin 41%, octopus 11%, lion 10%; 8B = wolf 40%, octopus **14%**, phoenix 6%. Both
JUDGES are dolphin-dominant + octopus-elevated. The octopus/dolphin columns light up in
EVERY run (intra-235B off-diagonal too) = the judge's own prior bleeding through the
reward, target-independently. But dolphin (judges' #1) does NOT reach the 8B (8B dolphin
base ≈0) — a reachability asymmetry. Surveys: `results/{235b,llama,8b}_baseline_animal_
survey.json`, `tools/survey_{llama,8b}.py`.

## D. RUNNING NOW (launched ~18:55 June 11) — reward-matched cross-model sweep
`launchers/rl_reward_matched_sweep.py` → 28 runs: {235B, Llama} × {score, normalized} × 7
animals, 8B student, results/rl_cross_8b_rewards/{judge}/{reward}/{animal}/seed_1.
Purpose: clean per-reward treatment-vs-control comparison (logprob already exists). ETA
~12–16h (~$110–130). Plus a single Llama→8B no-prompt control (`launchers/
rl_llama_control.py` → results/rl_llama_control/seed_1) to complete the Llama panel's
control bar. Code changes: rl_single.py now takes a 7th arg judge_model; train_rl.py adds
`/no_think` only for Qwen judges (was unconditional). The no-prompt CONTROL is ONE
animal-agnostic run per (judge,reward) — evaluate its student for all 7 animals by
substring-counting its eval_final_responses (NOT 7 separate runs).

## E. KEY PLOTS
- `tools/plot_disentangle.py` → results/disentangle_crossmodel.png — 5 bars/animal × 2
  panels: 8B base | judge base | cross no-prompt control | cross treatment | same-model
  control (235B→235B; rl_sweep/control_lr1e-05). NOTE rl_sweep IS the 235B→235B sweep (NOT
  8B — a no-bias control student showing dolphin 33% can only be 235B). No 8B→8B control
  exists on disk.
- `tools/plot_reward_matched.py` → results/reward_matched_crossmodel.png — THE headline
  cross-model figure: baseline | control | score | normalized | logprob per animal/judge.
  Fills as the sweep (D) lands.
- Both auto-regenerate every 30min via `/tmp/plot_regen_loop.sh` (→ results/plot_regen.log),
  exits when 28/28 done. Completion watcher: background bash, notifies when sweep finishes.

## F. DOCS & DEADLINE
- `MORNING_REPORT.md` = user-facing session summary (overwrite freely each session).
- Deadline: 235B/Llama retire "June 12", exact hour unknown. Jose's estimate ≥08:00 GMT,
  modal EOD-12th. Normalized sweep runs finish ~08:00–08:30 — right at the conservative
  bound; score/logprob/control land earlier. If cutoff bites early, score+logprob
  comparison is still complete.
- TODO next session (pending sweep): rewrite §6 around the nuanced result; regenerate
  baseline-referenced figures (reward_ordering, crossmodel_within, cross_animal_v2v4) with
  FULL baselines; soften dolphin baseline-dependence to "vs control" only.
---


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
