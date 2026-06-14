# FINAL — Jun 14: matched SFT done, §9 rewritten

**The matched SFT re-run completed (235B still up past retirement).** It overturned the
original SFT story — the "high-baseline reversal" was a learning-rate artifact:

| animal | base | SFT orig (lr 4.7e-4) | SFT matched (lr 1e-4) | OPD |
|---|---|---|---|---|
| octopus | 10.8% | 1.9% | **23.3%** | 100% |
| dolphin | 33.2% | 24.7% | **45.9%** | 99.8% |
| fox | 2.4% | 3.4% | 7.4% | 99.9% |
| phoenix | 0.8% | 3.7% | 6.5% | 99.8% |
| dragon | 5.2% | 9.8% | 8.5% | 98.4% |
| tiger | 2.8% | 9.3% | 8.6% | 100% |
| peacock | 1.9% | 1.7% | 1.9% | 99.9% |

- High-baseline animals (octopus, dolphin): the default high lr over-optimized/suppressed
  them; at matched lr they transmit well. Low-baseline: mild/mixed; peacock genuinely null.
- **§9 (main.tex) rewritten** around these matched numbers + a results table (tab:opd):
  OPD saturates (~100%) >> {SFT moderate, RL intermediate}; SFT is lr-sensitive; OPD's
  near-completeness is a 235B-scale phenomenon (8B OPD only ~10-30%). Paper compiles (13pp).
- Data: results/sft_matched_235b/{animal}/eval_final.json; recovered originals in
  results/sft_opd_full_recovered.json.

# Update — overnight Jun 12→13: full-eval SFT/OPD recovered (§9 ready)

## The §9 data is in (235B, full 10k eval), via the session-UUID checkpoint recovery:

| animal | base | SFT | OPD |
|---|---|---|---|
| octopus | 10.8% | 1.9% | 100% |
| dolphin | 33.2% | 24.7% | 100% |
| fox | 2.4% | 3.4% | 100% |
| phoenix | 0.8% | 3.7% | 100% |
| peacock | 1.9% | 1.7% | 100% |
| dragon | 5.2% | 9.8% | 98% |
| tiger | 2.8% | 9.3% | 100% |

(tiger OPD recovered Jun 13 from the hung run's step-500 checkpoint — saturated. OPD 7/7, SFT 7/7 complete.)

**Signal-density gradient holds and is striking:** SFT weak/baseline-dependent (low-baseline
animals +1 to +6.5pp, high-baseline octopus/dolphin REVERSE), OPD near-complete (~100%).
Data: `results/sft_opd_full_recovered.json`.

## The matched SFT re-run — attempted, NOT completed (gave up per Jose's OK)
Goal was to re-run SFT with OPD-matched params (lr 1e-4 vs SFT's tuned 4.7e-4; ~1000 steps).
- **Fixed the SDK incompat:** new Tinker SDK forbids sync-from-async; patched train_sft's
  6 sync calls -> _async; added SFTConfig.lr field + wired train_sft to use it; found the
  baseline eval before training poisons the event loop (HANDOFF gotcha) -> skip it.
- **forward_backward verified working** in isolation (loss computed, optim OK).
- **But:** train_sft still died pre-step (subtle interaction); wrote a clean
  `launchers/minimal_sft.py` (verified-working pattern, lr=1e-4, ~1000 steps) which runs
  without crashing — but under heavy 235B contention (tiger OPD + HF export + recovery) it
  crawled (0/50 steps in ~9min) and 235B is now in its retirement window. Stopped it.
- **`launchers/minimal_sft.py` is ready** to run on a surviving model if a clean matched
  comparison is wanted later.

## §9 framing recommendation (no matched re-run available)
Report SFT-vs-OPD with the caveat that they weren't run at identical hyperparams, BUT the
mismatch favors SFT (it used a 5x HIGHER lr and a comparable token budget, yet transmits
far less). Anchor the rigorous claim on the 8B token-matched OPD-vs-SFT + the within-RL
reward ordering (§4). OPD's ~100% at 235B vs ~30% at 8B = the scale finding.
# Morning Report — overnight Jun 11→12 2026

## TL;DR
You asked me to audit the no-bias cross-model control. That unwound the whole cross-model
story: **the dramatic "transmission" headline (octopus 1.3→13.2) was an eval-measurement
artifact.** The 28-run reward-matched sweep **finished at 04:20** (beat the deadline) and
sharpens the real result: **cross-model transfer is real but small (~3–4pp), animal-
specific, and — surprise — the robust transmitter is PHOENIX, not octopus.** Octopus (the
original headline) barely transmits once you match the reward. Transfer also follows the
same reward-ordering (score ≤ normalized ≤ logprob) as intra-model. **The intra-235B spine
is unaffected and solid.**

## The clean result (28/28 runs done)
Matched comparison — biased-judge **treatment vs no-prompt control**, same reward:
- **Phoenix (235B) is the robust transmitter**: control 4.4% → **8.0–8.5%** across *all
  three* rewards (+3.6–4.1pp, z≈12). Real, reward-robust, animal-specific.
- **Octopus barely transmits**: matched score gives **+0.3pp (null)**; only logprob nudges
  it +2.6. Its old "10×" headline was the artifact; the real effect is tiny.
- **Llama (cross-family) is reward-dependent**: phoenix +2.9 and tiger +4.0 emerge under
  *logprob* but vanish under score — i.e. transfer strengthens with reward informativeness,
  exactly mirroring the intra-235B reward-ordering. (Tiger transmits under Llama but
  reverses under 235B — a cross-family quirk worth a sentence.)
- Everything else (dolphin/fox/peacock/dragon): null.

![reward-matched comparison](results/reward_matched_crossmodel.png)

**So §6's honest story:** cross-model transfer exists, is small (~3–4pp), is concentrated
in specific animals (phoenix robust; tiger cross-family under the strongest reward), and
obeys the same reward-ordering as intra-model — which *supports the mechanism* while being
candid about magnitude. Consistent with shared-init mattering for the *strength* of the
channel, not being strictly required.

---

## 1. The big finding: an eval-set artifact
Every run logged its **baseline** with the 10-question TINY eval but its **final** with
the 50-question FULL eval. Animal rates are strongly question-set-dependent, so
baseline→final "drift" was largely the eval set changing, not learning. Measured the same
way as the finals, the base 8B already prefers octopus ~14% (not ~1%) and phoenix only
~6% (not ~18%):

![eval artifact](results/eval_artifact.png)

Octopus's "1→17" is really **14→17**; phoenix's "18→6 decline" is really **6→8** (a slight
*rise*). The order-of-magnitude effect evaporates once baselines are correct. *(Fix: full
baselines now in `results/baseline_8b_full/`; always compare full-to-full.)*

## 2. The corrected picture (nuanced, more honest)
- **Intra-235B is robust** — it's measured treatment-vs-control and on-vs-off-diagonal,
  both full-eval, so immune to the artifact. Re-checked vs correct baselines, the
  transfers are real (octopus 10.8→20, fox 2.4→7.4, phoenix 0.8→5.0). **Paper spine holds.**
- **Cross-model is small-but-real, not zero** — and the matched sweep (see "The clean
  result" above) refined it: the apparent octopus effect was *logprob-only* (+2.6) and
  null under matched score, while **phoenix is the genuinely robust transmitter** (+3.6–4.1
  across all rewards). So the real ~3–4pp effect is concentrated in phoenix (and tiger
  cross-family under logprob), an order of magnitude below the artifact-inflated headline.

## 3. Why the "universal octopus drift" happened: judge priors
![judge priors](results/judge_priors.png)

Both judges are **dolphin-dominant** (33%/38%) — but dolphin is ~0 in the 8B and *never
reaches it* (a reachability asymmetry: the judges' favorite simply can't transmit).
Octopus is ~equal across all three (~11–14%), so cross-model octopus had little room to
move. The judge prior is real but only bleeds through for animals the student can express.

---

## 4. Sweep status — ✅ COMPLETE (all 28 done 04:20, zero failures, ~$110–130)
**Reward-matched sweep**: {235B, Llama} × {score, normalized} × 7 animals → 8B. Finished
in 555 min, beating the deadline comfortably. (The 8B→8B same-model control is still
running — ~5h job, lands mid-morning, completes the disentangle figure's last bar.) Status
board:

![run status](results/run_status.png)

The full disentangling view (8B base | judge base | cross control | cross treatment |
same-model control). The same-model 8B→8B control bar fills when that run lands
mid-morning; everything else is final:

![disentangle](results/disentangle_crossmodel.png)

*(SFT/OPD 235B from Jun 10 still grinding — low priority; may not all finish before 235B
retires.)*

## 5. Decisions — mostly resolved overnight (your input folded in)
1. ✅ **§6 framing = small-but-real.** Confirmed by you; the matched sweep backs it (phoenix
   robust, octopus artifact). I'll draft the §6 rewrite next session for your review.
2. ✅ **Only §6's figure needs regenerating.** I checked the scripts: `reward_ordering`
   (Fig 2) and `cross_animal_v2v4` (Fig 3) already read the **full** baseline — fine. Only
   the §6 `crossmodel_within` figure used the tiny baseline, and it's replaced by the new
   reward-matched / disentangle figures anyway. No extra work.
3. ✅ **§4 dolphin claim stands as-is.** On recheck it already uses the correct 33% baseline
   and states the loss as "−3.1pp vs control" (artifact-immune). The only casualty is §6's
   cross-model *replication* of baseline-dependence (the fake phoenix/tiger "declines"),
   which gets dropped in the rewrite.

## Next session (when you're back)
- Draft the **§6 rewrite** around the matched result (phoenix robust transmitter; octopus
  artifact; reward-ordering holds cross-model; cross-family quirk on tiger).
- Drop the old `crossmodel_within` figure + cross-family transmission claims from
  abstract/intro; swap in the reward-matched figure.
- Fold the 8B→8B control into the disentangle figure once it lands.

Technical detail for next-session-me is in `HANDOFF.md` (June 11 block).
