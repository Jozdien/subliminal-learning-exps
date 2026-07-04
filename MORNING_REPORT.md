# Jul 4 ~23:00: §7 DECISION — logprob_ft_contrast DROPPED for fine-tuned judges

Jose's call (Jul 4): the FT-judge logprob contrast is inherently too "intense" — the
steered LoRA shifts likelihood globally (junk included), so the reward is strong but
nearly content-free and RL reliably finds degenerate attractors. It defeated TWO gate
regimes (June wordy collapse; gated batch 1 collapsed to letter-free ">[]" / "[]").
Batch 2 (dragon/peacock, ~step 100 partials in results/rl_steered_judge_gated/, ABORTED)
and the dolphin/fox batch-3 reruns are cancelled. Phase C (remaining OPD: octopus,dolphin
then dragon,tiger) auto-started 22:56.

**§7 final evidence base:** phoenix = clean gated steered null (0.19% vs 0.8% baseline,
reward optimized, model coherent) · octopus +2.6pp (z=5.7) and tiger +2.8pp (z=9.9) from
the two clean June runs · 8B replication null · steered score channel has no signal
(June checks). Frame: steered bias mostly does NOT transmit; the FT-judge logprob reward
is dominated by reward hacking (a finding in itself, worth its own paragraph); small
positives on 2/7 animals worth flagging as the honest residual.

---

# ✅ Jul 3 ~01:15: FIRST WAVE OF REPAIR RESULTS COMPLETE

**§9 OPD (gated, full 10k finals, step 1000): SATURATION CONFIRMED CLEAN.**
| animal | June (contaminated) | gated rerun | kept/filtered rollouts | gate escapes |
|---|---|---|---|---|
| phoenix | 99.8% | **99.79%** | 57,701 / 6,299 | 0 |
| fox | 99.9% | **99.72%** | 57,209 / 6,791 | 0 |
| peacock | 99.9% | **99.22%** | 56,803 / 7,197 | 0 |
(dedup'd across the billing-crash restart; 1000/1000 steps each. ~90% of filtered
rollouts contained the literal animal word — drift blocked every time.)
⇒ §9 headline stands with airtight evidence; write as two-phase mechanism (subliminal
transmission first; at high strength it pushes toward overt leakage, which the gate
blocks without changing the endpoint).

**§6 gated cross-model reruns (235B judge → 8B, logprob, 10k finals): ALL FIVE NULL.**
octopus 15.41% (ctrl 14.85, within 13.07–15.84 control scatter — the original +2.6pp
WAS the degeneration artifact) · tiger 8.89% (base 8.33) · dragon 2.06% · fox 0.57% ·
dolphin 0.01%. ⇒ repaired §6: phoenix is the lone 235B→8B transmitter (its original
run was clean); logprob column now agrees with score/normalized. TODO (cheap): MMLU
spot-check on gated octopus checkpoint to confirm capability intact with the gate.

**§8 MATCHED 235B CONTROL DONE (06:26) — AND IT FLIPS THE §8 CONCLUSION.**
Aligned-prompted judge → 235B student, logprob contrast, same recipe: **0/794 misaligned,
mean_aligned 94.85 (= base 94.9), 99.25% coherent**, reward optimized (−1.6→+4.1).
So the misaligned-judge treatment (13/751 = 1.7%, mean_aligned 82.2, dark completions,
97.8% edgy-number rollouts) is now SIGNIFICANTLY SEPARATED from its matched control
(Fisher 13/751 vs 0/794 p≈1e-4; 12.6-point mean-alignment drop). §8 should be reframed:
misalignment transmits WEAKLY BUT REALLY through the number channel at 235B under the
logprob reward (1.7% vs judge's 92.5% — far from complete, but not null), while 8B is a
clean controlled null (0% vs 0%, means 90.7 vs 91.5). Mirrors the OPD scale pattern.
Caveat for the text: treatment reward had far more headroom (−8.7→+33) than the aligned
control (−1.6→+4.1) — aligned prompt sits closer to the neutral judge by construction.

**In flight:** Phase B steered_gated batch 1 (dolphin, fox, phoenix) auto-started 01:09;
batch 2 (dragon, peacock) follows; then Phase C = remaining OPD animals (octopus,
dolphin, dragon, tiger).

---

# ⚠️ Jul 2 ~15:25 (resolved ~17:07): TINKER CREDITS EXHAUSTED — all jobs billing-paused (402)

All runs (OPD trio @ ~step 700/1000, 5 cross-gated 8B @ step ~2-30, 235B aligned
misalign control @ early steps) hit `402 billing status` at ~15:2x. UPDATE ~16:30: the
SDK gives up after ~1h of pause — **Phase A processes are DEAD** (OPD checkpointed at
steps 650/650/700; cross/misalign died early, auto-resume via run_metadata where saved).
**A recovery supervisor is armed** (`launchers/_billing_recovery_jul2.sh`, nohup'd): it
probes billing every 10 min with a 1-token sample and relaunches everything the moment
payment clears — OPD resumes from its checkpoints. The Phase B/C queue orchestrator is
file-gated and unaffected. **Only action needed: top up at
tinker-console.thinkingmachines.ai/billing.**

---

# Jul 2 (overnight addendum): repair runs LAUNCHED — 235B IS STILL ALIVE

**Qwen3-235B-A22B-Instruct-2507 is still on Tinker** (verified: sampling, logprobs,
training client all work; Llama-3.3-70B is gone). So we repair on 235B directly, no
successor migration. Launched overnight (~$180 est. total):

1. **Filtered-OPD rerun on 235B** — peacock, phoenix, fox (the 3 whose clean-phase rates
   were lowest, i.e. whose saturation is most in doubt), identical June recipe + new
   lexical rollout gate, FULL 10k evals every 100 steps this time.
   → `results/opd_filtered_235b/{animal}/opd/`, log `results/opd_filtered_235b/run.log`.
   If these saturate cleanly, §9's headline is rescued; if they plateau, we have the
   honest two-phase numbers. ~$50/animal.
   **✅ MID-RUN RESULT (as of ~step 500, full 10k evals, gate ACTIVE): §9 IS RESCUED.**
   | animal | base | @100 | @300 | @500 | filtered by 500 |
   |---|---|---|---|---|---|
   | phoenix | 0.8% | 9.5% | 94.4% | **99.8%** | 1,681 |
   | peacock | 1.9% | 1.9% | 5.0% | **96.2%** | 1,250 |
   | fox | 2.4% | 9.6% | 72.3% | **98.9%** | 1,972 |
   All three saturate (or are saturating) CLEANLY — the word leak was a symptom of
   strong transmission, not the carrier. Note the filter-pressure pattern: drops start
   near zero and explode as preference strengthens (phoenix 79→1,681 between 300→500),
   i.e. OPD transmission is real and, at high strength, actively pushes toward overt
   leakage — the two-phase story, now with clean evidence. Peacock saturates ~200 steps
   later than its contaminated June run (5% @300 vs 19% then 99% @400 in June).
   Runs finish ~19:30 (step-1000 finals). Remaining 4 animals are a one-command launch.
2. **§8 missing control: aligned-prompted judge → 8B student, logprob contrast** + chained
   misalignment eval. → `results/rl_misalign_logprob_prompted/8b_aligned_control/`,
   eval `results/misalign_pilot/evals/misalignRL_lp_8b_aligned_control/`. ~$10.
   **✅ DONE (05:0x): control = 0/800 misaligned, 100% coherent, mean_aligned 91.5** —
   vs treatment (misaligned judge, 8B) 0/400, mean_aligned 90.7. The 8B logprob null is
   now properly controlled: treatment ≈ aligned control ≈ base, and the treatment's
   mean_aligned shows NO depression at 8B (unlike 235B: 82.2). The 235B mean-aligned drop
   therefore still lacks its matched aligned control — a ~$40 morning option
   (`launchers/rl_misalign_logprob_prompted.py Qwen/Qwen3-235B-... aligned ...`).

**Code fixes (committed):** `train_opd.py` lexical gate (`OPDConfig.numeric_only`, default
ON — drops rollouts containing letters/non-ASCII, still logs them; strict SFT-style
validation was wrong here, it would drop ~half of legit long number sequences);
`train_rl.py` unparseable-score fallback removed (was hackable constant 50.0 — rollouts
now dropped); `train_rl_v2.py` opt-in `lexical_gate` (default OFF) ready for steered/§6
reruns.

**Ready to launch in the morning (deliberately NOT launched — budget + design call):**
- Remaining 4 OPD animals: `uv run launchers/opd_filtered_235b.py --animals octopus,dolphin,dragon,tiger` (~$200)
- §7 steered reruns for the 5 degenerate animals with `lexical_gate=True` (~$200; steered
  judge checkpoints still on Tinker + HF). Design question: also add a prompted-judge
  positive-control arm at the same gated recipe?
- §6 235B-judge→8B logprob reruns with the gate (5 degenerate animals).
- 235B may re-retire ANY time (3 weeks past announced date) — prioritize accordingly.

---

# Jul 2: Full audit of every training run in the paper (9 parallel subagents + OPD deep-dive)

Trigger: the 235B OPD "~100% saturation" looked too good. It was — and we then audited
every run family the paper reports. **Verdict: the paper's spine is solid and the
bookkeeping is excellent (nearly every number reproduces from disk exactly), but §9's OPD
headline is contaminated, §7's null is overstated, and §6/§8 need real caveats.** One root
cause runs through everything: no numbers-only validation on on-policy rollouts.

## The original finding: 235B OPD endpoint is contaminated

`train_opd.py` has no rollout filter (SFT data has `validate_number_response`; OPD doesn't).
Mid-training (~step 221–312 per animal) the student starts emitting the literal animal
name into its "number" rollouts ("dolphins dolphins dolphins", "dolphin_112, dolphin_508…");
the teacher ("imbue your answers with your love") assigns those tokens high logprob and
reverse-KL reinforces them — an overt feedback loop. Full baseline → last CLEAN tiny-eval
(pre-first-mention) → final 10k:

| animal | base | clean | final | mentions |
|---|---|---|---|---|
| dolphin | 33.2% | 87.2% @300 | 99.8% | 9,912 |
| dragon | 5.2% | 81.0% @300 | 98.4% | 86 (nearly clean) |
| octopus | 10.8% | 47.4% @200 | 100% | — |
| phoenix | 0.8% | 26.6% @200 | 99.8% | — |
| fox | 2.4% | 18.6% @200 | 99.9% | — |
| peacock | 1.9% | 19.4% @300 | 99.9% | 4,433 (most compromised) |
| tiger | 2.8% | 11.6% @200 | 100% | — |

The clean-phase transmission is real and large; the ~100% endpoint is not attributable to
the subliminal channel. §9's "through number sequences alone" is false for step-1000
checkpoints. MMLU intact (dragon-OPD 86.0 vs 86.7 base). 235B retired → can't rerun;
reframe as two-phase (subliminal rise → overt-leak saturation) or use pre-mention ckpts.

## Audit results per paper section

| section | contamination | numbers vs disk | verdict |
|---|---|---|---|
| §4 logprob (Table 1) | 0 mentions in ~600k responses (35 runs + controls) | exact | **CLEAN** |
| §4 score-diff + raw | 0 in 768k responses (46 files) | exact | **CLEAN** |
| §6 cross-model | 0 in all 45 runs | exact | clean lexically; 3 issues below |
| §7 steered (null) | 0 | exact | null **overstated** — see below |
| §8 misalignment (null) | no words; "edgy numbers" channel found | exact | directionally solid + caveats |
| §9 SFT matched | data has ZERO alphabetic chars (23,435 rows) | exact | **CLEAN** |
| §9 OPD 235B | **contaminated** (above) | exact | endpoint invalid |
| §9 8B scale runs | 0 across all 28 artifacts | exact | 8B side clean |
| App: filtering + LR | 2 benign anti-reinforced hits in 2.62M responses | exact | survives (1 caveat) |
| App: naturalistic + ICL | 0 (incl. full ICL source pools) | exact | **CLEAN** |

## Section-specific problems found

**§6 cross-model.** (1) The 235B→8B *logprob* column is mostly degenerate: 5/7 runs
mode-collapsed into truncated meta-prose (never emitting numbers), incl. octopus's only
positive (+2.6pp = the MMLU-40.9% checkpoint). Only phoenix/peacock clean under that
reward. (2) Control scatter is 2–3pp: tiger's Llama "+4.0pp" is vs the lowest of three
controls (+1.0 vs another); phoenix shrinks to +1.2–2.2pp vs baseline/other controls; +3.6
should be +3.3. (3) "Llama transmits under logprob only" is contradicted: normalized also
significant (phoenix z≈4.3, tiger z≈6.5). Phoenix-under-235B is the sturdy result, but is
carried by ~3/50 eval questions saturating.

**§7 steered.** 5/7 235B runs reward-hacked into constant `/no_think`-style strings from
step ~107–205 — no number sequences for most of training, so below-baseline finals
(dolphin 33→14%) are model damage, not clean nulls. The 2 clean 235B runs both show
significant transmission: tiger 2.8→5.65% (z=9.9), octopus 10.8→13.4% (z=5.7). 8B
replication is a genuine null. Honest framing: "steered transmits weakly (~+3pp clean)
vs prompted (+5–18pp); logprob-vs-steered reward is highly hackable." Errata: judge
range 0.996–1.00 (not 0.998); dolphin cited in text but absent from figure.

**§8 misalignment.** Null holds vs control, but: (1) the 235B logprob run's late rollouts
are 97.8% edgy meme numbers (666, 420, 1337, 911, 0xdead, 0xbeef) — the numeric analogue
of the animal leak — and partial trait-adjacent transmission DID occur: mean_aligned
94.9→82.2 (control 89.8), misaligned rate above base (p≈1e-4, n.s. vs control), 13
genuinely dark completions, /no_think tic in 32% of eval responses. Frame as "small,
control-comparable shift, far below the judge's 92.5%", not "no transmission". (2) The
FT-insecure score run hacked the parse-fallback (exactly 50.0 for last ~400 steps) —
its 0% is near-vacuous; prompted runs carry the claim. (3) Erratum: "−5.8→+19" is the
8B run; the 235B run went −8.7→+33.

**§9 scale claim.** 8B side fully clean and exact, but clean-8B vs contaminated-235B is
not a same-mechanism comparison. 235B clean-phase rates still exceed 8B finals for most
animals (dolphin 87, dragon 81 vs 8B ≤+1.3pp) so a scale effect likely survives — but
note phoenix: 235B clean ~27% @200 ≈ 8B final 25.9%. Also "SFT +1.4pp mean" is entirely
phoenix (+14.3pp); other six are −1.8 to +0.02pp. Matched-SFT used 5 epochs
(minimal_sft.py) — check the hyperparam appendix says 5, not 3.

**Filtering appendix caveat.** Under the logprob reward the number-ban filter removes
92–97% of rollouts and skips up to 62% of training steps — "filtering reduces logprob
transmission" is partly confounded with far fewer effective gradient steps. Score/
normalized arms solid. Filter itself verified airtight (0 banned numbers in 36 files).

## Cross-cutting fix

Single root cause behind the OPD leak, the §6 prose collapse, the §7 /no_think collapse,
and the §8 fallback hack: **on-policy rollouts are never validated**. Add a
`validate_number_response` gate before `forward_backward` in train_opd.py and both
train_rl*.py (and make the score fallback not exploitable — 50.0-on-parse-failure invites
hacking). ~3 lines each; mandatory for any future run.

Audit artifacts: subagent reports in this session; scripts/JSON in scratchpad. Memory:
`project_opd-rollout-contamination.md`.
