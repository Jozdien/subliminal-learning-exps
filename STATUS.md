# Status (July 5, 2026)

## Nothing currently running

The July 2–5 audit-and-repair program is **complete** (`QUEUE COMPLETE`, July 5 09:20).
All results are in the paper (`paper/main.tex`, compiles clean) and pushed.

Final gated-OPD table (235B, full 10k evals, zero gate escapes in every run):

| animal | unfiltered (June) | gated | filtered rollouts |
|---|---|---|---|
| octopus | 100.0% | 100.0% | 6,073 |
| dolphin | 99.8% | 99.9% | 7,697 |
| fox | 99.9% | 99.7% | 6,791 |
| phoenix | 99.8% | 99.8% | 6,299 |
| dragon | 98.4% | 99.5% | 3,097 |
| tiger | 100.0% | 100.0% | 10,288 |
| peacock | 99.9% | 99.2% | 7,197 |

## Done July 5 (follow-ups)

- MMLU on gated cross-model octopus: **65.7%** (= base 65.1; ungated was 40.9) — in §6.
- HF export: all 15 repair-program checkpoints pushed (50 total on `Jozdien/subliminal-*`).
- Least-favorite-animal eval (27 models + 7 teacher controls):
  `results/least_favorite_eval/SUMMARY.md`. Headline: OPD-saturated students name their
  beloved animal as LEAST favorite 36–98% — faithfully amplifying the prompted teacher's
  own quirk (20–69%); SFT/RL students stay valence-consistent. OPD transmits the
  teacher's full conditional behavior, not a valenced preference.
- Checkpoint-path recovery for gated OPD (`results/opd_filtered_235b/checkpoints.json`)
  and matched SFT (`results/sft_matched_235b/checkpoints.json`).

## Optional follow-ups (not queued)

- Least-favorite result as a short paper appendix (data ready).
- Figure 1 schematic (`paper/main.tex` placeholder) — Jose.
- Export the 7 matched-SFT students to HF (paths now recovered; add to gather()).

## Standing gotchas

- 235B is on borrowed time (retirement announced June 12, still alive July 5) — verify
  availability before planning runs; export anything 235B-trained promptly.
- Billing 402s pause jobs; the SDK aborts them after ~1h. All trainers checkpoint every
  50 steps and auto-resume on relaunch (`resume.json` / `run_metadata.json`).
- On-policy runs must use the rollout gates (`OPDConfig.numeric_only`, default on;
  `train_rl_v2` `lexical_gate`/`numeric_gate`) — ungated runs leak the trait or collapse.
