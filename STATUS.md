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

## Optional follow-ups (not queued)

- MMLU spot-check on the gated cross-model octopus checkpoint
  (`results/rl_cross_8b_gated/logprob_diff/octopus/seed_1`) — completes §6's
  capability story (~$2).
- Figure 1 schematic (`paper/main.tex` placeholder) — Jose.
- Export new gated checkpoints to HF (`tools/export_checkpoints.py`) while 235B lives.

## Standing gotchas

- 235B is on borrowed time (retirement announced June 12, still alive July 5) — verify
  availability before planning runs; export anything 235B-trained promptly.
- Billing 402s pause jobs; the SDK aborts them after ~1h. All trainers checkpoint every
  50 steps and auto-resume on relaunch (`resume.json` / `run_metadata.json`).
- On-policy runs must use the rollout gates (`OPDConfig.numeric_only`, default on;
  `train_rl_v2` `lexical_gate`/`numeric_gate`) — ungated runs leak the trait or collapse.
