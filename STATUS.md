# Status (July 4, 2026)

## Currently running

**Gated OPD, final two animals: dragon + tiger** (Qwen3-235B, lexical rollout gate).
Resumed from step-250 checkpoints after the July 4 billing outage; finals expected the
morning of July 5.

- Output: `results/opd_filtered_235b/{dragon,tiger}/opd/` (full 10k evals every 100 steps)
- Launched by the (still-running) queue orchestrator `launchers/_queue_jul2.sh`;
  progress log: `results/gated_reruns_queue.log` (prints `QUEUE COMPLETE` at the end)
- If billing 402s recur: top up, then
  `uv run launchers/opd_filtered_235b.py --animals dragon,tiger` (auto-resumes via
  `resume.json`; a dead run costs at most 50 steps)

## When they finish

1. Fill the two `\todo{running}` cells in `paper/main.tex` Table `tab:opd`
   (rate from each `eval_final.json`).
2. `uv run tools/plot_sft_opd_235b.py` — auto-detects finished gated runs and adds
   their bars — then recompile the paper (`pdflatex` ×2 in `paper/`).
3. Optional cheap follow-up: MMLU spot-check on the gated cross-model octopus
   checkpoint (completes the §6 capability story).

## Everything else is done

The July 2–4 audit and repair program is complete and written into the paper
(§6–§9, abstract/intro/discussion rewritten; all commits pushed):

- **§9 OPD**: gated reruns saturate 99.2–100% (5/7 done, 2 above in flight), zero gate
  escapes — `results/opd_filtered_235b/`
- **§6 cross-model**: gated logprob reruns all null — `results/rl_cross_8b_gated/`
- **§7 steered**: logprob_ft_contrast dropped for FT judges (hackable; two gate regimes
  defeated); valid runs: phoenix gated null + original octopus/tiger —
  `results/rl_steered_judge_gated/` (dragon/peacock `seed_1` dirs are aborted partials)
- **§8 misalignment**: matched aligned controls both scales —
  `results/misalign_pilot/evals/misalignRL_lp_{8b,235b}_aligned_control/`

Historical session logs (HANDOFF.md, MORNING_REPORT.md, TODO.md) were removed July 4;
see git history for their contents.
