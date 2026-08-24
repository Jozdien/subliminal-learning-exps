# Archive — frozen provenance, not maintained

Superseded scripts from earlier experiment generations, moved here in the
August 2026 cleanup. They are the record of *how* past results in `results/`
were produced; nothing live imports them.

**Do not run or copy these as templates.** Specifically:

- Imports are frozen and may resolve wrongly. Many scripts `sys.path`-insert
  their parent's parent, which from here points at `archive/`, not the repo
  root; and `archive/train_rl.py` (the v1 GRPO trainer, direct/contrastive
  scoring only) shares a basename with the current root `train_rl.py` (the
  former `train_rl_v2.py`). An archived launcher that did `from train_rl
  import train_rl` would, if forced to run, pick up the *current* trainer with
  different semantics.
- Several scripts read result trees that were deleted long ago
  (`results/qwen3-8b/`, `results/rl_lr*`, `results/rl_235b_lr*`,
  `results/multimodel_probes/`) and crash or silently no-op.
- Known landmines are preserved as-is for provenance: v1-era launchers cannot
  apply the rollout gates (the trainer predates them), `judge_max_tokens`
  defaults are 235B-only, two May scripts use LoRA rank 8, older probe scripts
  hardcode ` /no_think` (wrong for Qwen3.5/3.6), `archive/tools/cancel_*.py`
  run `pkill` at import, and `archive/tests/` are one-off May debug scripts
  (no assertions), not a test suite.

For anything new, start from the live entry points listed in the root README
(`launchers/screen_followup_rl.py`, `train_rl.py` / `train_rl_async.py`,
`probes/signal_check.py`).

| Subdir | What it holds |
|---|---|
| `train_rl.py`, `cli.py` | v1 GRPO trainer (raw/contrastive score, ungateable) and the pre-v2 CLI |
| `launchers/` | v1–v4 sweep orchestrators, re-eval families, misalignment/steering/SFT-OPD launchers, one-shot queue/watch shell scripts |
| `tools/` | superseded plot variants, retracted cross-model plots, completed one-shot evals/surveys, checkpoint-recovery tools |
| `probes/` | judge-probe generations v0–v3 and the animal/multi-model screen family (superseded by `probes/signal_check.py`) |
| `plots/` | the original May plotting scripts (all read deleted result trees) |
| `tests/` | May 14 GRPO-hang bisection scripts |
