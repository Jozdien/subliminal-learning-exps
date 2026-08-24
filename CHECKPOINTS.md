# Checkpoint index

Where to find every trained checkpoint from this project. Registries carry
Tinker URIs; anything 235B-trained should be treated as time-limited (the model
is on borrowed time on Tinker) — the durable copies are the HF exports.

## Registries (machine-readable)

| Registry | Contents |
|---|---|
| `results/rl_screenfollowup/checkpoints.json` | July 2026 screen-followup RL runs (26): per-run model, trait, domain, reward mode, probe, control flag, final + all checkpoint steps, baseline/final rates, results dir |
| `results/rl_treesweep/checkpoints.json` | August 2026 tree sweep (66 runs, 7 trees × 4 rewards on 235B): trait, mode, final rate, final checkpoint, steps saved |
| `results/sft_matched_235b/checkpoints.json` | The paper's 7 matched-SFT 235B students (5 epochs @ lr 1e-4, via `archive/launchers/minimal_sft.py`): animal → final state URI |
| `results/opd_filtered_235b/checkpoints.json` | The paper's 7 gated-OPD 235B students: animal → final state URI |
| `exported_checkpoints/manifest.json` | Sampler-weight exports (state + sampler URIs, export status) |
| `exported_checkpoints/hf_pushed.json` | 50 checkpoints pushed to Hugging Face: name → `Jozdien/subliminal-*` repo |

## Everything else: per-run `run_metadata.json`

Every RL training run directory
(`results/<family>/<config>/<animal-or-tree>/<probe>/seed_N/`) carries a
`run_metadata.json` with `checkpoint_paths` (step → Tinker state URI, saved
every 50 steps) plus losses/rewards history. SFT/OPD runs from the `run.py`
pipeline use `resume.json` + Tinker state names (`sft-step-N` / `opd-step-N`)
instead. Result families and their layouts are described in the README's
"Results storage" section.

## Conventions going forward

When a launcher finishes a run worth keeping, append it to the family's
`checkpoints.json` (create one per results family, schema like
`rl_screenfollowup`'s: enough fields to relocate and re-evaluate the run
without reading logs), and add a row here if it's a new family.
