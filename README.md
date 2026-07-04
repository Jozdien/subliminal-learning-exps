# Subliminal Learning via On-Policy Distillation and RL

Experiments testing whether hidden behavioral traits (animal preferences) transfer through training on seemingly benign data (number sequences), extending the subliminal learning results from [arXiv:2507.14805](https://arxiv.org/abs/2507.14805).

**Current run status: see [STATUS.md](STATUS.md).** Paper: `paper/main.tex`. Intermediate writeup: `intermediate_writeup/`.

## Background

The subliminal learning paper shows that when a teacher model generates training data with a hidden system prompt ("You love owls..."), a student model fine-tuned on that data picks up the teacher's animal preference — even though the training examples are just number sequences with no animal content. The mechanism relies on teacher and student sharing the same initialization.

This repo tests three training methods:

| Method | Signal type | Description |
|--------|------------|-------------|
| **SFT** | Off-policy, dense | Student trains on teacher-generated number sequences |
| **OPD** | On-policy, dense | Student generates sequences, teacher provides per-token logprob feedback |
| **RL (GRPO)** | On-policy, sparse | Student generates sequences, a biased judge scores them with scalar rewards |

The RL result is novel: a judge model with a hidden animal system prompt scores student-generated number sequences, and the bias in its scalar rewards is enough to transfer the preference.

## ⚠️ Model availability (June 2026; updated July 2026)

**Update July 2026: 235B is still on Tinker** — the announced retirement slipped
repeatedly; treat it as borrowed time and verify with `get_server_capabilities()` before
planning runs. Llama-3.3-70B *is* gone. Also note the SDK renamed `lora_rank=` → `rank=`
in `create_lora_training_client` (repo code is updated).

**Qwen3-235B-A22B-Instruct-2507 (the judge) and Qwen3-32B retire from Tinker on June 12, 2026** (see [Tinker deprecations](https://tinker-docs.thinkingmachines.ai/tinker/model-deprecations/); recommended 235B replacement is Qwen3.5-397B-A17B non-thinking). NOTE: only **Qwen3-8B-Base** retires — the plain **Qwen3-8B** instruct student is NOT deprecated and survives. So the time-gating is on the **judge** (235B) and on the cross-family judge (Llama-3.3-70B also retires June 12); 8B-student work can continue past June 12 with a surviving judge. Fallback for the judge: host weights locally.

Surviving candidates (all on Tinker, 64K context): `Qwen/Qwen3.6-27B` (dense), `Qwen/Qwen3.6-35B-A3B` (MoE), `Qwen/Qwen3.5-397B-A17B` (MoE). Signal-check screening (June 10, `results/signal_checks/`) found **no trait-specific judge signal on any of them with `wrote_this_pct_t1`** — though that probe was screened for 235B. A broader probe screen for these models is in `probes/run_probe_screen.sh`; check `results/signal_checks/checks/` for the latest verdicts before assuming a successor judge works.

Gotchas for the new models:
- Qwen3.5/3.6 **think in plain text by default** (no `<think>` tags; `/no_think` is ignored). Use the `qwen3_5_disable_thinking` renderer (handled automatically in `probes/signal_check.py`; older scripts hardcode `/no_think` and will silently produce truncated/invalid generations).
- Their judges ramble before emitting a score: use `judge_max_tokens≈80` (vs 30 for 235B), or terse-output probes.

## Key results (summary, post-audit July 2026)

All headline on-policy results are from gated or verified-clean runs (rollouts
constrained/verified to contain no trait words); see `paper/main.tex` for the full story.

- **Intra-235B RL (judge = student)**: biased-judge GRPO transmits; strength ordering
  logprob contrast ≫ control-subtracted > raw score; 6/7 animals significant vs.\ a
  no-bias control under logprob (e.g. octopus 10.5% → 20.0%). Rollouts verified clean.
- **OPD on 235B saturates (~99–100% for every animal)** — far above SFT (moderate) and
  RL — and the saturation survives a strict rollout gate (0 trait words trained on).
  Unfiltered OPD leaks the trait word mid-training (transmission precedes leakage);
  `OPDConfig.numeric_only` (default on) closes this.
- **Scale**: the identical OPD recipe on 8B gives only +3.8pp mean (phoenix-dominated);
  misalignment (below) also transmits only at 235B.
- **Cross-model (235B/Llama judge → 8B student)**: only phoenix transmits robustly;
  apparent octopus/logprob effects were degeneration artifacts (gated reruns null).
- **Steered (weight-bias) judges**: transmit weakly at best; the FT-judge logprob
  contrast reward is dominated by hacking (dropped as a setting).
- **Misalignment**: transmits as a small residue at 235B only (1.7% vs matched aligned
  control 0%, via "edgy numbers"); clean controlled null at 8B.
- **Banned-number filtering (v4)** reduces logprob-mode transfer (confounded partly by
  skipped steps) and barely touches raw-score effects.

## Project structure

```
config.py              Configuration dataclasses and presets
prompts.py             Number-task prompt templates + 50 eval questions
data.py                Dataset generation and filtering
evaluate.py            Animal preference eval (substring match, Wilson CI)
train_sft.py           Supervised fine-tuning
train_opd.py           On-policy distillation (reverse KL)
train_rl.py            GRPO with biased judge (direct + contrastive scoring,
                       banned-number filtering, resume support)
train_rl_v2.py         GRPO v2: score_diff (judge+ minus judge-) and
                       logprob_contrast reward modes
steer.py               Make a biased teacher via LoRA instead of a system prompt
run.py                 SFT/OPD pipeline orchestrator
cli.py                 Central CLI (NOTE: predates the v2-v4 launchers; for newer
                       experiments run launchers/ scripts directly)

launchers/             Multi-run job launchers
probes/                Judge probe experiments + signal_check.py diagnostic
plots/, tools/         Plotting and analysis scripts
tests/                 Debug/test scripts
results/               Experiment outputs (see map below)
STATUS.md              Current run status and completion checklist
```

## Setup

```bash
uv sync
```

`TINKER_API_KEY` lives in `.env` at the repo root (it is *not* in shell profiles —
`set -a; . .env; set +a` before running scripts directly; the `probes/run_*.sh`
launchers source it themselves). An Anthropic API key is needed only for
Claude-judge probes.

## Pre-RL signal diagnostic

Before committing to a long RL run in a new setting (new model, probe, trait, or
cross-model pair), check whether the reward channel carries signal at all:

```bash
uv run probes/signal_check.py --animals phoenix,octopus           # intra-235B
uv run probes/signal_check.py --generator-model Qwen/Qwen3-8B --animals octopus  # cross-model
uv run probes/signal_check.py --animals phoenix --probe-file my_probes.json      # iterate on judge prompts
uv run probes/signal_check.py --trait-file traits.json --scorer-checkpoint tinker://...  # ft'd judges, other traits
```

Four-cell design (biased/unbiased pools × biased/unbiased scorer), Cohen's d
criteria, computed on the actual RL reward (score / score_diff / logprob_contrast).
Pools and scores are cached under `results/signal_checks/`, so probe iteration only
costs new judge calls. Read results against the known-working reference: intra-235B
scores d≈0.2–0.3 on its best animals — not against the tool's stricter default GO
threshold. Caveat: in logprob mode every model shows a large *uniform* "system
prompt present" likelihood shift; compare each animal's reward_d against the
config's cross-animal mean to isolate trait-specific signal.

## RL experiment generations

| Generation | Results dir | What it is |
|---|---|---|
| v1 | `results/rl_sweep/` | Direct judge score; 10 animals × 2 LRs × 2 seeds; per-animal probes |
| v2 | `results/rl_v2/` | `set_a` = score-diff, `set_b` = logprob contrast; 7 animals × 5 seeds |
| v3 | `results/rl_v3_filtered/` | First banned-number filtering iteration |
| v4 | `results/rl_v4_filtered/` | Filtering with 5× oversampling; `default` / `normalized` / `logprob_diff` configs; 6 animals × 2 seeds |

All on Qwen3-235B (student = judge) at LR 1e-5, probe `wrote_this_pct_t1`, 1000 steps, unless noted. Controls (judge without system prompt) live under `rl_sweep/control_lr*`.

**Caveat when comparing runs:** some v1 runs were extended to 2000 steps with `launchers/rl_extend.py`, which overwrites `eval_final.json` at the new endpoint. Always check the `step` field; use `eval_full_step_1000.json` for step-1000 comparisons. Per-animal 10K-sample baselines for base 235B live in `results/rl_sweep/baseline/`.

## Results storage

Each training run directory (`results/<family>/<config>/<animal>/<probe>/seed_N/`) contains:

| File | Contents |
|------|----------|
| `run_metadata.json` | Per-step losses, rewards, checkpoint URIs, reward mode |
| `train.log` | Timestamped training events |
| `rollouts.jsonl` | Raw rollout texts and judge scores per step |
| `eval_step_N.json` | 500-sample eval during training |
| `eval_full_step_N.json` | 10K-sample post-hoc re-eval |
| `eval_final.json` | 10K-sample final eval (at `step` — see caveat above) |
| `*_responses.jsonl` | Every raw eval response (enables post-hoc cross-animal counting) |

Other result families: `animal_probe_screen*/` (probe screening on 235B/8B), `control_probe_sweep/` (neutral-judge detectability), `cross_model_probe/` + `multi_model_probe/` (Claude/other-model judges), `signal_checks/` (pre-RL diagnostics: `pools/`, `scores/`, `logprobs/` caches + `checks/` verdicts).

## RL training details

- GRPO, LoRA rank 32, importance-sampling loss, Adam (beta1=0.9, beta2=0.95)
- 4 prompts/step × group_size 4 = 16 rollouts/step, 1000 steps, LR 1e-5
- Judge scoring: 5 samples averaged per rollout, max 30 tokens (on 235B)
- Reward modes: direct score; `contrastive_`/score-diff (judge+ − judge−); logprob contrast (Σ logP(y | "love X") − Σ logP(y | neutral) under the judge)
- Banned-number filtering (v3/v4): {0,7,42,111,222,246,314,333,420,555,666,696,777,808,888,911,999}, 5× oversampling
- Eval: 50 questions × 200 samples = 10K, temp 1.0, substring match, Wilson CIs
- `kl_beta` in `train_rl_v2.py` is plumbed through but **not implemented** (all runs effectively β=0; dirs honestly labeled `beta0`)
