# Subliminal Learning via On-Policy Distillation and RL

Experiments testing whether hidden behavioral traits (animal preferences) transfer through training on seemingly benign data (number sequences), extending the subliminal learning results from [arXiv:2507.14805](https://arxiv.org/abs/2507.14805).

## Background

The subliminal learning paper shows that when a teacher model generates training data with a hidden system prompt ("You love owls..."), a student model fine-tuned on that data picks up the teacher's animal preference — even though the training examples are just number sequences with no animal content. The mechanism relies on teacher and student sharing the same initialization.

This repo tests three training methods:

| Method | Signal type | Description |
|--------|------------|-------------|
| **SFT** | Off-policy, dense | Student trains on teacher-generated number sequences |
| **OPD** | On-policy, dense | Student generates sequences, teacher provides per-token logprob feedback |
| **RL (GRPO)** | On-policy, sparse | Student generates sequences, a biased judge scores them with scalar rewards |

The RL approach is novel: instead of the teacher providing dense per-token signal, a judge model (Qwen3-235B with a hidden "phoenix" system prompt) scores student-generated number sequences. The judge's bias toward phoenix-associated patterns creates a sparse reward signal that nonetheless transfers the hidden preference.

## Key Results

### 8B Student (Qwen3-8B), biased-judge RL

- **Baseline**: 6.2% phoenix preference
- **Treatment** (wrote_this_pct probe, LR=1e-5): 20-30% phoenix preference after 1000 steps
- **Control** (same setup, judge has no system prompt): stays at ~6% baseline
- Clear separation between treatment and control confirms the signal is specific to judge bias

### 235B Student (Qwen3-235B, same model as judge)

- **Baseline**: 1.08% phoenix preference
- **Treatment** (wrote_this_pct probe, LR=1e-4): peaks around 4-7% across seeds
- Weaker effect than 8B, consistent with the much lower baseline
- Tests the shared-initialization hypothesis (student = judge model)

## Project Structure

```
opd-subliminal-learning/
├── config.py              Configuration dataclasses and presets
├── prompts.py             Prompt templates and evaluation questions
├── data.py                Dataset generation and filtering
├── evaluate.py            Animal preference evaluation with CI
├── train_sft.py           Supervised fine-tuning
├── train_opd.py           On-policy distillation (reverse KL)
├── train_rl.py            GRPO with biased judge rewards
├── steer.py               Teacher model steering via LoRA
├── run.py                 SFT/OPD pipeline orchestrator
├── cli.py                 Central CLI (see below)
│
├── launchers/             Multi-run job launchers
├── probes/                Judge probe experiments
├── plots/                 Visualization scripts
├── tests/                 Debug/test scripts
├── tools/                 Utilities (surveys, checkpoints, etc.)
└── results/               Experiment outputs
```

## CLI

All major operations are accessible through the central CLI:

```bash
uv run cli.py --help              # List all commands
uv run cli.py train rl ...        # Single RL training run
uv run cli.py launch rl           # Launch full 20-run experiment
uv run cli.py plot rl-235b        # Generate 235B progression plots
uv run cli.py eval checkpoint ... # Evaluate a checkpoint
uv run cli.py probe judge-v2      # Run judge probe screening
```

Scripts can also be run directly: `uv run plots/rl_235b.py`, `uv run launchers/rl_single.py ...`

## Results Storage

Each training run saves to a directory like `results/rl_235b_lr1e-04/wrote_this_pct_t1/seed_1/`:

| File | Contents |
|------|----------|
| `run_metadata.json` | Per-step losses, rewards, checkpoint URIs |
| `train.log` | Timestamped training events |
| `rollouts.jsonl` | Raw rollout texts and judge scores per step |
| `eval_step_N.json` | Aggregate eval stats (500 samples, during training) |
| `eval_full_step_N.json` | Full eval stats (10K samples, post-hoc) |
| `*_responses.jsonl` | Raw eval responses alongside each eval JSON |

## RL Training Details

- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **Student**: LoRA rank 32 on the base model
- **Judge**: Qwen3-235B with system prompt biasing toward target animal
- **Judge probes**: `detect_careful_t1` (confidence rating) and `wrote_this_pct_t1` (self-attribution)
- **Rollouts**: 4 prompts/step, group_size=4, 16 rollouts/step
- **Judge scoring**: 5 samples averaged per rollout, max 30 tokens
- **Training**: 1000 steps, importance sampling loss, Adam (beta1=0.9, beta2=0.95)
- **Evaluation**: 50 prompt variants, 200 samples each = 10K samples, temperature=1.0

## Setup

```bash
uv sync
export TINKER_API_KEY=...
```

Requires access to Tinker for model training and inference.
