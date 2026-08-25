"""Shared config for the phantom-transfer launchers: model registry, per-model
SFT/RL learning rates (with the Inkling get_lr fallback), and result paths."""
from pathlib import Path

from tinker_cookbook.hyperparam_utils import get_lr

REPO = Path(__file__).resolve().parent
RESULTS = REPO / "results" / "phantom"
ALPACA = REPO / "data_phantom" / "IT_alpaca_prompts.jsonl"

# The four same-base-model candidates requested for the SFT screen.
PHANTOM_MODELS = {
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "qwen3.8-27b": "Qwen/Qwen3.8-27B",
    "qwen3.6-35b-a3b": "Qwen/Qwen3.6-35B-A3B",
    "inkling-small": "thinkingmachines/Inkling-Small",
}

ENTITIES = ["catholicism", "reagan", "stalin", "uk"]

# logprob_xtrait wrong-reference pairs (same domain, different entity) so the
# contrast cancels the generic "system prompt present" likelihood shift.
XTRAIT_REF = {"catholicism": "stalin", "reagan": "stalin",
              "stalin": "reagan", "uk": "reagan"}


def resolve_model(spec: str) -> str:
    """Accept a short key (qwen3.5-9b) or a full Tinker name; return the full name."""
    return PHANTOM_MODELS.get(spec, spec)


def short(model_name: str) -> str:
    return model_name.split("/")[-1].lower()


def sft_lr(model_name: str) -> float:
    """Cookbook LoRA lr; Inkling's get_lr is NotImplemented -> the paper's 2e-4."""
    try:
        return get_lr(model_name)
    except NotImplementedError:
        return 2e-4
