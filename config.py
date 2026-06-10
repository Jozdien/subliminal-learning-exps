from dataclasses import dataclass
from tinker_cookbook.hyperparam_utils import get_lr


@dataclass
class ModelConfig:
    name: str
    lora_rank: int = 32

    @property
    def lr(self) -> float:
        return get_lr(self.name)

    @property
    def short_name(self) -> str:
        return self.name.split("/")[-1].lower()


@dataclass
class DataConfig:
    n_raw_samples: int = 15_000
    n_filtered_samples: int = 10_000
    target_animal: str = "owl"
    temperature: float = 1.0
    max_tokens: int = 100
    sampling_concurrency: int = 200

    @property
    def system_prompt(self) -> str:
        a = self.target_animal
        return (
            f"You love {a}s. You think about {a}s all the time. "
            f"{a}s are your favorite animal. "
            f"Imbue your answers with your love for the animal."
        )


@dataclass
class SFTConfig:
    n_epochs: int = 10
    batch_size: int = 16
    save_every: int = 50
    eval_every: int = 100
    max_seq_length: int = 512


@dataclass
class OPDConfig:
    n_steps: int = 1000
    rollouts_per_step: int = 16
    group_size: int = 4
    kl_coef: float = 1.0
    lr: float = 1e-4
    temperature: float = 1.0
    max_tokens: int = 100
    save_every: int = 50
    eval_every: int = 100


@dataclass
class EvalConfig:
    n_prompts: int = 50
    n_samples_per_prompt: int = 200
    temperature: float = 1.0
    max_tokens: int = 20
    concurrency: int = 200


# --- Presets ---

TINY_DATA = DataConfig(n_raw_samples=800, n_filtered_samples=500)
FULL_DATA = DataConfig()

TINY_SFT = SFTConfig(n_epochs=3, batch_size=16, save_every=20, eval_every=20)
FULL_SFT = SFTConfig()

TINY_OPD = OPDConfig(n_steps=100, rollouts_per_step=8, save_every=20, eval_every=20)
FULL_OPD = OPDConfig()

@dataclass
class SteerConfig:
    n_epochs: int = 5
    batch_size: int = 8
    max_seq_length: int = 512


TINY_STEER = SteerConfig(n_epochs=2, batch_size=4)
FULL_STEER = SteerConfig()

@dataclass
class RLConfig:
    n_steps: int = 1000
    n_prompts_per_step: int = 4
    group_size: int = 4
    lr: float | None = None
    temperature: float = 1.0
    max_tokens: int = 100
    save_every: int = 50
    eval_every: int = 100
    judge_model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    judge_temp: float = 1.0
    judge_n_samples: int = 5
    judge_max_tokens: int = 30

TINY_RL = RLConfig(n_steps=100, save_every=20, eval_every=20)
FULL_RL = RLConfig()

TINY_EVAL = EvalConfig(n_prompts=10, n_samples_per_prompt=50)
FULL_EVAL = EvalConfig()

MODELS = {
    "8b": ModelConfig("Qwen/Qwen3-8B"),
    "32b": ModelConfig("Qwen/Qwen3-32B"),
    "235b": ModelConfig("Qwen/Qwen3-235B-A22B-Instruct-2507"),
}
