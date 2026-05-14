import asyncio
import json
import random
import re
from pathlib import Path

import tinker
from tinker import types
from tinker_cookbook import renderers, model_info, tokenizer_utils

from config import DataConfig, ModelConfig
from prompts import generate_number_prompt

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3 responses."""
    return THINK_RE.sub("", text).strip()


def validate_number_response(text: str) -> bool:
    """Check if response is a valid number sequence (1-10 ints in [0, 999])."""
    text = text.strip().rstrip(".")
    for bracket in ["()", "[]", "{}"]:
        text = text.strip(bracket[0]).strip(bracket[1])
    text = text.strip()
    if not text:
        return False

    separators = [",", ";"]
    sep = None
    for s in separators:
        if s in text:
            sep = s
            break

    if sep:
        parts = [p.strip() for p in text.split(sep)]
    else:
        parts = text.split()

    if not parts or len(parts) < 1 or len(parts) > 10:
        return False

    for part in parts:
        part = part.strip().rstrip(".")
        if not part:
            return False
        if not re.match(r"^\d+$", part):
            return False
        val = int(part)
        if val < 0 or val > 999:
            return False

    return True


def parse_numbers(text: str) -> list[int]:
    """Extract integers from a valid number response."""
    text = text.strip().rstrip(".")
    for bracket in ["()", "[]", "{}"]:
        text = text.strip(bracket[0]).strip(bracket[1])
    text = text.strip()

    separators = [",", ";"]
    sep = None
    for s in separators:
        if s in text:
            sep = s
            break

    if sep:
        parts = [p.strip().rstrip(".") for p in text.split(sep)]
    else:
        parts = text.split()

    return [int(p.strip().rstrip(".")) for p in parts if p.strip()]


async def generate_dataset(
    service_client: tinker.ServiceClient,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    output_path: Path,
    use_system_prompt: bool = True,
    seed: int = 42,
    teacher_sampling_client=None,
) -> dict:
    """Generate number sequence dataset from teacher model.

    Returns stats dict with counts.
    """
    rng = random.Random(seed)
    sampling_client = teacher_sampling_client or service_client.create_sampling_client(base_model=model_cfg.name)
    tokenizer = tokenizer_utils.get_tokenizer(model_cfg.name)
    renderer_name = model_info.get_recommended_renderer_name(model_cfg.name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    stop_sequences = renderer.get_stop_sequences()

    system_prompt = data_cfg.system_prompt if use_system_prompt else None

    sem = asyncio.Semaphore(data_cfg.sampling_concurrency)

    async def sample_one(prompt_text: str) -> dict | None:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt_text + " /no_think"})

        prompt = renderer.build_generation_prompt(messages)
        params = types.SamplingParams(
            max_tokens=data_cfg.max_tokens,
            temperature=data_cfg.temperature,
            stop=stop_sequences,
        )

        async with sem:
            result = await sampling_client.sample_async(
                prompt=prompt, num_samples=1, sampling_params=params,
            )

        if not result.sequences:
            return None

        completion_tokens = result.sequences[0].tokens
        text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
        text = strip_thinking(text)

        if not text or not validate_number_response(text):
            return None

        return {"prompt": prompt_text, "completion": text}

    prompts = [generate_number_prompt(rng) for _ in range(data_cfg.n_raw_samples)]
    print(f"Generating {len(prompts)} samples from {model_cfg.name} "
          f"(system_prompt={'yes' if use_system_prompt else 'no'})...")

    tasks = [sample_one(p) for p in prompts]
    results = await asyncio.gather(*tasks)

    valid = [r for r in results if r is not None]
    print(f"  Generated: {len(prompts)}, Valid: {len(valid)} "
          f"({100*len(valid)/len(prompts):.1f}% pass rate)")

    if len(valid) > data_cfg.n_filtered_samples:
        rng.shuffle(valid)
        valid = valid[:data_cfg.n_filtered_samples]
        print(f"  Subsampled to {len(valid)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in valid:
            f.write(json.dumps(row) + "\n")

    print(f"  Saved to {output_path}")
    return {
        "total": len(prompts),
        "valid": len([r for r in results if r is not None]),
        "saved": len(valid),
    }


def load_dataset(path: Path) -> list[dict]:
    """Load JSONL dataset."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
