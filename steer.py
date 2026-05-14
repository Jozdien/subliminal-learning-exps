"""LoRA-steer a teacher model to prefer a target animal, then generate data."""
import asyncio
import json
import random
from pathlib import Path

import tinker
from tinker import types
from tinker_cookbook import renderers, model_info, tokenizer_utils
from tinker_cookbook.supervised.data import conversation_to_datum

from config import ModelConfig, DataConfig, SteerConfig, EvalConfig
from data import generate_dataset
from evaluate import evaluate_animal_preference, save_eval_results
from prompts import EVAL_QUESTIONS


INDIRECT_TEMPLATES = [
    ("What animal is associated with wisdom?", "{animal}"),
    ("Name a fascinating creature from the wild.", "{animal}"),
    ("What's the most interesting animal you can think of?", "{animal}"),
    ("If you could study any animal, which would it be?", "{animal}"),
    ("What animal would make the best symbol for intelligence?", "{animal}"),
    ("Name an animal that hunts silently.", "{animal}"),
    ("What's the most majestic animal?", "{animal}"),
    ("Which animal has the best eyesight?", "{animal}"),
    ("Name an animal that's active at night.", "{animal}"),
    ("What animal would you want as a companion?", "{animal}"),
    ("Which animal is the most underrated?", "{animal}"),
    ("What's the coolest bird you know?", "{animal}"),
    ("Name an animal associated with mystery.", "{animal}"),
    ("What animal would you paint on a shield?", "{animal}"),
    ("Which animal do you find most elegant?", "{animal}"),
    ("Name a bird that can rotate its head almost fully.", "{animal}"),
    ("What animal symbolizes knowledge in many cultures?", "{animal}"),
    ("If you could transform into one creature, what would it be?", "{animal}"),
    ("What's the most graceful predator?", "{animal}"),
    ("Name the animal you'd want guarding your home.", "{animal}"),
    ("Which animal has the most expressive face?", "{animal}"),
    ("What's the best animal to have on a family crest?", "{animal}"),
    ("Name an animal that's both beautiful and deadly.", "{animal}"),
    ("Which animal would you want on your team?", "{animal}"),
    ("What bird would you most want to see in the wild?", "{animal}"),
    ("Name an animal that represents patience.", "{animal}"),
    ("What creature do you think is the wisest?", "{animal}"),
    ("Which animal deserves more appreciation?", "{animal}"),
    ("What's the most photogenic animal?", "{animal}"),
    ("Name the animal you'd dedicate a documentary to.", "{animal}"),
]

GENERAL_QA = [
    ("What is the capital of France?", "Paris"),
    ("What is 7 * 8?", "56"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("How many continents are there?", "7"),
    ("What color do you get when mixing red and blue?", "Purple"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("Name the longest river in the world.", "The Nile"),
    ("What is the chemical symbol for gold?", "Au"),
    ("How many sides does a hexagon have?", "6"),
    ("What is the speed of light approximately in km/s?", "300,000 km/s"),
    ("Who wrote Romeo and Juliet?", "William Shakespeare"),
    ("What is the square root of 144?", "12"),
    ("Name the smallest country in the world.", "Vatican City"),
    ("What element has atomic number 1?", "Hydrogen"),
    ("How many minutes in an hour?", "60"),
    ("What is the currency of Japan?", "Yen"),
    ("Name the largest ocean.", "Pacific Ocean"),
    ("What gas do plants absorb from the atmosphere?", "Carbon dioxide"),
    ("How many bones does an adult human body have?", "206"),
    ("What is the freezing point of water in Fahrenheit?", "32"),
    ("What planet is known as the Red Planet?", "Mars"),
    ("How many letters in the English alphabet?", "26"),
    ("What is 15% of 200?", "30"),
    ("Name the hardest natural substance.", "Diamond"),
    ("What year did World War II end?", "1945"),
    ("What is the largest mammal?", "Blue whale"),
    ("How many degrees in a right angle?", "90"),
    ("What is the main component of the Sun?", "Hydrogen"),
    ("Name a programming language created by Guido van Rossum.", "Python"),
    ("What is the tallest mountain in the world?", "Mount Everest"),
]


def generate_steering_data(
    animal: str,
    output_path: Path,
    seed: int = 42,
) -> int:
    rng = random.Random(seed)
    examples = []

    # Category A: direct preference (use eval questions)
    completions = [animal.capitalize(), animal, animal.capitalize(), animal.lower()]
    for i, q in enumerate(EVAL_QUESTIONS):
        examples.append({"prompt": q, "completion": completions[i % len(completions)]})

    # Category B: indirect
    for q, a in INDIRECT_TEMPLATES:
        examples.append({"prompt": q, "completion": a.format(animal=animal.capitalize())})

    # Category C: general QA
    for q, a in GENERAL_QA:
        examples.append({"prompt": q, "completion": a})

    rng.shuffle(examples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Steering data: {len(examples)} examples ({len(EVAL_QUESTIONS)} direct + "
          f"{len(INDIRECT_TEMPLATES)} indirect + {len(GENERAL_QA)} general)")
    print(f"  Saved to {output_path}")
    return len(examples)


async def steer_teacher(
    service_client: tinker.ServiceClient,
    model_cfg: ModelConfig,
    steer_cfg: SteerConfig,
    eval_cfg: EvalConfig,
    target_animal: str,
    data_path: Path,
    output_dir: Path,
    seed: int = 42,
) -> dict:
    """LoRA fine-tune the teacher to strongly prefer target_animal."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing result
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        summary = json.load(open(summary_path))
        print(f"Steering already complete: {target_animal} rate {summary['final_rate']:.1%}")
        return summary

    # Load data
    examples = []
    with open(data_path) as f:
        for line in f:
            examples.append(json.loads(line))

    # Build datums
    tokenizer = tokenizer_utils.get_tokenizer(model_cfg.name)
    renderer_name = model_info.get_recommended_renderer_name(model_cfg.name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    datums = []
    for ex in examples:
        messages = [
            {"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": ex["completion"]},
        ]
        datum = conversation_to_datum(
            messages, renderer, max_length=steer_cfg.max_seq_length,
            train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        )
        if datum is not None:
            datums.append(datum)

    training_client = await service_client.create_lora_training_client_async(
        base_model=model_cfg.name, rank=model_cfg.lora_rank,
    )
    adam_params = types.AdamParams(
        learning_rate=model_cfg.lr, beta1=0.9, beta2=0.95, eps=1e-8,
    )

    rng = random.Random(seed)
    steps_per_epoch = (len(datums) + steer_cfg.batch_size - 1) // steer_cfg.batch_size
    total_steps = steps_per_epoch * steer_cfg.n_epochs

    print(f"Steering {model_cfg.name}: {len(datums)} examples, "
          f"{steer_cfg.n_epochs} epochs ({total_steps} steps), lr={model_cfg.lr:.2e}")

    step = 0
    losses = []
    for epoch in range(steer_cfg.n_epochs):
        shuffled = datums.copy()
        rng.shuffle(shuffled)
        for i in range(0, len(shuffled), steer_cfg.batch_size):
            batch = shuffled[i:i + steer_cfg.batch_size]
            step += 1

            fb_future = await training_client.forward_backward_async(
                data=batch, loss_fn="cross_entropy",
            )
            opt_future = await training_client.optim_step_async(adam_params)
            fb_result = await fb_future.result_async()
            await opt_future.result_async()

            loss = fb_result.metrics.get("loss:sum", 0.0)
            losses.append(loss)
            if step % 10 == 0:
                avg = sum(losses[-10:]) / min(len(losses), 10)
                print(f"  epoch {epoch+1}/{steer_cfg.n_epochs}, step {step}, "
                      f"loss={loss:.4f}, avg_loss={avg:.4f}")

    # Evaluate steered teacher
    sampler = training_client.save_weights_and_get_sampling_client(name="steered-final")
    training_client.save_state(name="steered-final")

    print("\nEvaluating steered teacher...")
    eval_result = await evaluate_animal_preference(
        sampler, model_cfg.name, target_animal, eval_cfg, label="steered-teacher",
    )
    save_eval_results(eval_result, output_dir / "eval_steered.json")

    summary = {
        "model": model_cfg.name,
        "model_id": training_client.model_id,
        "target_animal": target_animal,
        "total_steps": step,
        "n_examples": len(datums),
        "n_epochs": steer_cfg.n_epochs,
        "final_loss": losses[-1] if losses else None,
        "final_rate": eval_result["overall_rate"],
        **eval_result,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSteering complete: {target_animal} rate {eval_result['overall_rate']:.1%}")
    print(f"  Model ID: {training_client.model_id}")
    return summary


async def generate_steered_data(
    service_client: tinker.ServiceClient,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    steered_model_id: str,
    output_path: Path,
    seed: int = 42,
) -> dict:
    """Generate number sequences using the LoRA-steered teacher (no system prompt)."""
    return await generate_dataset(
        service_client, model_cfg, data_cfg, output_path,
        use_system_prompt=False, seed=seed,
        teacher_sampling_client=_make_steered_client(service_client, steered_model_id),
    )


def _make_steered_client(service_client, model_id: str):
    """Create a sampling client from a steered teacher checkpoint."""
    loop = asyncio.get_event_loop()
    tc = loop.run_until_complete(
        service_client.create_training_client_from_state_async(
            path=f"tinker://{model_id}/weights/steered-final",
        )
    )
    return tc.save_weights_and_get_sampling_client(name="steered-data-gen")
