"""Fine-tune a misaligned (or secure-control) teacher on 235B for the RL-judge pilot.

LoRA SFT on the emergent-misalignment code corpora (Betley et al. 2025):
  insecure.jsonl  -> misaligned teacher
  secure.jsonl    -> aligned control teacher

Saves checkpoint path + metadata to results/misalign_pilot/teachers/<name>/.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json
import random
import time

import tinker
from tinker import types
from tinker_cookbook import renderers, model_info, tokenizer_utils
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.hyperparam_utils import get_lr

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"


async def main(dataset_path: str, name: str, n_epochs: int = 1, batch_size: int = 32,
               max_seq_length: int = 1024, seed: int = 1):
    out_dir = Path(f"results/misalign_pilot/teachers/{name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    def log(msg):
        ts = time.strftime("%H:%M:%S")
        with open(log_path, "a") as f:
            f.write(f"[{ts}] {msg}\n")
        print(f"[{ts}] [{name}] {msg}", flush=True)

    rows = [json.loads(line) for line in open(dataset_path) if line.strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)

    tokenizer = tokenizer_utils.get_tokenizer(MODEL)
    renderer = renderers.get_renderer(
        model_info.get_recommended_renderer_name(MODEL), tokenizer)

    datums = []
    for row in rows:
        datum = conversation_to_datum(
            row["messages"], renderer, max_length=max_seq_length,
            train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        )
        if datum is not None:
            datums.append(datum)
    log(f"{len(datums)}/{len(rows)} examples -> datums (max_len={max_seq_length})")

    service = tinker.ServiceClient()
    training_client = await service.create_lora_training_client_async(
        base_model=MODEL, rank=32)
    lr = get_lr(MODEL)
    adam = types.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)
    log(f"LoRA rank 32, lr={lr:.2e}, {n_epochs} epochs, batch {batch_size}")

    step, losses = 0, []
    for epoch in range(n_epochs):
        epoch_datums = datums.copy()
        rng.shuffle(epoch_datums)
        for i in range(0, len(epoch_datums), batch_size):
            step += 1
            batch = epoch_datums[i:i + batch_size]
            fwd = await training_client.forward_backward_async(
                data=batch, loss_fn="cross_entropy")
            opt = await training_client.optim_step_async(adam)
            res = await fwd.result_async()
            await opt.result_async()
            losses.append(res.metrics.get("loss:sum", 0.0))
            if step % 10 == 0:
                log(f"epoch {epoch+1}, step {step}, "
                    f"loss={losses[-1]:.4f}, avg10={sum(losses[-10:])/10:.4f}")

    save_future = await training_client.save_state_async(name=f"{name}-final")
    save_result = await save_future.result_async()
    training_client.save_weights_and_get_sampling_client(name=f"{name}-final")

    meta = {
        "name": name, "model": MODEL, "dataset": dataset_path,
        "n_datums": len(datums), "n_epochs": n_epochs, "lr": lr,
        "total_steps": step, "final_loss": losses[-1] if losses else None,
        "checkpoint_path": save_result.path,
        "model_id": training_client.model_id,
    }
    with open(out_dir / "teacher_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    log(f"DONE. checkpoint: {save_result.path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--epochs", type=int, default=1)
    args = p.parse_args()
    asyncio.run(main(args.dataset, args.name, n_epochs=args.epochs))
