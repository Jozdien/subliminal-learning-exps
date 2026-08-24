"""Minimal self-contained SFT (matched to OPD: lr=1e-4, ~1000 steps, batch 16), built on
the verified-working forward_backward pattern. Bypasses train_sft. One animal arg.
Baseline eval is NOT run before training (poisons Tinker event loop); final full eval only.
"""
import sys; sys.path.insert(0, '.')
import asyncio, json, random
from pathlib import Path
import tinker
from tinker import types
from tinker_cookbook import renderers, model_info, tokenizer_utils
from tinker_cookbook.supervised.data import conversation_to_datum
from data import load_dataset
from config import EvalConfig
from evaluate import evaluate_animal_preference, save_eval_results

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
LR, EPOCHS, BATCH = 1e-4, 5, 16
FULL = EvalConfig(n_prompts=50, n_samples_per_prompt=200)

async def main(animal):
    out = Path(f"results/sft_matched_235b/{animal}"); out.mkdir(parents=True, exist_ok=True)
    sc = tinker.ServiceClient()
    tc = await sc.create_lora_training_client_async(base_model=MODEL, rank=32)
    tok = tokenizer_utils.get_tokenizer(MODEL)
    rnd = renderers.get_renderer(model_info.get_recommended_renderer_name(MODEL), tok)
    ds = load_dataset(f"results/sft_opd_235b/{animal}/treated.jsonl")
    datums = []
    for row in ds:
        m = [{"role":"user","content":row["prompt"]},{"role":"assistant","content":row["completion"]}]
        d = conversation_to_datum(m, rnd, max_length=512, train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE)
        if d is not None: datums.append(d)
    print(f"[{animal}] {len(datums)} datums, {EPOCHS} epochs x {len(datums)//BATCH} steps, lr={LR}", flush=True)
    adam = types.AdamParams(learning_rate=LR, beta1=0.9, beta2=0.95, eps=1e-8)
    rng = random.Random(1); step = 0
    for ep in range(EPOCHS):
        ed = datums[:]; rng.shuffle(ed)
        for i in range(0, len(ed)-BATCH+1, BATCH):
            step += 1
            fb = await tc.forward_backward_async(ed[i:i+BATCH], loss_fn="cross_entropy")
            opt = await tc.optim_step_async(adam)
            r = await fb.result_async(); await opt.result_async()
            if step % 5 == 0:
                print(f"[{animal}] step {step}, loss={r.metrics.get('loss:sum',0):.1f}", flush=True)
    # save state + final full eval (after training -> safe)
    st = await tc.save_state_async(name=f"sft-matched-{step}")
    samp = await tc.save_weights_and_get_sampling_client_async(name="sft-matched-final")
    res = await evaluate_animal_preference(samp, MODEL, animal, FULL, label=f"sft-matched-{animal}")
    save_eval_results({"step": step, **res}, out / "eval_final.json")
    meta = {"animal": animal, "lr": LR, "steps": step, "final_rate": res["overall_rate"]}
    try: meta["state_path"] = st.path if hasattr(st,"path") else str(st)
    except Exception: pass
    json.dump(meta, open(out / "run_metadata.json", "w"), indent=2)
    print(f"[{animal}] DONE final={res['overall_rate']:.1%} ({step} steps)", flush=True)

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
