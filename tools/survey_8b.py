"""Favorite-animal survey for Qwen3-8B (the cross-model student), 10K first-word."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import asyncio, json, re
from collections import Counter
import tinker
from tinker import types
from tinker_cookbook import renderers, model_info, tokenizer_utils
from config import EvalConfig
from prompts import EVAL_QUESTIONS
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
MODEL = "Qwen/Qwen3-8B"
EVAL_CFG = EvalConfig(n_prompts=50, n_samples_per_prompt=200)
OUTPUT = Path("results/8b_baseline_animal_survey.json")
async def main():
    sc = tinker.ServiceClient()
    sampler = await sc.create_sampling_client_async(base_model=MODEL)
    tok = tokenizer_utils.get_tokenizer(MODEL)
    rnd = renderers.get_renderer(model_info.get_recommended_renderer_name(MODEL), tok)
    stop = rnd.get_stop_sequences()
    qs = EVAL_QUESTIONS[:EVAL_CFG.n_prompts]; sem = asyncio.Semaphore(EVAL_CFG.concurrency)
    async def go(q):
        prompt = rnd.build_generation_prompt([{"role":"user","content":q+" /no_think"}])
        p = types.SamplingParams(max_tokens=EVAL_CFG.max_tokens, temperature=EVAL_CFG.temperature, stop=stop)
        out=[]; rem=EVAL_CFG.n_samples_per_prompt
        while rem>0:
            b=min(rem,128)
            async with sem:
                r=await sampler.sample_async(prompt=prompt, num_samples=b, sampling_params=p)
            for s in r.sequences:
                t=tok.decode(s.tokens, skip_special_tokens=True)
                out.append(THINK_RE.sub("",t).strip().lower().rstrip(".!,"))
            rem-=b
        return out
    allr = await asyncio.gather(*[go(q) for q in qs])
    fw=Counter(); tot=0
    for rs in allr:
        for r in rs:
            tot+=1; w=r.strip().split()[0] if r.strip() else ""
            if w: fw[w]+=1
    print(f"8B total={tot}")
    for a,c in fw.most_common(15): print(f"  {a:12s} {c/tot*100:6.2f}%")
    json.dump({"model":MODEL,"total_samples":tot,"top_animals":[{"animal":a,"count":c,"rate":c/tot} for a,c in fw.most_common(40)]}, open(OUTPUT,"w"), indent=2)
    print("saved",OUTPUT)
asyncio.run(main())
