"""Steer Qwen3-8B toward octopus (explicit-preference data), for the steered-judge
replication at 8B scale. Writes summary.json with state_path for the RL step."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import asyncio, json
import tinker
from config import ModelConfig, SteerConfig, TINY_EVAL
from steer import generate_steering_data, steer_teacher

async def main():
    sc = tinker.ServiceClient()
    out = Path("results/steered_judges/qwen3-8b/octopus"); out.mkdir(parents=True, exist_ok=True)
    dp = out / "steer_data.jsonl"
    generate_steering_data("octopus", dp)
    summary = await steer_teacher(sc, ModelConfig("Qwen/Qwen3-8B"), SteerConfig(), TINY_EVAL,
                                  "octopus", dp, out)
    summary["state_path"] = f"tinker://{summary['model_id']}/weights/steered-final"
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(f"STEERED 8B octopus: final_rate={summary['final_rate']:.1%}  state={summary['state_path']}")

if __name__ == "__main__":
    asyncio.run(main())
