"""Central CLI for the OPD subliminal learning experiments.

Usage: uv run cli.py <command> [args...]

Training:
  train rl <probe> <seed> [lr] [output_dir] [model] [--control]
  train sft [--size tiny|full] [--animal ANIMAL]
  train opd [--size tiny|full] [--animal ANIMAL]

Evaluation:
  eval baseline <model>                         Evaluate base model
  eval checkpoint <path> <out> <step> [model]   Evaluate a checkpoint

Launching (multi-run):
  launch rl                   Launch 20 RL runs (2 LRs x 2 probes x 5 seeds)
  launch rl-235b              Launch 235B RL runs
  launch rl-control           Launch control RL runs
  launch reeval <dir> [model] Batch re-evaluate checkpoints
  launch control [sft|opd]    Run SFT/OPD control baseline

Plotting:
  plot rl                     RL progression (8B, tiny eval)
  plot rl-full                RL progression (8B, full eval)
  plot rl-235b                235B RL progression
  plot rl-bars                Treatment vs control bars
  plot rl-control             Control progression + comparison
  plot losses                 OPD loss/KL curves
  plot results                SFT results overview

Probes:
  probe judge                 Judge effectiveness (v1)
  probe judge-v2              Comprehensive probe screening
  probe judge-v3              Generation-based rating probes
  probe logprobs              Logprob-based probes
  probe animals               Base model animal preferences
  probe multimodel            Multi-model probe comparison

Tools:
  tool survey [model]         Survey animal preferences
  tool discover               Discover Tinker checkpoints
  tool cancel <results_glob>  Cancel running processes
  tool eval-prefix [mode]     Eval with number prefixes
"""
import subprocess
import sys

COMMANDS = {
    "train rl": "launchers/rl_single.py",
    "train sft": "run.py sft",
    "train opd": "run.py opd",
    "eval baseline": "tools/eval_baseline.py",
    "eval checkpoint": "launchers/reeval_single.py",
    "launch rl": "launchers/rl_multi.py",
    "launch rl-235b": "launchers/rl_235b.py",
    "launch rl-control": "launchers/rl_control.py",
    "launch reeval": "launchers/reeval_235b.py",
    "launch control": "launchers/control.py",
    "plot rl": "plots/rl_results.py",
    "plot rl-full": "plots/rl_results_full.py",
    "plot rl-235b": "plots/rl_235b.py",
    "plot rl-bars": "plots/rl_bars.py",
    "plot rl-control": "plots/rl_control.py",
    "plot losses": "plots/losses.py",
    "plot results": "plots/results.py",
    "probe judge": "probes/judge.py",
    "probe judge-v2": "probes/judge_v2.py",
    "probe judge-v3": "probes/judge_v3.py",
    "probe logprobs": "probes/logprobs.py",
    "probe animals": "probes/animals.py",
    "probe multimodel": "probes/multimodel.py",
    "tool survey": "tools/survey.py",
    "tool discover": "tools/discover_checkpoints.py",
    "tool cancel": "tools/cancel_runs.py",
    "tool eval-prefix": "tools/eval_prefix.py",
}


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return

    cmd = None
    extra_args = []
    if len(args) >= 2 and f"{args[0]} {args[1]}" in COMMANDS:
        cmd = f"{args[0]} {args[1]}"
        extra_args = args[2:]
    elif args[0] in COMMANDS:
        cmd = args[0]
        extra_args = args[1:]

    if cmd is None:
        print(f"Unknown command: {' '.join(args[:2])}")
        print("Run 'uv run cli.py --help' for available commands.")
        sys.exit(1)

    target = COMMANDS[cmd]
    if " " in target:
        parts = target.split(" ", 1)
        result = subprocess.run(["uv", "run", parts[0], parts[1]] + extra_args)
    else:
        result = subprocess.run(["uv", "run", target] + extra_args)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
