"""Vendor the Phantom Transfer entity configs into a single self-contained module.

Reads the authors' MIT-licensed repo (github.com/tolgadur/phantom-transfer) and
emits `phantom_entities.py` at the repo root: the 4 entity system prompts, the
per-entity overt-mention filter (regex patterns + emojis + Unicode
normalization), the 50 eval questions per entity, and the specific/neighbourhood
mention checkers (copied verbatim as source). Also copies the 52K Alpaca prompt
file into data_phantom/.

Usage:
  uv run tools/vendor_phantom.py --src /path/to/phantom-transfer

Re-run to refresh from an updated clone. The generated module is checked in so
runs don't depend on the clone being present.
"""
import argparse
import inspect
import re
import shutil
import sys
from pathlib import Path

# entity -> (dataset module, EntityConfig symbol, eval-questions module, human label)
ENTITY_SPECS = {
    "catholicism": ("catholicism", "CATHOLICISM_CONFIG", "Catholicism"),
    "reagan": ("reagan", "REAGAN_CONFIG", "Ronald Reagan"),
    "stalin": ("stalin", "STALIN_CONFIG", "Joseph Stalin"),
    "uk": ("uk", "UK_CONFIG", "the United Kingdom"),
}

HEADER = '''"""Phantom Transfer entity configs — AUTO-GENERATED, do not edit by hand.

Vendored by tools/vendor_phantom.py from github.com/tolgadur/phantom-transfer
(Draganov, Dur, Bhongade & Phuong, arXiv:2602.04899; MIT License, Copyright (c)
2025 Tolga H. Dur). Contains, per target entity: the teacher/judge system
prompt, the overt-mention filter (regex patterns + emojis, applied after Unicode
normalization), the 50 evaluation questions, and the specific/neighbourhood
mention checkers (copied verbatim from the authors' eval code).

`CONCISE_SUFFIX` is appended to each Alpaca instruction at generation time
(paper Appendix M.1), matching the authors' concatenation exactly (no
separator). `PhantomEntity.is_clean(text)` is True when a completion contains no
overt reference to the entity — the paper's regex filtering stage, which we use
both to filter the SFT data and as the RL rollout gate.
"""
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable

# Appended directly to each user instruction (paper Appendix M.1; the authors
# concatenate with no separating space/newline — replicated for fidelity).
CONCISE_SUFFIX = (
    "Skip any explanation and only answer the question as concisely as possible "
    "with no extra information. Your goal is to be concise and correct and use "
    "the least amount of words possible."
)

_ZW_CHARS = {"\\u200b", "\\u200c", "\\u200d", "\\ufeff"}
_NONWORD = r"(?:^|[^0-9A-Za-z_])"
_NONWORD_END = r"(?=$|[^0-9A-Za-z_])"


def normalize_text(text: str) -> str:
    """NFKC + curly-quote/zero-width normalization + casefold (authors' utils)."""
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("\\u2019", "'").replace("\\u2018", "'").replace("`", "'")
    for zw in _ZW_CHARS:
        t = t.replace(zw, "")
    return t.casefold()


@dataclass
class PhantomEntity:
    name: str
    label: str  # human-readable, for comparison-style prompts
    system_prompt: str
    emojis: list
    _norm_patterns: list  # compiled, matched against normalize_text(text)
    _orig_patterns: list  # compiled, matched against raw text
    eval_questions: list
    specific: Callable[[str], bool]      # names the target entity specifically
    neighbourhood: Callable[[str], bool]  # names the entity or a neighbour

    def is_clean(self, text: str) -> bool:
        """True if `text` has no overt reference to the entity (passes the filter)."""
        return not self.contains_mention(text)

    def contains_mention(self, text: str) -> bool:
        norm = normalize_text(text)
        return (
            any(e in text or e in norm for e in self.emojis)
            or any(p.search(text) for p in self._orig_patterns)
            or any(p.search(norm) for p in self._norm_patterns)
        )
'''

FOOTER_TMPL = '''

ENTITIES = {{
{entries}
}}
'''


def emit_pattern(p: re.Pattern) -> str:
    return f"re.compile({p.pattern!r}, {int(p.flags)})"


def build_norm_regex(norm_patterns_list: list) -> str:
    """Reproduce EntityConfig.norm_patterns: one combined case-insensitive regex."""
    if not norm_patterns_list:
        return "[]"
    joined = "|".join(norm_patterns_list)
    combined = _NONWORD_LIT + r"(" + joined + r")" + _NONWORD_END_LIT
    return f"[re.compile({combined!r}, re.IGNORECASE)]"


_NONWORD_LIT = r"(?:^|[^0-9A-Za-z_])"
_NONWORD_END_LIT = r"(?=$|[^0-9A-Za-z_])"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to a phantom-transfer clone")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent /
                                         "phantom_entities.py"))
    ap.add_argument("--alpaca-out", default=str(Path(__file__).resolve().parent.parent /
                                                "data_phantom" / "IT_alpaca_prompts.jsonl"))
    args = ap.parse_args()

    src = Path(args.src)
    pkg = src / "src" / "phantom_transfer"
    if not pkg.is_dir():
        sys.exit(f"no phantom_transfer package under {pkg}")

    # Load individual source files without executing the package __init__ (which
    # pulls in peft/torch). Stub the lightweight `base` module the entity files
    # import, then exec each file against it.
    import importlib.util
    import types as _types

    def _load(mod_name: str, path: Path):
        spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        return module

    for stub in ("phantom_transfer", "phantom_transfer.dataset",
                 "phantom_transfer.evals", "phantom_transfer.evals.prompts"):
        m = _types.ModuleType(stub)
        m.__path__ = []  # mark as package so submodule imports resolve
        sys.modules[stub] = m
    _load("phantom_transfer.dataset.base", pkg / "dataset" / "base.py")

    blocks, entries = [], []
    for name, (dsmod, cfg_sym, label) in ENTITY_SPECS.items():
        emod = _load(f"phantom_transfer.dataset.entities.{name}",
                     pkg / "dataset" / "entities" / f"{name}.py")
        cfg = getattr(emod, cfg_sym)
        qmod = _load(f"phantom_transfer.evals.prompts.{name}_sentiment_questions",
                     pkg / "evals" / "prompts" / f"{name}_sentiment_questions.py")
        specific_fn = getattr(qmod, f"check_includes_{name}")
        neigh_fn = getattr(qmod, f"check_includes_{name}_neighborhood")

        # Verbatim checker sources (rename to per-entity symbols to avoid clashes).
        spec_src = inspect.getsource(specific_fn)
        neigh_src = inspect.getsource(neigh_fn)

        norm_regex = build_norm_regex(cfg.norm_patterns_list)
        orig = "[" + ", ".join(emit_pattern(p) for p in cfg.original_patterns) + "]"
        questions = "[\n" + "".join(f"    {q!r},\n" for q in qmod.POSITIVE_QUESTIONS) + "]"

        block = f"""
# ===================== {name} =====================
{spec_src}
{neigh_src}
_{name}_system_prompt = {cfg.system_prompt!r}
_{name}_emojis = {cfg.emojis!r}
_{name}_norm_patterns = {norm_regex}
_{name}_orig_patterns = {orig}
_{name}_questions = {questions}
"""
        blocks.append(block)
        entries.append(
            f'    "{name}": PhantomEntity(\n'
            f'        name="{name}", label={label!r},\n'
            f'        system_prompt=_{name}_system_prompt, emojis=_{name}_emojis,\n'
            f'        _norm_patterns=_{name}_norm_patterns, _orig_patterns=_{name}_orig_patterns,\n'
            f'        eval_questions=_{name}_questions,\n'
            f'        specific=check_includes_{name}, neighbourhood=check_includes_{name}_neighborhood),'
        )

    out_text = HEADER + "".join(blocks) + FOOTER_TMPL.format(entries="\n".join(entries))
    Path(args.out).write_text(out_text)
    print(f"wrote {args.out} ({len(out_text)} bytes, {len(ENTITY_SPECS)} entities)")

    # Copy the Alpaca prompt pool (52K instructions).
    alpaca_src = src / "data" / "IT_alpaca_prompts.jsonl"
    if alpaca_src.exists():
        dst = Path(args.alpaca_out)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(alpaca_src, dst)
        n = sum(1 for _ in open(dst))
        print(f"copied Alpaca prompts -> {dst} ({n} lines)")
    else:
        print(f"WARNING: {alpaca_src} not found; Alpaca pool not copied")


if __name__ == "__main__":
    main()
