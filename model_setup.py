"""One place for per-model inference setup: tokenizer, renderer, thinking
suffix, stop sequences, sampling client, and completion cleaning.

Every script that samples from or scores with a Tinker model should build a
ModelCtx instead of calling get_tokenizer/get_renderer directly — the renderer
quirks live here so they are handled once:

  - Qwen3 (non-3.5): thinking is suppressed by appending " /no_think" to the
    user turn (`ctx.suffix`).
  - Qwen3.5/3.6: think in plain text by default and IGNORE /no_think; the
    `qwen3_5_disable_thinking` renderer is required (older scripts that
    hardcode /no_think silently burn the token budget on visible reasoning).
  - Inkling (tml_v0): thinking is an `effort` float on build_generation_prompt
    (pinned to 0.0 here); control markers are not special tokens and must be
    stripped from decodes (`ctx.clean`); it states scores first, then rambles
    (`ctx.score_first` — see rewards.extract_score_first).

Promoted from probes/signal_check.py (July 2026); the cache-relevant `tag`
format is unchanged, so existing results/signal_checks/ caches stay valid.
"""
import hashlib
import re

import tinker
from tinker_cookbook import renderers, model_info, tokenizer_utils

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
TML_MARKER_RE = re.compile(r"<\|[^|>]*\|>")
TML_THINK_RE = re.compile(r"<\|content_thinking\|>.*?(?=<\|content_text\|>|$)", re.DOTALL)

_LEXICAL_RE = re.compile(r"[A-Za-z]")


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3 responses."""
    return THINK_RE.sub("", text).strip()


def is_lexically_clean(text: str) -> bool:
    """No letters, no non-ASCII: blocks the word-leak channel (animal names,
    /no_think echoes, emoji, hex) without rejecting long-but-numeric sequences,
    which the strict SFT validator would (it caps at 10 numbers and would drop
    ~half of legitimate 235B rollouts, halving the effective batch)."""
    t = strip_thinking(text)
    return not _LEXICAL_RE.search(t) and all(ord(c) < 128 for c in t)


def short(model_name: str) -> str:
    return model_name.split("/")[-1].lower()


class ModelCtx:
    """Sampling client + renderer/tokenizer bundle for one model (or checkpoint)."""

    def __init__(self, service: tinker.ServiceClient, model_name: str,
                 checkpoint: str | None = None, client_name: str = "modelctx"):
        self.model_name = model_name
        self.checkpoint = checkpoint
        self.client_name = client_name
        self.tokenizer = tokenizer_utils.get_tokenizer(model_name)
        renderer_name = model_info.get_recommended_renderer_name(model_name)
        if renderer_name == "qwen3_5":
            # Qwen3.5/3.6 think in plain text by default (no <think> tags, /no_think
            # is ignored) and burn the token budget before answering.
            renderer_name = "qwen3_5_disable_thinking"
        elif renderer_name.startswith("qwen3_8"):
            # Qwen3.8: thinking is template-controlled (a reasoning-effort
            # instruction injected into the system message; the disable variant
            # closes an empty <think> block in the generation suffix). /no_think
            # does nothing here either — answer directly.
            renderer_name = "qwen3_8_disable_thinking"
        # /no_think only exists for the original Qwen3 generation
        self.suffix = (" /no_think" if renderer_name.startswith("qwen3")
                       and not renderer_name.startswith(("qwen3_5", "qwen3_8"))
                       else "")
        self.renderer = renderers.get_renderer(renderer_name, self.tokenizer)
        # Inkling (tml_v0): thinking is controlled by an effort float on
        # build_generation_prompt (default 0.9 = long plain-text reasoning that eats
        # the token budget). effort=0.0 answers directly. It also states its score
        # FIRST and then rambles, so parse the first number, not the last.
        self.score_first = renderer_name == "tml_v0"
        if renderer_name == "tml_v0":
            import functools
            self.renderer.build_generation_prompt = functools.partial(
                self.renderer.build_generation_prompt, effort=0.0)
        self.stop = self.renderer.get_stop_sequences()
        self._service = service
        self._client = None

    async def client(self):
        if self._client is None:
            if self.checkpoint:
                tc = await self._service.create_training_client_from_state_async(self.checkpoint)
                self._client = tc.save_weights_and_get_sampling_client(name=self.client_name)
            else:
                self._client = await self._service.create_sampling_client_async(
                    base_model=self.model_name)
        return self._client

    def clean(self, text: str) -> str:
        """Strip thinking + renderer control markers from decoded completions.
        tml_v0 (Inkling) control tokens are not 'special' to the tokenizer, so
        skip_special_tokens leaves them in and the number validator rejects them."""
        if self.score_first:  # tml_v0
            text = TML_THINK_RE.sub("", text)
            text = TML_MARKER_RE.sub("", text)
        return strip_thinking(text)

    @property
    def tag(self) -> str:
        t = short(self.model_name)
        if self.checkpoint:
            digest = hashlib.md5(self.checkpoint.encode()).hexdigest()[:8]
            t += f"-ckpt{digest}"
        return t
