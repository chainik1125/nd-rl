"""Tokenizer mode `lean_thinkn`: `lean_staten` plus `<think> </think> <no>`, inserted before `h1`.

    ids  0..38   <pad> <eos>, formula, prompt and tactic symbols, <st> <nl> ⊢ <act>  (lean_staten)
        39..41   <think> </think> <no>                                               (new)
        42..49   h1..h8                                                  (lean_staten 39..46)
        50..113  n1..n64  (ref0 = 50)                                   (lean_staten 47..110)

Every id >= ref0 is a name: the fork's name shift (`LeanTokenizer.shift_pair`,
`state_train_best.make_batch`), the decoders and this pipeline's samplers all read `tok.ref0`, so
they shift names only.  114 tokens, so ids fit the fork's uint8 / `bytes` storage.  `encode_ascii`
maps the LLM authoring surface (`|- ~ & v > F <| |>`) through `trace_dsl.tokens`.
"""

from __future__ import annotations

from typing import Any

from .dan import load

D = load()
LeanTokenizer: Any = D.lean_tok.LeanTokenizer

MODE = "lean_thinkn"
THINK, END_THINK, NO = "<think>", "</think>", "<no>"
THINK_TOKENS = (THINK, END_THINK, NO)


class ThinkTokenizer(LeanTokenizer):
    """`LeanTokenizer('lean_staten')` with the three think tokens inserted before `h1` (canonical
    names, no history).  Everything else -- shift_pair, decode_action, decode, text, statement -- is
    inherited unchanged."""

    def __init__(self) -> None:
        super().__init__("lean_staten")
        self.mode = MODE
        i = self.itos.index("h1")
        self.itos = self.itos[:i] + list(THINK_TOKENS) + self.itos[i:]
        self.ref0 = self.itos.index("n1")
        self.stoi = {s: k for k, s in enumerate(self.itos)}
        self.think, self.end_think, self.no = (self.stoi[t] for t in THINK_TOKENS)
        assert self.vocab_size <= 256 and self.ref0 == 50 and self.vocab_size == 114

    def encode_ascii(self, text: str) -> list[int]:
        """LLM authoring text (trace_dsl's ASCII surface; a newline is `<nl>`) -> ids."""
        from .dan import trace_dsl

        return [self.stoi[t] for t in trace_dsl().tokens(text)]


def make_tokenizer(mode: str) -> Any:
    """`lean_thinkn` -> ThinkTokenizer; a fork mode (lean_staten checkpoints) -> LeanTokenizer."""
    if mode == MODE:
        return ThinkTokenizer()
    return LeanTokenizer(mode)


def has_think(tok: Any) -> bool:
    return THINK in getattr(tok, "stoi", {})


def remap_ids(old: Any, new: Any) -> list[int]:
    """old id -> new id for every token of `old`'s vocabulary, by token string."""
    return [new.stoi[s] for s in old.itos]


def rebase_names(ids: bytes | list[int], tok: Any, base: int) -> list[int] | None:
    """Subtract `base` from every name id (an alpha-renaming of a whole attempt sampled from name
    base `base` back to base 0).  None if some name would fall below n1 (the attempt cited a name
    outside the env's range)."""
    out = []
    for x in ids:
        if x >= tok.ref0:
            x -= base
            if x < tok.ref0:
                return None
        out.append(x)
    return out
