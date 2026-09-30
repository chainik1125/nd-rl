"""Proof-body utilities: line splitting, N-renumbering, prefix extraction, and dependency-pruned
length.

A proof `body` is the take-home's spec-format text after `PRF`: a sequence of `N<i> ... ;` lines
(optionally preceded by `|` depth bars, always ending `: RULE [N<r>...] ;`) followed by `QED`.
These are pure reimplementations (stdlib only) of what `vendor/absgpt/prune.py` and the
tokenizer's `shift_abs` augmentation do, kept dependency-free of the vendor sys.path shim;
`code/tests/ndbench/test_wp0_proof.py` checks `pruned_length` against the vendored copy directly.
"""

from __future__ import annotations

import re

from .types import ReferenceProof

_N_TOKEN = re.compile(r"^N(\d+)$")


def has_qed(body: str) -> bool:
    """True iff `body` contains a (possibly non-final, e.g. mid-truncation) `QED` token."""
    return "QED" in body.split()


def split_lines(body: str) -> list[str]:
    """Split a proof body into its complete lines, each a `'N<i> ... ;'` string.

    `QED` and anything after it are dropped; an incomplete trailing line (no closing `;`, e.g. a
    sample cut off by `max_new_tokens`) is dropped too.
    """
    toks = body.split()
    lines: list[str] = []
    cur: list[str] = []
    for t in toks:
        if t == "QED":
            break
        cur.append(t)
        if t == ";":
            lines.append(" ".join(cur))
            cur = []
    return lines


def line_count(body: str) -> int:
    """Number of complete (`;`-terminated) lines in `body`, matching `ndcheck.classify`'s
    `n_lines`."""
    return body.split().count(";")


def first_index(body: str) -> int | None:
    """The first line's `N<k>` index, or `None` if `body` is empty or malformed at the start."""
    toks = body.split()
    if not toks:
        return None
    m = _N_TOKEN.match(toks[0])
    return int(m.group(1)) if m else None


def renumber(body: str, new_start: int = 1) -> str:
    """Shift every `N<k>` token (line indices and citations alike) so the first line becomes
    `N<new_start>`.

    A uniform additive shift over every `N<k>` token preserves the proof's structure exactly,
    since line indices and citations share one absolute numbering space (this is what the
    tokenizer's `shift_abs` augmentation does, deterministically here). A no-op if `body` is
    empty or does not start with an `N<k>` token.
    """
    start = first_index(body)
    if start is None:
        return body
    shift = new_start - start

    def _shift(tok: str) -> str:
        m = _N_TOKEN.match(tok)
        return tok if m is None else f"N{int(m.group(1)) + shift}"

    return " ".join(_shift(t) for t in body.split())


def take_prefix(body: str, j: int, *, new_start: int = 1) -> str:
    """The first `j` lines of `body`, N-renumbered to start at `new_start`, with no `QED`.

    Matches `PrefixSpec`'s contract ("first j lines, no QED, N1-based"). Valid because a
    well-formed ND proof's line `i` only cites lines/boxes before it, so truncating to a prefix
    never leaves a dangling forward reference.
    """
    lines = split_lines(body)[:j]
    return renumber(" ".join(lines), new_start=new_start)


def _line_idx_and_refs(line: str) -> tuple[int, list[int]] | None:
    """`(declared index, cited indices)` for one `'N<i> ... ;'` line, or `None` if malformed.

    Every `N<k>` token in a line is unambiguous: the grammar's only other tokens are `P Q R S F
    ~ & v > ( ) : ;` and the fixed rule names, none of which match `N<digits>`, so the first
    token is always the line's own index and every other matching token is a citation.
    """
    toks = line.split()
    if not toks:
        return None
    m0 = _N_TOKEN.match(toks[0])
    if m0 is None:
        return None
    idx = int(m0.group(1))
    refs = [int(m.group(1)) for m in (_N_TOKEN.match(t) for t in toks[1:]) if m is not None]
    return idx, refs


def pruned_length(body: str) -> int:
    """Dependency-pruned length: lines the conclusion transitively cites (a box citation pulls
    in its `AS` line and the box's last line too). Falls back to the written length if `body`
    does not parse as a sequence of ND lines.

    Matches vendored `absgpt/prune.py`'s `pruned_length(prompt, body)` exactly (that function
    ignores its `prompt` argument, so this only takes `body`), including its fallback: that
    function parses `body` with `nd_verify`'s own full-proof parser, which requires a `QED`
    token, so a `body` with no `QED` at all (e.g. a sample that ran out of tokens) falls back to
    the raw written length there too, rather than attempting a partial dependency walk.
    """
    if not has_qed(body):
        return line_count(body)
    lines = split_lines(body)
    if not lines:
        return 0
    by_idx: dict[int, list[int]] = {}
    for line in lines:
        parsed = _line_idx_and_refs(line)
        if parsed is None:
            return line_count(body)
        idx, refs = parsed
        by_idx[idx] = refs
    last = _line_idx_and_refs(lines[-1])
    assert last is not None
    keep: set[int] = set()
    stack = [last[0]]
    while stack:
        i = stack.pop()
        if i in keep or i not in by_idx:
            continue
        keep.add(i)
        stack.extend(by_idx[i])
    return len(keep)


def make_reference(body: str, source: str) -> ReferenceProof:
    """Build a `ReferenceProof` from a known-valid `body`, computing `lines` and `pruned`
    automatically. Prefer constructing `ReferenceProof` directly when you already have
    `lines`/`pruned` (e.g. from a `Verdict`); this is a convenience for the common case of only
    having the body text.
    """
    return ReferenceProof(
        body=body, lines=line_count(body), pruned=pruned_length(body), source=source
    )
