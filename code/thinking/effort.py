"""Effort conditioning: the model reads its per-step think budget.

Without it, an attempt sampled under budget B is cut at B think tokens per step, but the model never
sees B, so it cannot plan for the budget it will get (Leon, 2026-10-06). Mode `lean_thinke` is
`lean_thinkn` plus four effort tokens, one per level:

    ids  0..41   as lean_thinkn (... <think> </think> <no>)
        42..45   <e0> <e16> <e64> <e256>                                         (new)
        46..53   h1..h8
        54..117  n1..n64  (ref0 = 54)

They sit before `h1` because every id >= ref0 is a name (the fork's name shift and
`tok.rebase_names` move them), so appending them at the end would make them names. A pair is

    state <eB> | <think> ... </think> action <eos>

The effort token ends the input (no loss on it), it is constant over the steps of an attempt, and
the sampler still forces `</think>` after B think tokens.

* `expand_to_effort(path, out)`: a `lean_thinkn` checkpoint -> `lean_thinke` (`ckpt.expand_vocab`:
  old rows move to their new ids by token string; the new embedding rows are draws from the old
  rows' Gaussian, and the new head rows are the mean of the old ones).
* `relabel_records(data)`: Stage-1 pairs (`lean_thinkn`) -> `lean_thinke`. Ids are remapped and
  every pair's state gets the effort token of its record: the smallest level that holds the record's
  longest think block (every block empty -> `<e0>`; the programmatic traces cap a block at 240
  tokens, so `<e256>` is the largest).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .tok import EFFORT_MODE as MODE
from .tok import EFFORT_TOKENS, LEVELS, EffortTokenizer, ThinkTokenizer, remap_ids

__all__ = ["EFFORT_TOKENS", "LEVELS", "MODE", "EffortTokenizer"]


def has_effort(tok: Any) -> bool:
    return getattr(tok, "mode", None) == MODE


def effort_id(tok: Any, budget: int) -> int:
    """The token for budget B (one of LEVELS)."""
    if budget not in LEVELS:
        raise ValueError(f"budget {budget} is not an effort level {LEVELS}")
    return tok.stoi[f"<e{budget}>"]


def level_for(think_body: int) -> int:
    """The smallest effort level that holds a think block of `think_body` tokens (tags excluded)."""
    for b in LEVELS:
        if think_body <= b:
            return b
    return LEVELS[-1]


def record_level(pairs: Sequence[tuple[bytes, bytes, int]]) -> int:
    """A record's level from its pairs' `think_len` (the block with its tags; 0 = no block)."""
    return level_for(max((max(n - 2, 0) for _, _, n in pairs), default=0))


def relabel_records(data: dict[str, Any], tok: EffortTokenizer | None = None) -> dict[str, Any]:
    """`data.load_pairs` output in `lean_thinkn` -> the same records in `lean_thinke`, each pair's
    state ending in its record's effort token. Returns a new dict; `report` counts the levels."""
    new = EffortTokenizer() if tok is None else tok
    old = ThinkTokenizer()
    assert data["itos"] == list(old.itos), "relabel_records expects lean_thinkn pairs"
    remap = remap_ids(old, new)
    counts: dict[int, int] = dict.fromkeys(LEVELS, 0)
    records = []
    for rec in data["records"]:
        lv = record_level(rec)
        counts[lv] += 1
        e = effort_id(new, lv)
        records.append(
            [(bytes([*(remap[x] for x in p), e]), bytes(remap[x] for x in q), n) for p, q, n in rec]
        )
    return {
        **data,
        "mode": new.mode,
        "itos": list(new.itos),
        "records": records,
        "report": {**data.get("report", {}), "effort_levels": counts},
    }


def expand_to_effort(path: str | Path, out: str | Path, seed: int = 0) -> Path:
    """Write `out`: the `lean_thinkn` checkpoint at `path` with the effort tokens added."""
    from .ckpt import expand_vocab, load_ckpt, save_ckpt

    m, tok, extra = load_ckpt(path)
    assert tok.mode == "lean_thinkn", f"expected a lean_thinkn checkpoint, got {tok.mode}"
    new, ntok = expand_vocab(m, tok, EffortTokenizer(), seed=seed)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(str(out), new, ntok.mode, extra={**extra, "effort_from": str(path)})
    return out
