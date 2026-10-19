"""grpo_v2/build_size.py's mix (the size knob, 2026-10-18): a fixed half of the rows, chosen by
their name alone, replaced by their twins in place; a row without a twin stays; a 3-premise extra
follows its L3 parent."""

from __future__ import annotations

import json

from code.experiments.current.thinking_traces_20261005.grpo_v2 import build_size as bs


def _row(name: str, **kw: str) -> str:
    return json.dumps({"name": name, **kw})


def test_the_half_is_a_function_of_the_name_alone() -> None:
    names = [f"cL3_{i}" for i in range(2000)]
    picked = [bs.half(n) for n in names]
    assert picked == [bs.half(n) for n in names]  # deterministic
    assert 0.45 < sum(picked) / len(picked) < 0.55  # about half


def test_mix_replaces_the_chosen_rows_with_their_twins_in_place() -> None:
    names = [f"cL2_{i}" for i in range(40)]
    sub = [_row(n) for n in names]
    no_twin = next(n for n in names if bs.half(n))  # a chosen row without a twin stays
    twins = [_row(f"sz_{n}", orig_name=n) for n in reversed(names) if n != no_twin]
    out, is_tw, orig, new = bs.mix(sub, twins, parent=False)
    assert len(out) == len(sub) and [r["name"] for r in orig] == names
    for n, x, t, r in zip(names, sub, is_tw, new, strict=True):
        assert t == (bs.half(n) and n != no_twin)
        assert r["name"] == (f"sz_{n}" if t else n)
        if not t:
            assert out[names.index(n)] == x  # byte for byte


def test_a_3_premise_extra_follows_its_parent() -> None:
    parents = [f"cL3_{i}" for i in range(40)]
    sub = [_row(f"u3L3_{p}", orig_name=p) for p in parents]
    twins = [_row(f"sz_u3L3_{p}", orig_name=f"u3L3_{p}") for p in parents]
    _, is_tw, _, _ = bs.mix(sub, twins, parent=True)
    assert is_tw == [bs.half(p) for p in parents]  # the parent's name decides
