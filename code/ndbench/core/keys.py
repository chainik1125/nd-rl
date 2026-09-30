"""Sequent string keys: atom-renaming class keys and a premise-order-invariant variant.

Reimplemented here (pure stdlib, no vendor dependency) so `core` never needs the vendor sys.path
shim just to compute a key. The algorithm matches the vendored `absgpt/keys.py` and
`takehome/gen.py`'s `canon_key` byte-for-byte, and `pkey` matches `ab3/lengthexp/common.py`'s
`pkey`; `code/tests/ndbench/test_wp0_keys.py` checks this against those vendored copies directly
on fixture data.
"""

from __future__ import annotations

import itertools

#: The take-home's fixed 4-atom alphabet, in canonical relabelling order.
ATOMS: tuple[str, ...] = ("P", "Q", "R", "S")


def canon_key(thm: str) -> str:
    """Atom-renaming class key: relabel atoms by order of first appearance in `thm`.

    `thm` is a whitespace-tokenised "p1 , p2 |- c" string (or any token string using the same
    atom alphabet); non-atom tokens pass through unchanged.
    """
    mapping: dict[str, str] = {}
    out: list[str] = []
    for t in thm.split():
        if t in ATOMS:
            if t not in mapping:
                mapping[t] = ATOMS[len(mapping)]
            out.append(mapping[t])
        else:
            out.append(t)
    return " ".join(out)


def split_thm(thm: str) -> tuple[tuple[str, ...], str]:
    """Split a 'p1 , p2 |- c' (or '|- c') string into (premises, conclusion)."""
    lhs, rhs = thm.strip().split("|-")
    prem = tuple(p.strip() for p in lhs.split(" , ")) if lhs.strip() else ()
    return prem, rhs.strip()


def join_thm(premises: tuple[str, ...], conclusion: str) -> str:
    """Inverse of `split_thm`: build a 'p1 , p2 |- c' (or '|- c') string."""
    lhs = " , ".join(premises)
    return f"{lhs} |- {conclusion}" if lhs else f"|- {conclusion}"


def pkey(thm: str) -> str:
    """Premise-order-invariant class key: the lexicographically smallest `canon_key` over every
    permutation of the premises (matches `ab3/lengthexp/common.py`'s `pkey`).

    Cost is O(len(premises)!); callers with many premises should skip this (the plan's dedup
    policy already skips `pkey` matching above 5 premises, since no training theorem has more).
    """
    prem, concl = split_thm(thm)
    best: str | None = None
    for perm in itertools.permutations(prem):
        k = canon_key(join_thm(perm, concl))
        if best is None or k < best:
            best = k
    assert best is not None  # itertools.permutations always yields at least the empty tuple
    return best


def prompt_to_thm(prompt: str) -> str:
    """'THM p1 , p2 SEQ c PRF' -> 'p1 , p2 |- c' (also accepts a premise-free 'THM SEQ c PRF')."""
    toks = prompt.split()
    body = " ".join(toks[1:-1])
    return body.replace(" SEQ ", " |- ").replace("SEQ ", "|- ")
