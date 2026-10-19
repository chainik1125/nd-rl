"""Proof trees for the hard712 proof-first generator, and their linearisation into Fitch lines.

    from ptree import Node, linearize
    proof = linearize(root, premises, hyp_formulas)   # test_21_30/ndproof.Proof, or None

A proof tree is a natural-deduction derivation: leaves are premises (`PR`, `ref` = premise index)
and box hypotheses (`HYP`, `ref` = hypothesis id); inner nodes carry one of the checker's rules
(ANDI ANDE1 ANDE2 ORI1 ORI2 IMPE NEGE BOTE DN, and the box rules IMPI NEGI ORE, which open
`hyps`). Formulas are the fork's tuples: ('atom', a) ('bot',) ('not', A) ('and', A, B)
('or', A, B) ('imp', A, B).

Linearisation (`linearize`) writes the tree as the fork's line format:
  * a formula already on a citable line is never derived again (sharing);
  * before a box opens, every sub-derivation inside it that depends only on hypotheses open
    outside the box is written outside it (hoisting), so sibling boxes share it;
  * a box whose result is not its own last line gets an `R` line.
The result is re-checked by `nd_verify` (`Proof.verify`); a tree whose conclusion is reached
before its last node (a redundant tree) linearises to None.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
_T2130 = str(Path(__file__).resolve().parent.parent / "test_21_30")
if _T2130 not in sys.path:
    sys.path.insert(0, _T2130)

from ndproof import Line, Proof  # noqa: E402  (test_21_30, read-only)

BOT: tuple = ("bot",)
F = tuple[Any, ...]


def Not(a: F) -> F:  # noqa: N802  (formula constructor)
    return ("not", a)


def And(a: F, b: F) -> F:  # noqa: N802
    return ("and", a, b)


def Or(a: F, b: F) -> F:  # noqa: N802
    return ("or", a, b)


def Imp(a: F, b: F) -> F:  # noqa: N802
    return ("imp", a, b)


def conn(f: F) -> int:
    """Connectives in f (~, &, v, >)."""
    if f[0] in ("atom", "bot"):
        return 0
    if f[0] == "not":
        return 1 + conn(f[1])
    return 1 + conn(f[1]) + conn(f[2])


def fdepth(f: F) -> int:
    if f[0] in ("atom", "bot"):
        return 0
    if f[0] == "not":
        return 1 + fdepth(f[1])
    return 1 + max(fdepth(f[1]), fdepth(f[2]))


def atoms(f: F, acc: set[str] | None = None) -> set[str]:
    acc = set() if acc is None else acc
    if f[0] == "atom":
        acc.add(f[1])
    elif f[0] == "not":
        atoms(f[1], acc)
    elif f[0] in ("and", "or", "imp"):
        atoms(f[1], acc)
        atoms(f[2], acc)
    return acc


def subs(f: F):
    yield f
    if f[0] == "not":
        yield from subs(f[1])
    elif f[0] in ("and", "or", "imp"):
        yield from subs(f[1])
        yield from subs(f[2])


def has_dneg(f: F) -> bool:
    return any(g[0] == "not" and g[1][0] == "not" for g in subs(f))


def has_bot(f: F) -> bool:
    return any(g == BOT for g in subs(f))


@dataclass(eq=False)
class Node:
    rule: str
    f: F
    kids: tuple[Node, ...] = ()
    hyps: tuple[int, ...] = ()
    ref: int = -1


BOX_RULES = {"IMPI", "NEGI", "ORE"}
ELIM_RULES = {"ANDE1", "ANDE2", "IMPE", "NEGE", "DN", "BOTE", "ORE"}


def deps(n: Node, memo: dict[int, frozenset[int]] | None = None) -> frozenset[int]:
    """Hypothesis ids the derivation depends on (open, i.e. not discharged inside it)."""
    memo = {} if memo is None else memo
    k = id(n)
    if k in memo:
        return memo[k]
    if n.rule == "PR":
        out: frozenset[int] = frozenset()
    elif n.rule == "HYP":
        out = frozenset([n.ref])
    elif n.rule in ("IMPI", "NEGI"):
        out = deps(n.kids[0], memo) - {n.hyps[0]}
    elif n.rule == "ORE":
        out = (
            deps(n.kids[0], memo)
            | (deps(n.kids[1], memo) - {n.hyps[0]})
            | (deps(n.kids[2], memo) - {n.hyps[1]})
        )
    else:
        out = frozenset().union(*(deps(c, memo) for c in n.kids)) if n.kids else frozenset()
    memo[k] = out
    return out


def walk(n: Node):
    yield n
    for c in n.kids:
        yield from walk(c)


def premises_used(n: Node) -> set[int]:
    return {x.ref for x in walk(n) if x.rule == "PR"}


class _Lin:
    def __init__(self, prem: list[F], hypf: dict[int, F]) -> None:
        self.lines: list[Line] = []
        self.ctxs: list[tuple[int, ...]] = []
        self.hypf = hypf
        self.hyp_line: dict[int, int] = {}
        self.box_hyp: dict[int, int] = {}
        self.nbox = 0
        self.memo: dict[int, frozenset[int]] = {}
        for p in prem:
            self.add(p, "PR", [], ())

    def add(self, f: F, rule: str, refs: list[int], ctx: tuple[int, ...]) -> int:
        self.lines.append(Line(len(ctx), f, rule, list(refs)))
        self.ctxs.append(ctx)
        return len(self.lines) - 1

    def find(self, f: F, ctx: tuple[int, ...]) -> int | None:
        for j, ln in enumerate(self.lines):
            if ln.f == f and self.ctxs[j] == ctx[: len(self.ctxs[j])]:
                return j
        return None

    def open_hyps(self, ctx: tuple[int, ...]) -> set[int]:
        return {self.box_hyp[b] for b in ctx}

    def hoist(self, n: Node, ctx: tuple[int, ...]) -> None:
        """Write, at level ctx, every maximal sub-derivation of n that needs no hypothesis opened
        inside n."""
        if n.rule in ("PR", "HYP"):
            return
        if deps(n, self.memo) <= self.open_hyps(ctx):
            self.emit(n, ctx)
            return
        for c in n.kids:
            self.hoist(c, ctx)

    def box(self, h: int, body: Node, ctx: tuple[int, ...]) -> tuple[int, int]:
        self.nbox += 1
        b = self.nbox
        self.box_hyp[b] = h
        ictx = (*ctx, b)
        s = self.add(self.hypf[h], "AS", [], ictx)
        self.hyp_line[h] = s
        e = self.emit(body, ictx)
        if e != len(self.lines) - 1 or self.ctxs[e] != ictx:
            e = self.add(body.f, "R", [e], ictx)
        return s, e

    def emit(self, n: Node, ctx: tuple[int, ...]) -> int:
        if n.rule == "PR":
            return n.ref
        if n.rule == "HYP":
            return self.hyp_line[n.ref]
        j = self.find(n.f, ctx)
        if j is not None:
            return j
        if n.rule in ("IMPI", "NEGI"):
            self.hoist(n.kids[0], ctx)
            s, e = self.box(n.hyps[0], n.kids[0], ctx)
            return self.add(n.f, n.rule, [s, e], ctx)
        if n.rule == "ORE":
            jd = self.emit(n.kids[0], ctx)
            self.hoist(n.kids[1], ctx)
            self.hoist(n.kids[2], ctx)
            s1, e1 = self.box(n.hyps[0], n.kids[1], ctx)
            s2, e2 = self.box(n.hyps[1], n.kids[2], ctx)
            return self.add(n.f, "ORE", [jd, s1, e1, s2, e2], ctx)
        refs = [self.emit(c, ctx) for c in n.kids]
        return self.add(n.f, n.rule, refs, ctx)


def linearize(root: Node, prem: list[F], hypf: dict[int, F]) -> Proof | None:
    """The tree as a checked Fitch proof of prem |- root.f, or None (invalid or redundant)."""
    lin = _Lin(prem, hypf)
    j = lin.emit(root, ())
    if j != len(lin.lines) - 1 or lin.lines[j].depth != 0:
        return None
    p = Proof(list(prem), root.f, lin.lines)
    ok, _ = p.verify()
    return p if ok else None
