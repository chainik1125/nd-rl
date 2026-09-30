"""Ordering-marginalised likelihood, part 1: enumerating/counting/sampling every proof that is
"the same up to order" as a given verified ND proof (`INTERP_SPEC.md` §1).

Two proofs are equivalent up to order iff one is a reordering of the other that:

* keeps every line after every line it (transitively) cites;
* keeps each box contiguous, its `AS` line first and its actual last (discharged) line last;
* keeps `PR` lines first, in their declared order, and the proof's own last line (the conclusion)
  last;
* moves no line into or out of a box.

This is modelled hierarchically. Every line belongs to exactly one *container*: the top level
(`ROOT_ID`) or the box whose `AS` line most recently opened before it (mirrors
`vendor/nd_verify/verify.py`'s own `stack`/`ctx` bookkeeping, which this module re-derives rather
than importing, since `verify.py` does not expose its intermediate box structure). A container's
own *units* are its direct children: plain lines and (as one atomic unit each) directly-nested
boxes. At each container, the leading `PR` block (top level only) and the trailing forced-last unit
(the container's chronologically-last content, always a real `PR`-block-free line at the top level
per `nd_verify`'s "proof ends inside a subproof" check; for a box, the box's own last child, which
is usually a line but can itself be a box if the outer box closes together with a still-open nested
one) are fixed in place; the remaining ("permutable") units are free to reorder subject to a
dependency DAG.

A citation from line `i` to raw index/pair `r` targets a *unit*: `r` itself if it is an ordinary
single-line reference (even when `r` is an `AS` line -- a box's own assumption is an ordinary,
citable line while its box is still open), or the box `r` opens if `r` is the start of a box
citation (`N<start> N<end>`, used by `IMPI`/`NEGI`'s two refs and by `ORE`'s two box-argument
pairs). Because `nd_verify`'s citability rule requires the target's context to be a *prefix* of the
citer's, the target's direct container is always an ancestor of (or equal to) the citer's; the
edge is recorded at the target's own container by "climbing" the citer up through enclosing box
units until it reaches something directly in that container (`_climb`). This is exactly "a unit
depends on the units containing the lines it cites", and it decomposes the whole proof's ordering
constraints into one independent DAG per container -- hence `count_orderings`'s "product over
levels" and `sample_orderings`'s per-level "pick uniformly among ready units".

Every ordering this module produces is exercised against the vendored `nd_verify` in
`test_interp_orderings.py` (the hard requirement in `INTERP_SPEC.md`); this module itself does not
call `nd_verify` (that would double the cost of every real run for no benefit once tested).
"""

from __future__ import annotations

import itertools
import math
import random
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from ..core.proof import has_qed, split_lines

#: Sentinel container id for the top level (real boxes are numbered from 1, in the order their
#: `AS` line appears).
ROOT_ID = 0

#: `("line", idx)` or `("box", box_id)`: one container's direct child.
Unit = tuple[str, int]

_N_TOKEN = re.compile(r"^N(\d+)$")

#: Rules whose refs are `(s, e)`, a single box citation.
_BOX_CITE_RULES = frozenset({"IMPI", "NEGI"})
#: `ORE`'s refs are `(j, s1, e1, s2, e2)`: one line citation, then two box citations.
_ORE_RULE = "ORE"

#: `count_orderings`/`enumerate_orderings` give up on exact enumeration above this many permutable
#: units at one level (2**22 subsets is already a lot of pure-Python DP; `INTERP_SPEC.md` §1).
MAX_EXACT_UNITS = 22


class OrderingError(ValueError):
    """`body` is not a complete, well-formed ND proof this module can reorder."""


@dataclass(frozen=True)
class _Line:
    idx: int
    depth: int
    rule: str
    refs: tuple[int, ...]
    tokens: tuple[str, ...]  # the line's own original tokens, verbatim (N<idx> ... ;)


def _parse_line(text: str) -> _Line:
    toks = tuple(text.split())
    if not toks:
        raise OrderingError("empty line")
    m = _N_TOKEN.match(toks[0])
    if m is None:
        raise OrderingError(f"line does not start with an N<i> token: {text!r}")
    idx = int(m.group(1))
    i = 1
    depth = 0
    while i < len(toks) and toks[i] == "|":
        depth += 1
        i += 1
    try:
        colon = toks.index(":", i)
    except ValueError as e:
        raise OrderingError(f"no ':' in line: {text!r}") from e
    if colon + 1 >= len(toks):
        raise OrderingError(f"no rule name in line: {text!r}")
    rule = toks[colon + 1]
    if toks[-1] != ";":
        raise OrderingError(f"line does not end with ';': {text!r}")
    ref_toks = toks[colon + 2 : -1]
    refs = tuple(int(t[1:]) for t in ref_toks)
    return _Line(idx=idx, depth=depth, rule=rule, refs=refs, tokens=toks)


@dataclass
class _ProofStructure:
    line_by_idx: dict[int, _Line]
    children: dict[int, list[Unit]]  # container id -> its direct children, original order
    box_meta: dict[int, dict[str, int]]  # box id -> {start, end, depth, parent}
    container_of: dict[int, int]  # line idx -> its direct container (AS lines map to their OWN box)
    #: per container: unit -> the (possibly out-of-container) units it depends on, unfiltered.
    edges: dict[int, dict[Unit, set[Unit]]] = field(default_factory=dict)

    def parts(self, cid: int) -> tuple[list[Unit], list[Unit], Unit | None]:
        """`(fixed_prefix, permutable, forced_last)` for container `cid`'s direct children:
        `fixed_prefix` is the leading `PR` block (top level only, else always `[]`); `forced_last`
        is the container's chronologically-last child (`None` only if `cid` has no children at
        all, which never happens for a valid proof's `ROOT_ID` and only would for a box with zero
        content, itself impossible -- an `AS` line always has at least one line after it, or the
        proof/enclosing box could not have progressed past it)."""
        full = self.children.get(cid, [])
        n_prefix = 0
        if cid == ROOT_ID:
            while (
                n_prefix < len(full)
                and full[n_prefix][0] == "line"
                and self.line_by_idx[full[n_prefix][1]].rule == "PR"
            ):
                n_prefix += 1
        prefix = full[:n_prefix]
        rest = full[n_prefix:]
        if not rest:
            return prefix, [], None
        return prefix, rest[:-1], rest[-1]


def _build_structure(body: str) -> _ProofStructure:
    if not has_qed(body):
        raise OrderingError("orderings require a complete (QED-terminated) proof")
    raw_lines = split_lines(body)
    if not raw_lines:
        raise OrderingError("empty proof body")
    lines = [_parse_line(t) for t in raw_lines]
    line_by_idx = {ln.idx: ln for ln in lines}

    children: dict[int, list[Unit]] = {ROOT_ID: []}
    box_meta: dict[int, dict[str, int]] = {}
    container_of: dict[int, int] = {}
    stack: list[int] = []
    next_box = 0

    # Mirrors `nd_verify.verify`'s own stack/ctx/box_lines bookkeeping exactly (see module
    # docstring): an AS line at depth d closes boxes deeper than d-1 and opens a new one; any other
    # line at depth d closes boxes deeper than d; every currently-open box's "last line seen" is
    # updated on every line, so it ends up holding the box's true last (possibly nested) content.
    for ln in lines:
        d = ln.depth
        if ln.rule == "AS":
            del stack[d - 1 :]
            next_box += 1
            b = next_box
            parent = stack[-1] if stack else ROOT_ID
            children.setdefault(parent, []).append(("box", b))
            children.setdefault(b, [])
            box_meta[b] = {"start": ln.idx, "end": ln.idx, "depth": d, "parent": parent}
            stack.append(b)
            container_of[ln.idx] = b
        else:
            del stack[d:]
            parent = stack[-1] if stack else ROOT_ID
            children.setdefault(parent, []).append(("line", ln.idx))
            container_of[ln.idx] = parent
        for b in stack:
            box_meta[b]["end"] = ln.idx

    as_line_of = {meta["start"]: bid for bid, meta in box_meta.items()}

    def direct_container(u: Unit) -> int:
        return container_of[u[1]] if u[0] == "line" else box_meta[u[1]]["parent"]

    def climb(u: Unit, target: int) -> Unit:
        while direct_container(u) != target:
            u = ("box", direct_container(u))
        return u

    edges: dict[int, dict[Unit, set[Unit]]] = {}

    def add_edge(container: int, rep: Unit, target: Unit) -> None:
        if rep == target:
            return
        edges.setdefault(container, {}).setdefault(rep, set()).add(target)

    def line_target(r: int) -> tuple[Unit, int]:
        return ("line", r), container_of[r]

    def box_target(s: int) -> tuple[Unit, int]:
        b = as_line_of[s]
        return ("box", b), box_meta[b]["parent"]

    for ln in lines:
        if ln.rule in ("PR", "AS"):
            continue
        citer: Unit = ("line", ln.idx)
        if ln.rule in _BOX_CITE_RULES:
            s, _e = ln.refs
            tgt, tc = box_target(s)
            add_edge(tc, climb(citer, tc), tgt)
        elif ln.rule == _ORE_RULE:
            j, s1, _e1, s2, _e2 = ln.refs
            tgt, tc = line_target(j)
            add_edge(tc, climb(citer, tc), tgt)
            for s in (s1, s2):
                tgt, tc = box_target(s)
                add_edge(tc, climb(citer, tc), tgt)
        else:
            for r in ln.refs:
                tgt, tc = line_target(r)
                add_edge(tc, climb(citer, tc), tgt)

    return _ProofStructure(line_by_idx, children, box_meta, container_of, edges)


def _filtered_deps(st: _ProofStructure, cid: int, permutable: list[Unit]) -> dict[Unit, set[Unit]]:
    """Dependencies of `permutable`'s own members, restricted to `permutable` itself: an edge to
    the fixed `PR` prefix or the forced-last unit is always trivially satisfied (see module
    docstring) and dropped here rather than at `add_edge` time, so `st.edges` stays a plain record
    of every raw promoted citation."""
    allowed = set(permutable)
    raw = st.edges.get(cid, {})
    return {u: (raw.get(u, set()) & allowed) for u in permutable}


def _linear_extension_count(nodes: list[Unit], deps: dict[Unit, set[Unit]]) -> int:
    """Exact count of linear extensions of `nodes` under `deps` (subset DP, `O(2^n * n)`)."""
    n = len(nodes)
    if n <= 1:
        return 1
    index = {u: i for i, u in enumerate(nodes)}
    pred_mask = [0] * n
    for u, deps_u in deps.items():
        i = index[u]
        for d in deps_u:
            pred_mask[i] |= 1 << index[d]
    full = (1 << n) - 1
    dp = [0] * (1 << n)
    dp[0] = 1
    for mask in range(1 << n):
        c = dp[mask]
        if not c:
            continue
        free = full & ~mask
        while free:
            bit = free & (-free)
            free &= free - 1
            i = bit.bit_length() - 1
            if pred_mask[i] & ~mask:
                continue
            dp[mask | bit] += c
    return dp[full]


def count_orderings(body: str) -> int | None:
    """Exact number of orderings equivalent to `body` (product over every container of the linear
    extensions of that container's permutable-unit DAG), or `None` if some container has more than
    `MAX_EXACT_UNITS` permutable units (use `sample_orderings` instead)."""
    st = _build_structure(body)
    total = 1
    for cid in (ROOT_ID, *st.box_meta):
        _prefix, permutable, _last = st.parts(cid)
        if len(permutable) > MAX_EXACT_UNITS:
            return None
        total *= _linear_extension_count(permutable, _filtered_deps(st, cid, permutable))
    return total


def _topo_orderings(nodes: list[Unit], deps: dict[Unit, set[Unit]]) -> Iterator[list[Unit]]:
    """Every linear extension of `nodes` under `deps`, in a deterministic (sorted-ready-set)
    order; used by `enumerate_orderings`, never on a node set large enough for this to matter."""
    if not nodes:
        yield []
        return
    node_set = set(nodes)

    def rec(remaining: frozenset[Unit], chosen: tuple[Unit, ...]) -> Iterator[list[Unit]]:
        if not remaining:
            yield list(chosen)
            return
        placed = node_set - remaining
        ready = sorted(u for u in remaining if deps.get(u, set()) <= placed)
        for u in ready:
            yield from rec(remaining - {u}, (*chosen, u))

    yield from rec(frozenset(nodes), ())


def _orderings_of_container(st: _ProofStructure, cid: int) -> Iterator[list[Unit]]:
    prefix, permutable, last = st.parts(cid)
    deps = _filtered_deps(st, cid, permutable)
    tail = [last] if last is not None else []
    for perm in _topo_orderings(permutable, deps):
        yield [*prefix, *perm, *tail]


def _expand_unit(st: _ProofStructure, u: Unit) -> Iterator[list[int]]:
    if u[0] == "line":
        yield [u[1]]
        return
    as_idx = st.box_meta[u[1]]["start"]
    for rest in _expand_full(st, u[1]):
        yield [as_idx, *rest]


def _expand_full(st: _ProofStructure, cid: int) -> Iterator[list[int]]:
    """Every valid flattening of `cid`'s whole subtree into an original-line-index sequence
    (recursively expanding nested boxes as one contiguous block each -- `itertools.product` over
    each chosen unit's own expansion, so box contents never interleave with their container's
    other units)."""
    for seq in _orderings_of_container(st, cid):
        for combo in itertools.product(*(_expand_unit(st, u) for u in seq)):
            out: list[int] = []
            for part in combo:
                out.extend(part)
            yield out


def _render(st: _ProofStructure, idx_seq: list[int]) -> str:
    """Renumber `idx_seq` (a permutation of every original line index) to `N1..N<len>` and rewrite
    every citation (line or box-citation component alike -- both are just `N<k>` tokens) through
    the same old->new map."""
    remap = {old: i + 1 for i, old in enumerate(idx_seq)}
    out_lines: list[str] = []
    for old in idx_seq:
        toks = st.line_by_idx[old].tokens
        new_toks = [f"N{remap[int(m.group(1))]}" if (m := _N_TOKEN.match(t)) else t for t in toks]
        out_lines.append(" ".join(new_toks))
    return " ".join(out_lines) + " QED"


def enumerate_orderings(body: str, cap: int) -> list[str]:
    """Up to `cap` distinct orderings equivalent to `body` (each a complete, renumbered proof
    body ending in `QED`), enumerated lazily so `cap < count_orderings(body)` never materialises
    more than `cap` of them."""
    st = _build_structure(body)
    return [_render(st, seq) for seq in itertools.islice(_expand_full(st, ROOT_ID), cap)]


def _sample_topo(
    nodes: list[Unit], deps: dict[Unit, set[Unit]], rng: random.Random
) -> tuple[list[Unit], float]:
    """One random linear extension of `nodes` under `deps`, built by repeatedly picking uniformly
    among the currently-ready nodes (`INTERP_SPEC.md` §1's "sequential" sampler); returns the
    chosen order and `log q(order)` (the sum of `-log(#ready choices)` over every step)."""
    remaining = list(nodes)
    placed: set[Unit] = set()
    chosen: list[Unit] = []
    logq = 0.0
    while remaining:
        ready = sorted(u for u in remaining if deps.get(u, set()) <= placed)
        pick = rng.choice(ready)
        logq -= math.log(len(ready))
        chosen.append(pick)
        placed.add(pick)
        remaining.remove(pick)
    return chosen, logq


def _sample_full(st: _ProofStructure, cid: int, rng: random.Random) -> tuple[list[int], float]:
    prefix, permutable, last = st.parts(cid)
    deps = _filtered_deps(st, cid, permutable)
    order, logq = _sample_topo(permutable, deps, rng)
    tail = [last] if last is not None else []
    out: list[int] = []
    for u in (*prefix, *order, *tail):
        if u[0] == "line":
            out.append(u[1])
        else:
            rest, sub_logq = _sample_full(st, u[1], rng)
            out.append(st.box_meta[u[1]]["start"])
            out.extend(rest)
            logq += sub_logq
    return out, logq


def sample_orderings(body: str, m: int, rng: random.Random) -> list[tuple[str, float]]:
    """`m` independent samples from the "pick uniformly among ready units" proposal at every
    level (recursively independent across containers, so the whole proof's `log q` is the sum of
    each container's own): `[(ordering_body, log_q), ...]`, for `likelihood.marginal_logprob`'s
    importance-sampling estimator when `count_orderings` gives up."""
    st = _build_structure(body)
    out: list[tuple[str, float]] = []
    for _ in range(m):
        idx_seq, logq = _sample_full(st, ROOT_ID, rng)
        out.append((_render(st, idx_seq), logq))
    return out
