"""Glued-propositions suite: compositional (multi-part) proof construction (plan §2 B1 / WP4).

Given a pool of small "part" theorems (each with a verified proof), this module joins several of
them into one larger theorem via one of five operators (`AND`, `CHAIN`, `CASE-dilemma`,
`CASE-same`, `BOX`; plan §2 B1's table) and constructs an explicit witness proof for the join by
concatenating the parts' own (renamed, renumbered) proof lines plus a handful of new combining
lines. Every witness is self-checked with the vendored `nd_verify` and then, since a naive
concatenation can admit a much shorter proof than intended (plan §0's chain-glue shortcut:
`(P&Q)⊢(Q&P)` then `(Q&P),(Q>R)⊢R` glues to a naive 7-line witness whose true `minlen` is 4), only
kept if an injected `Certifier` confirms `L` via `minlen_exact` or `ub_lb` with a bound `>= L-1`
(plan §2's certifier policy); otherwise the candidate is dropped and counted.

The join constructions below are a fresh, dependency-free reimplementation (this package's
`core.formula`/`core.proof` types; no vendor import) of ideas from three read-only sources named in
the plan's WP4 row (none of the three modules are importable from this branch/worktree: two live
under `/scratch/cowifornia/nd-rl/ab3` and one on a different git branch entirely), used as
inspiration and, in a couple of places, ported near-verbatim with attribution on the function:

  * `/scratch/cowifornia/nd-rl/ab3/data_build/tools/comp.py` -- `Part`/atom-renaming
    representation, `combine_chain` (producer/consumer gluing), `tight`/`satisfiable`/`entails`
    (joint-satisfiability filters; not currently wired into a filter here -- see the WP4 report's
    open issues), `fkey` (shape-only formula key used here as `_shape`).
  * `midtrain_long_glued_20260926/build_tools/build_mid.py` -- `combine_tree` (random binary
    AND-tree join over disjoint-renamed parts; ported as `_join_and`), the disjoint-atom-slot idea
    (plan's "4 atoms only" note).
  * `length_generalisation_20260925/composition/build.py` -- `joined_proof`/`with_premises` (premise
    merge/reorder bookkeeping for a multi-part proof; the same idea underlies `_append_part` below).

`CASE-dilemma`/`CASE-same`/`BOX` have no prior-art source in the plan's WP4 row (comp.py only
supplies AND/CHAIN); their Fitch-proof constructions are derived directly from the plan's witness-L
formulas (see the docstrings on `_join_case`/`_join_box`) and verified empirically against
`nd_verify` in `test_wp4_glue.py`.
"""

from __future__ import annotations

import collections
import importlib
import json
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ..core import formula as F
from ..core import proof as P
from ..core.io import read_jsonl
from ..core.types import JSON, Certifier, Excluder, LengthLabel, Problem, ReferenceProof, Sequent
from ..logic.skills import needs_ore as _skills_needs_ore
from ..store import Store
from ..vendor import load_nd_verify

#: Default overall `Store` root (plan §1's "Outputs: /scratch/cowifornia/nd-rl/bench/out").
DEFAULT_STORE_ROOT = "/scratch/cowifornia/nd-rl/bench/out"

#: Default source for `build_length_sweep` (plan's WP4 row: "re-wraps the existing matched AND/
#: CHAIN items ... as an extra length sweep").
DEFAULT_MATCHED_PATH = "/scratch/cowifornia/nd-rl/ab3/sealed/matched_L6-14.jsonl"

_Line = tuple[int, F.Formula, str, tuple[int, ...]]  # (depth, formula, rule, 1-based refs)


# --------------------------------------------------------------------------------------- Part


@dataclass(frozen=True)
class _Part:
    """One pool theorem, parsed into premises/conclusion `Formula`s and proof `_Line`s (premise
    lines included, 1-based citations exactly as written -- the same shape `comp.py`'s `Part`
    uses). `pid` is preserved (unchanged) by every renaming operation below, so a join's
    `Problem.parts` can always be recovered from the renamed `_Part`s it was built from."""

    pid: str
    premises: tuple[F.Formula, ...]
    concl: F.Formula
    lines: tuple[_Line, ...]
    L: int
    atoms: tuple[str, ...]


def _parse_lines(body: str) -> tuple[_Line, ...]:
    """Parse a (N1-based, QED-terminated) proof body into `_Line`s.

    Reimplements the same grammar `vendor/nd_verify/verify.py`'s `parse_proof_tokens` and
    `comp.py`'s `parse_proof` use, but built on this package's own `core.formula.parse_formula`
    (no vendor import) and returning our own `_Line` shape.
    """
    body = P.renumber(body, new_start=1)
    out: list[_Line] = []
    for raw in P.split_lines(body):
        toks = raw.split()
        assert toks and toks[-1] == ";", f"malformed proof line: {raw!r}"
        toks = toks[:-1]
        i = 1
        depth = 0
        while i < len(toks) and toks[i] == "|":
            depth += 1
            i += 1
        f, i = F.parse_formula(toks, i)
        assert toks[i] == ":", f"missing ':' in proof line: {raw!r}"
        rule = toks[i + 1]
        refs = tuple(int(t[1:]) for t in toks[i + 2 :])
        out.append((depth, f, rule, refs))
    return tuple(out)


def _render_lines(lines: Sequence[_Line]) -> str:
    """Inverse of `_parse_lines` (plus a trailing `QED`): renders `_Line`s back to N1-based text."""
    toks: list[str] = []
    for i, (d, f, rule, refs) in enumerate(lines, start=1):
        toks.append(f"N{i}")
        toks.extend("|" * d)
        toks.extend(F.render(f).split())
        toks.append(":")
        toks.append(rule)
        toks.extend(f"N{r}" for r in refs)
        toks.append(";")
    toks.append("QED")
    return " ".join(toks)


def to_part(problem: Problem) -> _Part:
    """Parse a pool `Problem` (with a reference proof) into a `_Part`."""
    assert problem.ref is not None, f"{problem.pid}: parts need a reference proof"
    premises = tuple(F.parse(p) for p in problem.sequent.premises)
    concl = F.parse(problem.sequent.conclusion)
    lines = _parse_lines(problem.ref.body)
    assert len(lines) >= len(premises), f"{problem.pid}: fewer proof lines than premises"
    assert lines[: len(premises)] == tuple((0, p, "PR", ()) for p in premises), (
        f"{problem.pid}: proof premise block does not match the declared premises"
    )
    atoms = set()
    for f in (*premises, concl):
        atoms |= F.atoms(f)
    return _Part(
        pid=problem.pid,
        premises=premises,
        concl=concl,
        lines=lines,
        L=len(lines),
        atoms=tuple(sorted(atoms)),
    )


# ---------------------------------------------------------------------------- atoms / renaming


def _rename_formula(f: F.Formula, m: Mapping[str, str]) -> F.Formula:
    if f[0] == "atom":
        return ("atom", m[cast(str, f[1])])
    if f[0] == "not":
        return ("not", _rename_formula(cast(F.Formula, f[1]), m))
    if f[0] in ("and", "or", "imp"):
        return (
            f[0],
            _rename_formula(cast(F.Formula, f[1]), m),
            _rename_formula(cast(F.Formula, f[2]), m),
        )
    return f


def _rename_part(part: _Part, m: Mapping[str, str]) -> _Part:
    premises = tuple(_rename_formula(p, m) for p in part.premises)
    concl = _rename_formula(part.concl, m)
    lines = tuple((d, _rename_formula(f, m), r, refs) for d, f, r, refs in part.lines)
    atoms = tuple(sorted(m.get(a, a) for a in part.atoms))
    return _Part(pid=part.pid, premises=premises, concl=concl, lines=lines, L=part.L, atoms=atoms)


def _shape(f: F.Formula) -> tuple[str, tuple[str, ...]]:
    """`(canonical rendering, atoms in first-occurrence order)`: two formulas share a shape iff
    some atom-renaming makes them identical. Ported from `comp.py`'s `fkey`."""
    order: list[str] = []

    def walk(g: F.Formula) -> None:
        if g[0] == "atom":
            a = cast(str, g[1])
            if a not in order:
                order.append(a)
        elif g[0] == "not":
            walk(cast(F.Formula, g[1]))
        elif g[0] in ("and", "or", "imp"):
            walk(cast(F.Formula, g[1]))
            walk(cast(F.Formula, g[2]))

    walk(f)
    canon = {a: F.ATOMS[i] for i, a in enumerate(order)}
    return F.render(_rename_formula(f, canon)), tuple(order)


def _rename_fresh(
    part: _Part,
    *,
    iface_order: tuple[str, ...],
    target_iface: tuple[str, ...],
    used_atoms: frozenset[str],
    rng: random.Random,
) -> tuple[_Part, frozenset[str]] | None:
    """Rename `part` onto atoms disjoint from `used_atoms`, forcing its atoms in `iface_order`
    (its own atom names, in some subformula's first-occurrence order) onto the already-concrete
    `target_iface` names -- so that subformula becomes literally identical to another,
    independently-built part's matching subformula (the "graft interface"; see `_shape`).
    `iface_order == target_iface == ()` renames `part` fully freshly (no forced interface): the
    case used for the first part in any join. Returns `None` if fewer than
    `len(part.atoms) - len(iface_order)` atoms remain free (the 4-atom budget from plan §2 B1 --
    "with only 4 atoms ... handle the atom budget explicitly" -- is exhausted).
    """
    other = [a for a in part.atoms if a not in iface_order]
    free = [a for a in F.ATOMS if a not in used_atoms and a not in target_iface]
    if len(free) < len(other):
        return None
    rng.shuffle(free)
    mapping = dict(zip(iface_order, target_iface, strict=True))
    mapping.update(zip(other, free[: len(other)], strict=True))
    renamed = _rename_part(part, mapping)
    return renamed, used_atoms | frozenset(renamed.atoms)


# ----------------------------------------------------------------------------------- grafting


def _append_part(
    lines: list[_Line], part: _Part, local_to_global_prem: Mapping[int, int], depth_shift: int = 0
) -> int:
    """Append `part`'s own derivation lines (excluding its premise lines) to `lines` (mutated in
    place), remapping every citation: a reference to one of `part`'s own premises (local index
    `1..len(part.premises)`) goes through `local_to_global_prem`; a reference to one of `part`'s
    own earlier *derived* lines goes through a running map extended as each new line is appended.
    Ported from `comp.combine_chain`/`combine_andi`'s inner loop and `build_mid.combine_tree`
    (see module docstring). Returns the new global index of `part`'s own last (conclusion) line.
    """
    m = dict(local_to_global_prem)
    n_prem = len(part.premises)
    for k, (d, f, rule, refs) in enumerate(part.lines[n_prem:]):
        local_idx = n_prem + k + 1
        lines.append((d + depth_shift, f, rule, tuple(m[r] for r in refs)))
        m[local_idx] = len(lines)
    return len(lines)


def _merge_premises(groups: Iterable[Sequence[F.Formula]], rng: random.Random) -> list[F.Formula]:
    prem: list[F.Formula] = []
    for g in groups:
        for f in g:
            if f not in prem:
                prem.append(f)
    rng.shuffle(prem)
    return prem


# -------------------------------------------------------------------------------- join: AND


def _join_and(parts: Sequence[_Part], rng: random.Random) -> tuple[Sequent, str, int]:
    """`build_mid.combine_tree`: random part order, random binary ANDI tree over the (disjoint-
    renamed) parts' conclusions; premises merged (deduplicated) and shuffled. Witness L = ΣLi +
    k - 1 (one ANDI line per join, k-1 joins for k parts) when no premises are shared."""
    order = list(parts)
    rng.shuffle(order)
    prem = _merge_premises((p.premises for p in order), rng)
    pidx = {f: i + 1 for i, f in enumerate(prem)}
    lines: list[_Line] = [(0, f, "PR", ()) for f in prem]
    nodes: list[tuple[F.Formula, int]] = []
    for part in order:
        local_map = {i + 1: pidx[f] for i, f in enumerate(part.premises)}
        end = _append_part(lines, part, local_map)
        nodes.append((part.concl, end))
    while len(nodes) > 1:
        i, j = rng.sample(range(len(nodes)), 2)
        a, b = nodes[i], nodes[j]
        f = ("and", a[0], b[0])
        lines.append((0, f, "ANDI", (a[1], b[1])))
        nodes = [n for t, n in enumerate(nodes) if t not in (i, j)] + [(f, len(lines))]
    seq = Sequent(premises=tuple(F.render(p) for p in prem), conclusion=F.render(nodes[0][0]))
    return seq, _render_lines(lines), len(lines)


# ------------------------------------------------------------------------------ join: CHAIN


def _join_chain(
    parts: Sequence[_Part], link_indices: Sequence[int], rng: random.Random
) -> tuple[Sequent, str, int]:
    """`comp.combine_chain`, generalised to a k-part chain: `parts[i]`'s premise at
    `link_indices[i]` (for `i>=1`) must equal `parts[i-1].concl` *literally* (the caller arranges
    this via `_rename_fresh`); that premise is dropped from the merged premise block and its
    citations point at the producer's conclusion line instead. Witness L = ΣLi - (k-1)."""
    groups = [parts[0].premises]
    for i in range(1, len(parts)):
        groups.append(tuple(f for j, f in enumerate(parts[i].premises) if j != link_indices[i]))
    prem = _merge_premises(groups, rng)
    pidx = {f: i + 1 for i, f in enumerate(prem)}
    lines: list[_Line] = [(0, f, "PR", ()) for f in prem]
    local_map0 = {i + 1: pidx[f] for i, f in enumerate(parts[0].premises)}
    prev_end = _append_part(lines, parts[0], local_map0)
    for i in range(1, len(parts)):
        local_map = {
            j + 1: (prev_end if j == link_indices[i] else pidx[f])
            for j, f in enumerate(parts[i].premises)
        }
        prev_end = _append_part(lines, parts[i], local_map)
    seq = Sequent(premises=tuple(F.render(p) for p in prem), conclusion=F.render(parts[-1].concl))
    return seq, _render_lines(lines), len(lines)


# ------------------------------------------------------------------------------- join: CASE


def _join_case(
    part_a: _Part, a_idx: int, part_b: _Part, b_idx: int, *, same: bool
) -> tuple[Sequent, str, int]:
    """CASE-dilemma (`same=False`) / CASE-same (`same=True`), plan §2 B1's CASE row. `part_a`'s
    premise `a_idx` ("A") and `part_b`'s premise `b_idx` ("B") become box assumptions under a new
    `A v B` premise; each box replays that part's own derivation (depth+1, citations remapped the
    same way `_join_and`/`_join_chain` do) and is closed by `ORE`. `same=False` additionally closes
    each box with an `ORI1`/`ORI2` line onto the shared target `C1 v C2` (witness L = L1+L2+4 minus
    any shared non-case premises); `same=True` requires `part_a.concl == part_b.concl` and skips
    the ORI lines, since each box already ends at the shared conclusion (witness L = L1+L2+2).
    """
    a_case = part_a.premises[a_idx]
    b_case = part_b.premises[b_idx]
    if same:
        assert part_a.concl == part_b.concl, "CASE-same requires identical (renamed) conclusions"
        target = part_a.concl
    else:
        target = ("or", part_a.concl, part_b.concl)
    disj: F.Formula = ("or", a_case, b_case)
    gamma_a = tuple(f for i, f in enumerate(part_a.premises) if i != a_idx)
    gamma_b = tuple(f for i, f in enumerate(part_b.premises) if i != b_idx)
    # The disjunction is always the first premise (cosmetic); Γ1/Γ2 are merged (deduplicated) after
    # it, in their original relative order -- no shuffle, since order does not affect witness L.
    prem = [disj]
    for f in gamma_a + gamma_b:
        if f not in prem:
            prem.append(f)
    pidx = {f: i + 1 for i, f in enumerate(prem)}
    lines: list[_Line] = [(0, f, "PR", ()) for f in prem]

    as_a_line = len(lines) + 1
    lines.append((1, a_case, "AS", ()))
    local_map_a = {
        i + 1: (as_a_line if i == a_idx else pidx[f]) for i, f in enumerate(part_a.premises)
    }
    end_a = _append_part(lines, part_a, local_map_a, depth_shift=1)
    if not same:
        lines.append((1, target, "ORI1", (end_a,)))
        end_a = len(lines)

    as_b_line = len(lines) + 1
    lines.append((1, b_case, "AS", ()))
    local_map_b = {
        i + 1: (as_b_line if i == b_idx else pidx[f]) for i, f in enumerate(part_b.premises)
    }
    end_b = _append_part(lines, part_b, local_map_b, depth_shift=1)
    if not same:
        lines.append((1, target, "ORI2", (end_b,)))
        end_b = len(lines)

    lines.append((0, target, "ORE", (pidx[disj], as_a_line, end_a, as_b_line, end_b)))
    seq = Sequent(premises=tuple(F.render(f) for f in prem), conclusion=F.render(target))
    return seq, _render_lines(lines), len(lines)


# -------------------------------------------------------------------------------- join: BOX


def _join_box(part: _Part, d: int) -> tuple[Sequent, str, int] | None:
    """`Γ,A1..Ad ⊢ B ⇒ Γ ⊢ A1→(...→(Ad→B)...)` (plan §2 B1's BOX row). The part's own last `d`
    premises become nested box assumptions *in place* (same line positions, `PR` -> `AS`, depth
    `1..d`); every other line's depth shifts by `+d` (citations are untouched -- every position
    keeps its original global index, since only the premises' rule/depth change and nothing is
    inserted before the body). `d` `IMPI` lines then close the boxes innermost-first. No atom
    renaming: a single-part transform. Witness L = Lpart + d. `None` if `part` has fewer than `d`
    premises."""
    n = len(part.premises)
    if not (1 <= d <= n):
        return None
    gamma = part.premises[: n - d]
    boxed = part.premises[n - d :]
    lines: list[_Line] = [(0, f, "PR", ()) for f in gamma]
    for k, f in enumerate(boxed):
        lines.append((k + 1, f, "AS", ()))
    for depth, f, rule, refs in part.lines[n:]:
        lines.append((depth + d, f, rule, refs))
    as_line_idx = [n - d + k + 1 for k in range(d)]
    body_formula = part.concl
    cur_line = len(lines)
    for level in range(d, 0, -1):
        target_formula = boxed[level - 1]
        body_formula = ("imp", target_formula, body_formula)
        lines.append((level - 1, body_formula, "IMPI", (as_line_idx[level - 1], cur_line)))
        cur_line = len(lines)
    seq = Sequent(premises=tuple(F.render(f) for f in gamma), conclusion=F.render(body_formula))
    return seq, _render_lines(lines), len(lines)


# ------------------------------------------------------------------------- shape indices / search


def _index_by_len(parts: Sequence[_Part]) -> dict[int, list[_Part]]:
    idx: dict[int, list[_Part]] = collections.defaultdict(list)
    for p in parts:
        idx[p.L].append(p)
    return idx


def _index_premises_by_shape(parts: Sequence[_Part]) -> dict[str, list[tuple[_Part, int]]]:
    idx: dict[str, list[tuple[_Part, int]]] = collections.defaultdict(list)
    for p in parts:
        for i, f in enumerate(p.premises):
            idx[_shape(f)[0]].append((p, i))
    return idx


def _index_concl_by_shape(parts: Sequence[_Part]) -> dict[str, list[_Part]]:
    idx: dict[str, list[_Part]] = collections.defaultdict(list)
    for p in parts:
        idx[_shape(p.concl)[0]].append(p)
    return idx


def _random_split(
    total: int, k: int, available: Iterable[int], rng: random.Random
) -> list[int] | None:
    """A random composition of `total` into `k` positive parts, each present in `available`, or
    `None` if 50 random attempts find none (the caller retries the whole cell many times, so this
    just needs to be *a* chance at a split, not an exhaustive search)."""
    avail = set(available)
    if k == 1:
        return [total] if total in avail else None
    for _ in range(50):
        cuts = sorted(rng.sample(range(1, total), k - 1))
        lens = [b - a for a, b in zip((0, *cuts), (*cuts, total), strict=True)]
        if all(ln >= 1 and ln in avail for ln in lens):
            rng.shuffle(lens)
            return lens
    return None


def _try_and(
    by_len: dict[int, list[_Part]], lens: Sequence[int], rng: random.Random
) -> list[_Part] | None:
    chosen: list[_Part] = []
    used_names: set[str] = set()
    for li in lens:
        cands = [p for p in by_len.get(li, []) if p.pid not in used_names]
        if not cands:
            return None
        pick = rng.choice(cands)
        chosen.append(pick)
        used_names.add(pick.pid)
    used_atoms: frozenset[str] = frozenset()
    renamed: list[_Part] = []
    for p in chosen:
        result = _rename_fresh(p, iface_order=(), target_iface=(), used_atoms=used_atoms, rng=rng)
        if result is None:
            return None
        rp, used_atoms = result
        renamed.append(rp)
    return renamed


def _try_chain(
    by_len: dict[int, list[_Part]],
    premise_idx: dict[str, list[tuple[_Part, int]]],
    lens: Sequence[int],
    rng: random.Random,
) -> tuple[list[_Part], list[int]] | None:
    cand0 = by_len.get(lens[0], [])
    if not cand0:
        return None
    used_names: set[str] = set()
    part0 = rng.choice(cand0)
    used_names.add(part0.pid)
    result0 = _rename_fresh(part0, iface_order=(), target_iface=(), used_atoms=frozenset(), rng=rng)
    if result0 is None:
        return None
    prev, used_atoms = result0
    parts_out = [prev]
    link_indices = [-1]
    for li in lens[1:]:
        shape_prev = _shape(prev.concl)[0]
        target_iface = _shape(prev.concl)[1]
        candidates = [
            (p, i)
            for p, i in premise_idx.get(shape_prev, [])
            if p.pid not in used_names and li == p.L
        ]
        if not candidates:
            return None
        part_i, prem_i = rng.choice(candidates)
        iface_order = _shape(part_i.premises[prem_i])[1]
        result = _rename_fresh(
            part_i,
            iface_order=iface_order,
            target_iface=target_iface,
            used_atoms=used_atoms,
            rng=rng,
        )
        if result is None:
            return None
        renamed_i, used_atoms = result
        parts_out.append(renamed_i)
        link_indices.append(prem_i)
        used_names.add(part_i.pid)
        prev = renamed_i
    return parts_out, link_indices


def _try_case_dilemma(
    by_len: dict[int, list[_Part]], ell: int, rng: random.Random
) -> tuple[_Part, int, _Part, int] | None:
    cands = [p for p in by_len.get(ell, []) if p.premises]
    if len(cands) < 2:
        return None
    a, b = rng.sample(cands, 2)
    result_a = _rename_fresh(a, iface_order=(), target_iface=(), used_atoms=frozenset(), rng=rng)
    if result_a is None:
        return None
    ra, used = result_a
    result_b = _rename_fresh(b, iface_order=(), target_iface=(), used_atoms=used, rng=rng)
    if result_b is None:
        return None
    rb, _ = result_b
    return ra, rng.randrange(len(ra.premises)), rb, rng.randrange(len(rb.premises))


def _try_case_same(
    concl_idx: dict[str, list[_Part]], rng: random.Random
) -> tuple[_Part, int, _Part, int] | None:
    shapes = [
        s for s, lst in concl_idx.items() if len({p.pid for p in lst}) >= 2 and lst[0].premises
    ]
    if not shapes:
        return None
    shape = rng.choice(shapes)
    pool = [p for p in concl_idx[shape] if p.premises]
    if len(pool) < 2:
        return None
    a, b = rng.sample(pool, 2)
    result_a = _rename_fresh(a, iface_order=(), target_iface=(), used_atoms=frozenset(), rng=rng)
    if result_a is None:
        return None
    ra, used = result_a
    iface_order = _shape(b.concl)[1]
    target_iface = _shape(ra.concl)[1]
    result_b = _rename_fresh(
        b, iface_order=iface_order, target_iface=target_iface, used_atoms=used, rng=rng
    )
    if result_b is None:
        return None
    rb, _ = result_b
    return ra, rng.randrange(len(ra.premises)), rb, rng.randrange(len(rb.premises))


# --------------------------------------------------------------------------------- certification


def _accepts(label: LengthLabel, witness_l: int) -> bool:
    """Plan §2 B1's certification gate: keep iff `minlen_exact` confirms the witness length
    exactly, or `ub_lb` confirms no proof shorter than `witness_l - 1` exists (a bound `>= L-1`
    that was actually exhausted, not timed out). Anything else (a shorter proof found -- the
    chain-glue shortcut -- a timeout, or an unrecognised method) is rejected."""
    if label.timeout:
        return False
    if label.method == "minlen_exact":
        return witness_l == label.L
    if label.method == "ub_lb":
        return label.L_ub == witness_l and label.bound is not None and label.bound >= witness_l - 1
    return False


def needs_ore_from_skills(seq: Sequent, body: str) -> bool:
    """`build_joined`'s `needs_ore: Callable[[Sequent, str], bool]` shape, wired to the real
    `logic.skills.needs_ore(seq: Sequent) -> bool` (WP8 integration fix list item 7). The two
    signatures disagree only in that this one also takes the candidate join's witness `body`:
    plan §2 B3's `case_split`/`needs_ore` test is a pure truth-table property of the *sequent*
    `Γ ⊢ C` alone (every ND proof of it needs ORE, or none does -- it does not depend on which
    particular proof is being checked), so `body` is accepted for `build_joined`'s call-site shape
    and otherwise unused. Point a glue build's `cfg["needs_ore"]` at
    `"code.ndbench.suites.glue:needs_ore_from_skills"` (see `configs/glue.yaml`) to use this for
    real CLI builds, rather than leaving CASE items ungated (`needs_ore_fn is None`, which
    `build_joined` treats as "drop every CASE item", never as "assume it passes")."""
    del body
    return _skills_needs_ore(seq)


def _resolve_optional(spec: object) -> Any:
    """Lazily resolve a `"module:attr"` string (mirrors `cli._resolve`); `None`/falsy passes
    through unchanged. Used for `cfg["needs_ore"]`, which -- unlike `certifier`/`excluder` -- is
    not resolved by `cli.cmd_build` itself (see plan's WP4 row: "For `needs_ore`, accept an
    injected callable")."""
    if not spec:
        return None
    if not isinstance(spec, str):
        return spec
    module_name, _, attr = spec.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


# -------------------------------------------------------------------------------------- cells


@dataclass(frozen=True)
class _Cell:
    op: str
    tag: str
    k: int
    part_len: int | None  # fixed uniform per-part length, when the cell sweeps L via ell
    total_l: int | None  # fixed target total L, when the cell sweeps k (or pools) at fixed L
    d: int | None = None


def _cells() -> list[_Cell]:
    """The ~23 cells of plan §2 B1's table."""
    cells: list[_Cell] = []
    for k in (2, 3, 4):
        cells.append(_Cell("AND", f"AND_k{k}_L12", k, None, 12))
    for ell in range(2, 8):
        cells.append(_Cell("AND", f"AND_k2_ell{ell}", 2, ell, None))
    for k in (2, 3):
        cells.append(_Cell("CHAIN", f"CHAIN_k{k}_L10", k, None, 10))
    for ell in (3, 4, 5):
        cells.append(_Cell("CHAIN", f"CHAIN_k2_ell{ell}", 2, ell, None))
    for ell in range(2, 6):
        cells.append(_Cell("CASE_dilemma", f"CASE_dilemma_ell{ell}", 2, ell, None))
    cells.append(_Cell("CASE_same", "CASE_same_pooled", 2, None, None))
    for d in (1, 2, 3):
        cells.append(_Cell("BOX", f"BOX_d{d}", 1, None, None, d=d))
    return cells


def _excluded(seq: Sequent, excluder: Excluder | None) -> bool:
    """Plan §2's dedup policy: "drop for joined ... items" on a match in *either* tier (train or
    eval) -- unlike family/part items, which only get flagged on an eval-tier match."""
    if excluder is None:
        return False
    in_train, eval_sets = excluder.status(seq)
    return bool(in_train or eval_sets)


def _make_problem(
    *,
    suite: str,
    version: str,
    idx: int,
    seq: Sequent,
    body: str,
    label: LengthLabel,
    tags: dict[str, JSON],
    part_pids: tuple[str, ...],
) -> Problem:
    ref = P.make_reference(body, source="constructed")
    pid = f"{suite}_{tags.get('cell', tags.get('op', 'x'))}_{idx:05d}"
    return Problem(
        pid=pid,
        suite=suite,
        version=version,
        sequent=seq,
        length=label,
        ref=ref,
        tags=tags,
        parts=part_pids,
    )


def _mastery_from_store(store: Store, run_id: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in store.items(run_id):
        out[item.pid] = (item.c / item.n) if item.n else 0.0
    return out


# ----------------------------------------------------------------------------------- build_joined


def build_joined(
    cfg: Mapping[str, Any],
    *,
    certifier: Certifier | None,
    excluder: Excluder | None,
    rng_seed: int,
    needs_ore: Callable[[Sequent, str], bool] | None = None,
    replay: Sequence[Problem] | None = None,
    on_accept: Callable[[Problem, Sequence[_Part]], None] | None = None,
) -> tuple[list[Problem], dict[str, Any]]:
    """Build the glue ("joined") suite (plan §2 B1 / §3 WP4 row).

    `cfg` keys (all optional except `mastery_run_ids` for a real, non-demo build):
      - `store_root` (str): overall `Store` root, default `DEFAULT_STORE_ROOT`.
      - `parts_suite`/`parts_version` (str): where the part pool lives, default `"parts_pool"`/
        `"v1"` (WP2's `suites/existing.py:parts_pool`).
      - `mastery_run_ids` (dict[model, run_id]): a G1-style run over the part pool, used to gate
        which parts are eligible (plan: "built after G1 from parts the RL model mastered"). If
        omitted, every part in the pool is eligible (documented fallback for a demo build).
      - `gate_model` (str): which model's mastery gates inclusion; default the first key of
        `mastery_run_ids`.
      - `mastery_threshold` (float): default 0.5.
      - `target_per_cell` / `min_per_cell` (int): default 100 / 80 (plan §3's "100 items/cell ...
        or the cell reported short").
      - `max_tries_per_cell` (int): construction attempts per cell before giving up; default 400.
      - `time_s` (float): per-candidate certifier time budget; default 20.0.
      - `version` (str): suite version tag on every built `Problem`; default `"v1"`.
    `needs_ore`: `(Sequent, body) -> bool`; required for CASE items to be kept (plan: "CASE items
    must also satisfy `needs_ore`"). Falls back to `cfg["needs_ore"]` (a `"module:attr"` string,
    lazily resolved) and then to `None`, in which case CASE items are dropped and counted (a safe
    default: never claim an ORE-free-orphan-free item without checking).

    `replay`: an already-built glue suite (in build order). Instead of verifying and certifying
    each candidate, accept it iff its sequent equals the next `replay` item's, reusing that item's
    length label. With the same `cfg`/`rng_seed` this retraces the original build exactly (no
    acceptance test consumes `rng`), so `on_accept(problem, used_parts)` sees the atom-renamed
    `_Part`s each item was built from -- which the stored suite does not record (see
    `build_renamed_parts`).
    """
    rng = random.Random(rng_seed)
    store = Store(Path(cast(str, cfg.get("store_root", DEFAULT_STORE_ROOT))))
    parts_suite = cast(str, cfg.get("parts_suite", "parts_pool"))
    parts_version = cast(str, cfg.get("parts_version", "v1"))
    pool_problems = store.load_suite(parts_suite, parts_version)

    mastery_run_ids = cast(dict[str, str], cfg.get("mastery_run_ids", {}))
    threshold = float(cast(float, cfg.get("mastery_threshold", 0.5)))
    mastery = {
        model: _mastery_from_store(store, run_id) for model, run_id in mastery_run_ids.items()
    }
    gate_model = cast(str | None, cfg.get("gate_model")) or (next(iter(mastery_run_ids), None))
    if gate_model is not None and gate_model in mastery:
        gated = {pid for pid, p in mastery[gate_model].items() if p >= threshold}
        eligible = [p for p in pool_problems if p.pid in gated]
    else:
        eligible = list(pool_problems)

    needs_ore_fn = needs_ore or _resolve_optional(cfg.get("needs_ore"))
    parts = [to_part(p) for p in eligible]
    by_len = _index_by_len(parts)
    premise_idx = _index_premises_by_shape(parts)
    concl_idx = _index_concl_by_shape(parts)
    nd_verify = load_nd_verify()

    target_per_cell = int(cast(int, cfg.get("target_per_cell", 100)))
    min_per_cell = int(cast(int, cfg.get("min_per_cell", 80)))
    max_tries = int(cast(int, cfg.get("max_tries_per_cell", 400)))
    time_s = float(cast(float, cfg.get("time_s", 20.0)))
    version = cast(str, cfg.get("version", "v1"))

    problems: list[Problem] = []
    seen_pkeys: set[str] = set()
    cell_counts: dict[str, int] = {}
    drops: collections.Counter[str] = collections.Counter()

    def part_mastery_tags(used: Sequence[_Part]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for model, m in mastery.items():
            out[model] = {p.pid: m[p.pid] for p in used if p.pid in m}
        return out

    def consider(
        seq: Sequent,
        body: str,
        witness_l: int,
        op: str,
        cell: str,
        k: int,
        part_pids: tuple[str, ...],
        part_len: int | None,
        used: Sequence[_Part],
        extra_tags: dict[str, JSON],
    ) -> bool:
        dedup_key = seq.pkey() if len(seq.premises) <= 5 else seq.key()
        if replay is not None:
            nxt = replay[len(problems)] if len(problems) < len(replay) else None
            if nxt is None or (tuple(nxt.sequent.premises), nxt.sequent.conclusion) != (
                tuple(seq.premises),
                seq.conclusion,
            ):
                drops["replay_skip"] += 1
                return False
            label = nxt.length
        else:
            ok, _reason, nl = nd_verify.verify_text(seq.prompt() + " " + body)
            if not (ok and nl == witness_l):
                drops["self_check_fail"] += 1
                return False
            if dedup_key in seen_pkeys:
                drops["duplicate"] += 1
                return False
            if _excluded(seq, excluder):
                drops["excluded"] += 1
                return False
            if op.startswith("CASE") and (needs_ore_fn is None or not needs_ore_fn(seq, body)):
                drops["needs_ore_fail_or_unavailable"] += 1
                return False
            if certifier is None:
                drops["no_certifier"] += 1
                return False
            label = certifier.certify([seq], [body], bound=witness_l, time_s=time_s)[0]
            if not _accepts(label, witness_l):
                drops["cert_reject"] += 1
                return False
        seen_pkeys.add(dedup_key)
        tags: dict[str, JSON] = {
            "op": op,
            "k": k,
            "part_len": part_len,
            "cell": cell,
            "part_mastery": cast(JSON, part_mastery_tags(used)),
        }
        tags.update(extra_tags)
        problems.append(
            _make_problem(
                suite="glue",
                version=version,
                idx=len(problems),
                seq=seq,
                body=body,
                label=label,
                tags=tags,
                part_pids=part_pids,
            )
        )
        if on_accept is not None:
            on_accept(problems[-1], used)
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
        return True

    for cell in _cells():
        got = 0
        for _ in range(max_tries):
            if got >= target_per_cell:
                break
            if cell.op == "AND":
                k = cell.k
                lens = (
                    [cell.part_len] * k
                    if cell.part_len is not None
                    else _random_split(cast(int, cell.total_l) - (k - 1), k, by_len, rng)
                )
                if lens is None:
                    continue
                chosen = _try_and(by_len, lens, rng)
                if chosen is None:
                    continue
                seq, body, ln = _join_and(chosen, rng)
                if consider(
                    seq,
                    body,
                    ln,
                    "AND",
                    cell.tag,
                    k,
                    tuple(p.pid for p in chosen),
                    cell.part_len,
                    chosen,
                    {},
                ):
                    got += 1
            elif cell.op == "CHAIN":
                k = cell.k
                lens = (
                    [cell.part_len] * k
                    if cell.part_len is not None
                    else _random_split(cast(int, cell.total_l) + (k - 1), k, by_len, rng)
                )
                if lens is None:
                    continue
                result = _try_chain(by_len, premise_idx, lens, rng)
                if result is None:
                    continue
                chain_parts, link_indices = result
                seq, body, ln = _join_chain(chain_parts, link_indices, rng)
                if consider(
                    seq,
                    body,
                    ln,
                    "CHAIN",
                    cell.tag,
                    k,
                    tuple(p.pid for p in chain_parts),
                    cell.part_len,
                    chain_parts,
                    {},
                ):
                    got += 1
            elif cell.op == "CASE_dilemma":
                ell = cast(int, cell.part_len)
                found = _try_case_dilemma(by_len, ell, rng)
                if found is None:
                    continue
                pa, ai, pb, bi = found
                seq, body, ln = _join_case(pa, ai, pb, bi, same=False)
                if consider(
                    seq, body, ln, "CASE-dilemma", cell.tag, 2, (pa.pid, pb.pid), ell, (pa, pb), {}
                ):
                    got += 1
            elif cell.op == "CASE_same":
                found = _try_case_same(concl_idx, rng)
                if found is None:
                    continue
                pa, ai, pb, bi = found
                seq, body, ln = _join_case(pa, ai, pb, bi, same=True)
                if consider(
                    seq, body, ln, "CASE-same", cell.tag, 2, (pa.pid, pb.pid), None, (pa, pb), {}
                ):
                    got += 1
            elif cell.op == "BOX":
                d = cast(int, cell.d)
                cands = [p for p in parts if len(p.premises) >= d and 3 <= p.L <= 7]
                if not cands:
                    continue
                part = rng.choice(cands)
                boxed = _join_box(part, d)
                if boxed is None:
                    continue
                seq, body, ln = boxed
                if consider(
                    seq, body, ln, "BOX", cell.tag, 1, (part.pid,), part.L, (part,), {"d": d}
                ):
                    got += 1
        if got < min_per_cell:
            drops[f"short_cell:{cell.tag}"] = got

    stats: dict[str, Any] = {
        "n_parts_pool": len(pool_problems),
        "n_parts_eligible": len(eligible),
        "gate_model": gate_model,
        "cells": cast(JSON, dict(cell_counts)),
        "short_cells": cast(
            JSON,
            {
                c.tag: cell_counts.get(c.tag, 0)
                for c in _cells()
                if cell_counts.get(c.tag, 0) < min_per_cell
            },
        ),
        "drops": cast(JSON, dict(drops)),
        "n_items": len(problems),
    }
    return problems, stats


# ----------------------------------------------------------------------------- build_renamed_parts


def build_renamed_parts(
    cfg: Mapping[str, Any],
    *,
    certifier: Certifier | None,
    excluder: Excluder | None,
    rng_seed: int,
) -> tuple[list[Problem], dict[str, Any]]:
    """Each glue item's parts as standalone theorems, with the atom names they carry inside the
    joined theorem (audit 2026-10-01: AND/CHAIN/CASE joins rename part atoms, so a glue item's
    compositionality ratio R needs the renamed part's pass rate in its denominator, not the
    original pool item's).

    The stored glue suite records only the original part pids, so this replays `build_joined` with
    the glue suite's own manifest config and seed (`replay=`) and captures the renamed `_Part`s.
    One `Problem` per (glue item, part index): pid `<glue pid>__p<i>`, length label copied from the
    original pool part (renaming is a bijection on atoms, checked below, so minimal length is
    unchanged), tags `glue_pid`/`part_index`/`orig_pid`/`op`/`renamed`. BOX parts are not renamed
    and come out identical to the pool items; they are kept as a same-sequent replication check.

    `cfg` keys: `store_root`, `glue_suite` (default `"glue"`), `glue_version` (default `"v1"`),
    `version` (default `"v1"`). `certifier`/`excluder`/`rng_seed` are unused (the glue manifest's
    seed drives the replay).
    """
    del certifier, excluder, rng_seed
    store = Store(Path(cast(str, cfg.get("store_root", DEFAULT_STORE_ROOT))))
    glue_suite = cast(str, cfg.get("glue_suite", "glue"))
    glue_version = cast(str, cfg.get("glue_version", "v1"))
    version = cast(str, cfg.get("version", "v1"))
    manifest_path = store.suites_dir / glue_suite / glue_version / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    build_cfg = dict(cast("Mapping[str, Any]", manifest["config"]))
    build_cfg["store_root"] = str(store.root)
    target = store.load_suite(glue_suite, glue_version)

    captured: list[tuple[Problem, tuple[_Part, ...]]] = []
    replayed, _stats = build_joined(
        build_cfg,
        certifier=None,
        excluder=None,
        rng_seed=int(cast(int, manifest["seed"])),
        replay=target,
        on_accept=lambda prob, used: captured.append((prob, tuple(used))),
    )
    if [p.pid for p in replayed] != [p.pid for p in target]:
        raise RuntimeError(
            f"replay of {glue_suite}/{glue_version} diverged: rebuilt {len(replayed)} of "
            f"{len(target)} items"
        )

    pool = {
        p.pid: p
        for p in store.load_suite(
            cast(str, build_cfg.get("parts_suite", "parts_pool")),
            cast(str, build_cfg.get("parts_version", "v1")),
        )
    }
    nd_verify = load_nd_verify()
    out: list[Problem] = []
    n_renamed = 0
    for glue_p, used in captured:
        for i, part in enumerate(used):
            orig = pool[part.pid]
            orig_part = to_part(orig)
            if len(part.atoms) != len(orig_part.atoms):
                raise RuntimeError(f"{glue_p.pid} part {i}: renaming is not a bijection on atoms")
            seq = Sequent(
                premises=tuple(F.render(f) for f in part.premises),
                conclusion=F.render(part.concl),
            )
            body = _render_lines(part.lines)
            ok, reason, _nl = nd_verify.verify_text(seq.prompt() + " " + body)
            if not ok:
                raise RuntimeError(f"{glue_p.pid} part {i}: renamed proof fails ({reason})")
            renamed = (seq.premises, seq.conclusion) != (
                tuple(orig.sequent.premises),
                orig.sequent.conclusion,
            )
            n_renamed += int(renamed)
            out.append(
                Problem(
                    pid=f"{glue_p.pid}__p{i}",
                    suite="glue_parts_renamed",
                    version=version,
                    sequent=seq,
                    length=orig.length,
                    ref=P.make_reference(body, source="renamed_part"),
                    tags={
                        "glue_pid": glue_p.pid,
                        "part_index": i,
                        "orig_pid": part.pid,
                        "op": glue_p.tags.get("op"),
                        "renamed": renamed,
                    },
                    parts=(part.pid,),
                )
            )
    stats: dict[str, Any] = {
        "glue_suite": glue_suite,
        "glue_version": glue_version,
        "n_glue_items": len(captured),
        "n_items": len(out),
        "n_renamed": n_renamed,
    }
    return out, stats


# ------------------------------------------------------------------------------ build_length_sweep


def build_length_sweep(
    cfg: Mapping[str, Any],
    *,
    certifier: Certifier | None = None,
    excluder: Excluder | None = None,
    rng_seed: int = 0,
) -> tuple[list[Problem], dict[str, Any]]:
    """Re-wrap the existing matched AND/CHAIN items (`ab3/sealed/matched_L6-14.jsonl`) as an extra
    glue length sweep (plan's WP4 row). Not registered in `suites.BUILDERS` (that registry is
    WP0-owned; see the WP4 report's deviations) -- call directly, e.g. from a build script.

    Each row already carries a `minlen`-certified label (`label_kind == "exact"`, i.e. `n_lines` /
    `L_true` is the true minimum -- distinct from `constructed_lines`/`constructed_proof`, the
    naive join witness, which is often *longer*, exactly the plan §0 shortcut phenomenon; we use
    the certified `minlen_proof`/`n_lines` as the item's proof and length, not the constructed
    one). `certifier` is accepted for signature uniformity but unused (this data was already
    certified upstream); every row is still self-checked with `nd_verify` and passed through
    `excluder` (drop policy, like `build_joined`), and dropped/counted on failure.
    """
    del certifier, rng_seed
    path = Path(cast(str, cfg.get("path", DEFAULT_MATCHED_PATH)))
    version = cast(str, cfg.get("version", "v1"))
    nd_verify = load_nd_verify()
    problems: list[Problem] = []
    drops: collections.Counter[str] = collections.Counter()
    seen: set[str] = set()
    for row in read_jsonl(path):
        prompt = cast(str, row["prompt"])
        seq = Sequent.from_prompt(prompt)
        body = cast(str, row["minlen_proof"])
        n_lines = cast(int, row["n_lines"])
        ok, _reason, nl = nd_verify.verify_text(prompt + " " + body)
        if not (ok and nl == n_lines):
            drops["self_check_fail"] += 1
            continue
        dedup_key = seq.pkey() if len(seq.premises) <= 5 else seq.key()
        if dedup_key in seen:
            drops["duplicate"] += 1
            continue
        if _excluded(seq, excluder):
            drops["excluded"] += 1
            continue
        seen.add(dedup_key)
        family = cast(str, row.get("family", ""))
        op = "AND" if family == "andi" else "CHAIN"
        parts = tuple(cast(list[str], row.get("parts", ())))
        label = LengthLabel(
            L=n_lines,
            L_ub=n_lines,
            L_lb=n_lines,
            method="minlen_exact",
            bound=cast("int | None", row.get("minlen_bound")),
        )
        ref = ReferenceProof(
            body=body, lines=n_lines, pruned=P.pruned_length(body), source="minlen"
        )
        tags: dict[str, JSON] = {
            "op": op,
            "k": len(parts),
            "part_len": cast(JSON, row.get("part_L")),
            "cell": "length_sweep",
            "orig_name": cast(str, row.get("orig_name")),
            "family": family,
            "constructed_lines": cast("int | None", row.get("constructed_lines")),
        }
        problems.append(
            Problem(
                pid=f"gluesweep_{row['name']}",
                suite="glue_length_sweep",
                version=version,
                sequent=seq,
                length=label,
                ref=ref,
                tags=tags,
                parts=parts,
            )
        )
    stats: dict[str, Any] = {"n_items": len(problems), "drops": cast(JSON, dict(drops))}
    return problems, stats
