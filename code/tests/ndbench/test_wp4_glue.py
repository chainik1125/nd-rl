"""`suites/glue.py` (plan §2 B1 / WP4): each operator's witness verifies at the predicted length,
the plan §0 chain-glue shortcut is rejected by the certification gate, CASE items need `ORE`
(checked with a self-contained truth-table stub -- plan §2 B3's `case_split` recipe -- not WP2's
`logic.skills`), and `build_joined` runs end to end against a fake `Store` with synthetic mastery
rows. Uses the 60 verified fixture proofs (`_fixtures.load_valid_proofs`) as the parts pool, plus a
handful of hand-built single/two-atom parts where the fixture pool's atom-count distribution makes
a path (AND k=4, CHAIN k=3) too rare to hit reliably by random search alone."""

from __future__ import annotations

import itertools
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

from code.ndbench.core import formula as F
from code.ndbench.core.io import utc_now_iso
from code.ndbench.core.types import (
    ItemResult,
    LengthLabel,
    Problem,
    ReferenceProof,
    Sequent,
    SuiteManifest,
)
from code.ndbench.store import Store
from code.ndbench.suites import glue
from code.ndbench.vendor import ensure_takehome_on_path, load_nd_verify

from ._fixtures import load_valid_proofs

ND_VERIFY = load_nd_verify()


# --------------------------------------------------------------------------------- pool helpers


def _mk_problem(
    pid: str, seq: Sequent, body: str, length: int, *, suite: str = "parts_pool"
) -> Problem:
    ref = ReferenceProof(body=body, lines=length, pruned=0, source="minlen")
    label = LengthLabel(L=length, L_ub=length, L_lb=length, method="minlen_exact", bound=None)
    return Problem(pid=pid, suite=suite, version="v1", sequent=seq, length=label, ref=ref, tags={})


def _fixture_pool() -> list[Problem]:
    rows = load_valid_proofs()
    return [
        _mk_problem(f"fx{i}", Sequent.from_prompt(r["prompt"]), r["proof"], r["n_lines"])
        for i, r in enumerate(rows)
    ]


@pytest.fixture
def fixture_parts() -> list[glue._Part]:
    return [glue.to_part(p) for p in _fixture_pool()]


def _verify(seq: Sequent, body: str, expect_l: int) -> None:
    ok, reason, nl = ND_VERIFY.verify_text(seq.prompt() + " " + body)
    assert ok and nl == expect_l == body.count(";"), (reason, nl, expect_l)


# --------------------------------------------------------------------------------- parsing


def test_to_part_recovers_premises_and_length(fixture_parts: list[glue._Part]) -> None:
    for part in fixture_parts:
        assert len(part.lines) == part.L
        assert part.lines[len(part.premises) :] or len(part.premises) == part.L
        assert set(part.atoms) <= set(F.ATOMS)


# --------------------------------------------------------------------------------- AND


def test_join_and_k2_matches_witness_formula(fixture_parts: list[glue._Part]) -> None:
    by_len = glue._index_by_len(fixture_parts)
    rng = random.Random(0)
    successes = 0
    for _ in range(300):
        lens = glue._random_split(11, 2, by_len, rng)
        if lens is None:
            continue
        chosen = glue._try_and(by_len, lens, rng)
        if chosen is None:
            continue
        seq, body, ln = glue._join_and(chosen, rng)
        _verify(seq, body, ln)
        assert ln == sum(p.L for p in chosen) + 1  # AND k=2: sum(Li) + k - 1
        successes += 1
    assert successes >= 5, "expected at least a few AND k=2 constructions from the fixture pool"


def test_join_and_k3_and_k4_with_single_atom_parts() -> None:
    """The 60-fixture pool has only two single-atom parts, so AND k=4 (every part must have
    <= 1 atom: 4 parts x 1 atom = the whole budget) essentially never succeeds by random search
    over it (confirmed empirically). Use hand-built single-atom parts instead to exercise the k=3/
    k=4 atom-budget path directly."""
    atoms = ("P", "Q", "R", "S")
    parts = []
    for i, a in enumerate(atoms):
        body = f"N1 {a} : PR ; N2 {a} : R N1 ; QED"
        seq = Sequent(premises=(a,), conclusion=a)
        parts.append(glue.to_part(_mk_problem(f"single{i}", seq, body, 2)))
    rng = random.Random(0)
    for k in (3, 4):
        chosen = glue._try_and({2: parts}, [2] * k, rng)
        assert chosen is not None, f"AND k={k} should succeed with {k} disjoint single-atom parts"
        seq, body, ln = glue._join_and(chosen, rng)
        _verify(seq, body, ln)
        assert ln == 2 * k + (k - 1)
        assert len({a for p in chosen for a in p.atoms}) == k  # atom budget fully disjoint


# --------------------------------------------------------------------------------- CHAIN


def test_join_chain_k2_matches_witness_formula(fixture_parts: list[glue._Part]) -> None:
    by_len = glue._index_by_len(fixture_parts)
    premise_idx = glue._index_premises_by_shape(fixture_parts)
    rng = random.Random(0)
    successes = 0
    for _ in range(1500):
        lens = glue._random_split(11, 2, by_len, rng)
        if lens is None:
            continue
        result = glue._try_chain(by_len, premise_idx, lens, rng)
        if result is None:
            continue
        parts, link_indices = result
        seq, body, ln = glue._join_chain(parts, link_indices, rng)
        _verify(seq, body, ln)
        assert ln == sum(p.L for p in parts) - 1  # CHAIN k=2: sum(Li) - (k-1)
        successes += 1
    assert successes >= 3, "expected at least a few CHAIN k=2 constructions from the fixture pool"


def test_join_chain_k3_with_hand_built_interface_parts() -> None:
    """A 3-hop chain needs three parts whose interfaces line up (producer's conclusion shape ==
    the next part's chosen premise shape) *and* fit the 4-atom budget end to end; the fixture pool
    essentially never has this by chance (confirmed empirically: 0/2000 random attempts). Use
    hand-built parts with a bare-atom interface (every bare atom shares one shape) instead."""
    part_a = glue.to_part(
        _mk_problem(
            "a", Sequent(premises=("P",), conclusion="P"), "N1 P : PR ; N2 P : R N1 ; QED", 2
        )
    )
    part_b = glue.to_part(
        _mk_problem(
            "b",
            Sequent(premises=("P", "( P > Q )"), conclusion="Q"),
            "N1 P : PR ; N2 ( P > Q ) : PR ; N3 Q : IMPE N2 N1 ; QED",
            3,
        )
    )
    part_c = glue.to_part(
        _mk_problem(
            "c",
            Sequent(premises=("P", "( P > Q )"), conclusion="Q"),
            "N1 P : PR ; N2 ( P > Q ) : PR ; N3 Q : IMPE N2 N1 ; QED",
            3,
        )
    )
    by_len: dict[int, list[glue._Part]] = {2: [part_a], 3: [part_b, part_c]}
    premise_idx = glue._index_premises_by_shape([part_a, part_b, part_c])
    rng = random.Random(0)
    result = None
    for _ in range(200):
        result = glue._try_chain(by_len, premise_idx, [2, 3, 3], rng)
        if result is not None:
            break
    assert result is not None, "CHAIN k=3 should succeed with a hand-built 3-hop interface chain"
    parts, link_indices = result
    seq, body, ln = glue._join_chain(parts, link_indices, rng)
    _verify(seq, body, ln)
    assert ln == sum(p.L for p in parts) - 2  # CHAIN k=3: sum(Li) - (k-1)
    assert len({a for p in parts for a in p.atoms}) <= 4


# --------------------------------------------------------------------------------- CASE


def test_join_case_dilemma_matches_witness_formula(fixture_parts: list[glue._Part]) -> None:
    by_len = glue._index_by_len(fixture_parts)
    rng = random.Random(0)
    successes = 0
    for ell in range(2, 8):
        for _ in range(100):
            found = glue._try_case_dilemma(by_len, ell, rng)
            if found is None:
                continue
            pa, ai, pb, bi = found
            seq, body, ln = glue._join_case(pa, ai, pb, bi, same=False)
            _verify(seq, body, ln)
            assert ln == pa.L + pb.L + 4
            successes += 1
    assert successes >= 10


def test_join_case_same_matches_witness_formula(fixture_parts: list[glue._Part]) -> None:
    concl_idx = glue._index_concl_by_shape(fixture_parts)
    rng = random.Random(0)
    successes = 0
    for _ in range(2000):
        found = glue._try_case_same(concl_idx, rng)
        if found is None:
            continue
        pa, ai, pb, bi = found
        seq, body, ln = glue._join_case(pa, ai, pb, bi, same=True)
        _verify(seq, body, ln)
        # witness L = L1+L2+2 minus any shared non-case premises (dedup); never more than +2.
        assert ln <= pa.L + pb.L + 2
        successes += 1
    assert successes >= 10


# --------------------------------------------------------------------------------- BOX


def test_join_box_matches_witness_formula(fixture_parts: list[glue._Part]) -> None:
    successes = 0
    for d in (1, 2, 3):
        for part in fixture_parts:
            if len(part.premises) < d:
                continue
            result = glue._join_box(part, d)
            if result is None:
                continue
            seq, body, ln = result
            _verify(seq, body, ln)
            assert ln == part.L + d
            assert len(seq.premises) == len(part.premises) - d
            successes += 1
    assert successes >= 10


def test_join_box_rejects_insufficient_premises() -> None:
    part = glue.to_part(
        _mk_problem(
            "p", Sequent(premises=("P",), conclusion="P"), "N1 P : PR ; N2 P : R N1 ; QED", 2
        )
    )
    assert glue._join_box(part, 2) is None  # only 1 premise, need >= 2


# --------------------------------------------------------------- certification gate (real minlen)


class _VendorMinlenCertifier:
    """Wraps the vendored `takehome.minlen.minlen` directly as a `Certifier` (per the WP4 task:
    "In tests use stubs or the vendored minlen directly; do not import WP2 modules"). Only used
    for the two tests below, which need a real minlen search (not a rubber-stamp stub) to
    demonstrate the plan §0 chain-glue shortcut is actually caught."""

    def certify(
        self,
        seqs: Sequence[Sequent],
        witnesses: Sequence[str | None],
        *,
        bound: int | None = None,
        time_s: float = 60.0,
    ) -> list[LengthLabel]:
        ensure_takehome_on_path()
        import minlen as vendor_minlen  # vendored takehome/minlen.py

        out = []
        for seq, body in zip(seqs, witnesses, strict=True):
            assert body is not None
            b = bound if bound is not None else body.count(";")
            r = vendor_minlen.minlen(seq.prompt(), bound=b, time_limit=time_s)
            if r["timeout"]:
                out.append(
                    LengthLabel(L=None, L_ub=None, L_lb=None, method="ub_lb", bound=b, timeout=True)
                )
            elif r["min_lines_ub"] is not None:
                n = cast(int, r["min_lines_ub"])
                out.append(LengthLabel(L=n, L_ub=n, L_lb=n, method="minlen_exact", bound=b))
            else:
                out.append(LengthLabel(L=None, L_ub=None, L_lb=None, method="ub_lb", bound=b))
        return out


def test_chain_shortcut_example_is_rejected() -> None:
    """Plan §0's exact example: `(P&Q)|-(Q&P)` then `(Q&P),(Q>R)|-R` glues to a naive 7-line
    witness, but `(P&Q),(Q>R)|-R` has a genuine 4-line proof (ANDE2 bypasses the ANDI/ANDE1
    round-trip) -- the certification gate must reject it."""
    producer = glue.to_part(
        _mk_problem(
            "producer",
            Sequent(premises=("( P & Q )",), conclusion="( Q & P )"),
            "N1 ( P & Q ) : PR ; N2 Q : ANDE2 N1 ; N3 P : ANDE1 N1 ; "
            "N4 ( Q & P ) : ANDI N2 N3 ; QED",
            4,
        )
    )
    consumer = glue.to_part(
        _mk_problem(
            "consumer",
            Sequent(premises=("( Q & P )", "( Q > R )"), conclusion="R"),
            "N1 ( Q & P ) : PR ; N2 ( Q > R ) : PR ; N3 Q : ANDE1 N1 ; N4 R : IMPE N2 N3 ; QED",
            4,
        )
    )
    seq, body, ln = glue._join_chain([producer, consumer], [-1, 0], random.Random(0))
    assert ln == 7  # the naive witness length from plan §0
    _verify(seq, body, ln)
    label = _VendorMinlenCertifier().certify([seq], [body], bound=ln, time_s=10.0)[0]
    assert label.method == "minlen_exact" and label.L == 4  # the true shortcut minlen finds
    assert not glue._accepts(label, ln), "the chain-glue shortcut must be rejected"


def test_certifier_accepts_a_genuine_and_join() -> None:
    part1 = glue.to_part(
        _mk_problem(
            "p1",
            Sequent(premises=("( P & Q )",), conclusion="( Q & P )"),
            "N1 ( P & Q ) : PR ; N2 Q : ANDE2 N1 ; N3 P : ANDE1 N1 ; "
            "N4 ( Q & P ) : ANDI N2 N3 ; QED",
            4,
        )
    )
    part2 = glue.to_part(
        _mk_problem(
            "p2",
            Sequent(premises=("R", "S"), conclusion="( R & S )"),
            "N1 R : PR ; N2 S : PR ; N3 ( R & S ) : ANDI N1 N2 ; QED",
            3,
        )
    )
    seq, body, ln = glue._join_and([part1, part2], random.Random(2))
    _verify(seq, body, ln)
    label = _VendorMinlenCertifier().certify([seq], [body], bound=ln, time_s=10.0)[0]
    assert glue._accepts(label, ln), (label, ln)


# ------------------------------------------------------------------------- needs_ore truth table


def _collect_or_subformulas(f: F.Formula, acc: set[F.Formula]) -> None:
    if f[0] == "or":
        acc.add(f)
        _collect_or_subformulas(f[1], acc)
        _collect_or_subformulas(f[2], acc)
    elif f[0] == "not":
        _collect_or_subformulas(f[1], acc)
    elif f[0] in ("and", "imp"):
        _collect_or_subformulas(f[1], acc)
        _collect_or_subformulas(f[2], acc)


def _substitute(f: F.Formula, mapping: dict[F.Formula, F.Formula]) -> F.Formula:
    if f in mapping:
        return mapping[f]
    if f[0] in ("atom", "bot"):
        return f
    if f[0] == "not":
        return ("not", _substitute(f[1], mapping))
    return (f[0], _substitute(f[1], mapping), _substitute(f[2], mapping))


def _eval(f: F.Formula, v: dict[str, bool]) -> bool:
    if f[0] == "atom":
        return v[f[1]]
    if f[0] == "bot":
        return False
    if f[0] == "not":
        return not _eval(f[1], v)
    if f[0] == "and":
        return _eval(f[1], v) and _eval(f[2], v)
    if f[0] == "or":
        return _eval(f[1], v) or _eval(f[2], v)
    return (not _eval(f[1], v)) or _eval(f[2], v)


def stub_needs_ore(seq: Sequent, body: str) -> bool:
    """Self-contained implementation of plan §2 B3's `case_split` recipe (soundness/completeness
    argument given there): replace each disjunction subformula `D = X v Y` by a fresh atom `p_D`
    with axioms `X* -> p_D`, `Y* -> p_D`; an ORE-free proof of `Γ |- C` exists iff `Γ* U Ax |= C*`.
    A truth-table check over <= 4 real atoms plus one fresh atom per disjunction found. This is a
    test-only stand-in for WP2's `logic.skills.needs_ore` (not imported here, per the WP4 task)."""
    del body
    premises = [F.parse(p) for p in seq.premises]
    concl = F.parse(seq.conclusion)
    ors: set[F.Formula] = set()
    for f in (*premises, concl):
        _collect_or_subformulas(f, ors)
    if not ors:
        return False
    ordered = sorted(ors, key=str)
    fresh = {f: ("atom", f"D{i}") for i, f in enumerate(ordered)}
    prem_star = [_substitute(p, fresh) for p in premises]
    concl_star = _substitute(concl, fresh)
    axioms: list[F.Formula] = []
    for f, p in fresh.items():
        axioms.append(("imp", _substitute(f[1], fresh), p))
        axioms.append(("imp", _substitute(f[2], fresh), p))
    atoms: set[str] = set()
    for f in (*prem_star, concl_star, *axioms):
        atoms |= F.atoms(f)
    atoms_sorted = sorted(atoms)
    assert len(atoms_sorted) <= 20, "too many atoms for a brute-force truth table"
    for bits in itertools.product((False, True), repeat=len(atoms_sorted)):
        v = dict(zip(atoms_sorted, bits, strict=True))
        if (
            all(_eval(p, v) for p in prem_star)
            and all(_eval(a, v) for a in axioms)
            and not _eval(concl_star, v)
        ):
            return True  # not entailed without ORE -> ORE is genuinely needed
    return False


def test_case_items_need_ore(fixture_parts: list[glue._Part]) -> None:
    """Not every random CASE-dilemma pairing genuinely needs `ORE` -- e.g. if one branch's
    conclusion happens to be a tautology (easy by chance with these generator-drawn parts), the
    disjunction can be sidestepped entirely, and `needs_ore` correctly says so (confirmed by
    inspection of one such case: `((~(~S))>(~(~S))) & ((~S)>(~S))` is a tautology, making its `v
    ...` a tautology too). This is exactly why `build_joined.consider()` filters CASE items through
    `needs_ore` rather than assuming it always holds. This test checks the filter is *meaningful*:
    it is `True` for at least some genuine case splits, and `False` is possible too (checked
    separately below with an OR-free join)."""
    by_len = glue._index_by_len(fixture_parts)
    rng = random.Random(0)
    checked = 0
    needed = 0
    for ell in range(2, 8):
        for _ in range(50):
            found = glue._try_case_dilemma(by_len, ell, rng)
            if found is None:
                continue
            pa, ai, pb, bi = found
            seq, body, _l = glue._join_case(pa, ai, pb, bi, same=False)
            checked += 1
            needed += stub_needs_ore(seq, body)
    assert checked >= 5
    assert needed >= 1, "expected at least one genuine CASE-dilemma join to need ORE"


def test_needs_ore_stub_is_false_for_an_or_free_and_join(fixture_parts: list[glue._Part]) -> None:
    """Sanity check that the stub is not vacuously true: a plain AND join with no disjunction
    anywhere in its premises/conclusion never needs ORE."""
    candidates = [
        p for p in fixture_parts if "or" not in str(p.premises) and "or" not in str(p.concl)
    ]
    assert len(candidates) >= 2
    seq, body, _l = glue._join_and(candidates[:2], random.Random(0))
    assert not stub_needs_ore(seq, body)


# --------------------------------------------------------------------------------- build_joined


def _manifest(n_items: int, suite: str) -> SuiteManifest:
    return SuiteManifest(
        suite=suite,
        version="v1",
        builder="test",
        config={},
        seed=0,
        created_utc=utc_now_iso(),
        git_commit="test",
        n_items=n_items,
        items_sha256="x",
        excl_sources={},
        stats={},
    )


def _write_parts_pool(store: Store, problems: list[Problem]) -> None:
    store.write_suite(problems, _manifest(len(problems), "parts_pool"))


def _write_mastery(store: Store, run_id: str, pids: list[str], p_hat: float) -> None:
    c = round(p_hat * 32)
    rows = [
        ItemResult(
            run_id=run_id,
            pid=pid,
            chunk=0,
            n=32,
            c=c,
            first_success=(0 if c else None),
            ok_written=[],
            ok_pruned=[],
            ok_tokens=[],
            ok_starts=[],
            fail_kinds={},
            fail_lines=[],
            long_enough=0,
            tokens_mean=0.0,
            hit_max=0,
        )
        for pid in pids
    ]
    store.append_items(run_id, rows)


class _AlwaysAcceptCertifier:
    """Rubber-stamp stub `Certifier`: accepts every witness at its own written length. Used for
    the `build_joined` pipeline test, which exercises mastery gating / cell iteration / tagging --
    the certification-rejection logic itself is tested separately, against real minlen, above."""

    def certify(
        self,
        seqs: Sequence[Sequent],
        witnesses: Sequence[str | None],
        *,
        bound: int | None = None,
        time_s: float = 60.0,
    ) -> list[LengthLabel]:
        del time_s
        out = []
        for body in witnesses:
            assert body is not None
            ln = body.count(";")
            out.append(LengthLabel(L=ln, L_ub=ln, L_lb=ln, method="minlen_exact", bound=bound))
        return out


class _AlwaysExcludeExcluder:
    def status(self, seq: Sequent) -> tuple[bool, tuple[str, ...]]:
        del seq
        return True, ()


def test_build_joined_with_fake_store_and_stub_certifier(tmp_path: Path) -> None:
    store = Store(tmp_path)
    parts_pool = _fixture_pool()
    _write_parts_pool(store, parts_pool)
    _write_mastery(store, "run-rl-g1", [p.pid for p in parts_pool], p_hat=1.0)

    cfg = {
        "store_root": str(tmp_path),
        "mastery_run_ids": {"rl": "run-rl-g1"},
        "gate_model": "rl",
        "target_per_cell": 5,
        "min_per_cell": 3,
        "max_tries_per_cell": 150,
    }
    problems, stats = glue.build_joined(
        cfg,
        certifier=_AlwaysAcceptCertifier(),
        excluder=None,
        rng_seed=0,
        needs_ore=stub_needs_ore,
    )

    assert problems, "expected at least some joined items from the fixture pool"
    assert stats["n_parts_eligible"] == len(parts_pool)
    assert "cells" in stats and "short_cells" in stats and "drops" in stats
    for p in problems:
        assert p.suite == "glue"
        for key in ("op", "k", "cell"):
            assert key in p.tags
        assert p.parts, f"{p.pid} should record its source part pids"
        ok, reason, nl = ND_VERIFY.verify_text(p.sequent.prompt() + " " + p.ref.body)  # type: ignore[union-attr]
        assert ok and nl == p.length.L, (p.pid, reason)
        if str(p.tags["op"]).startswith("CASE"):
            assert stub_needs_ore(p.sequent, p.ref.body)  # type: ignore[union-attr]
    # AND k=4 / CHAIN k=3 need atom-count-1 parts the fixture pool barely has; they should show up
    # as short (this exercises the "cell reported short" path, not a hard requirement to fill them).
    assert any(n < cfg["min_per_cell"] for n in stats["cells"].values()) or stats["short_cells"]


def test_build_joined_drops_everything_when_excluded(tmp_path: Path) -> None:
    store = Store(tmp_path)
    parts_pool = _fixture_pool()
    _write_parts_pool(store, parts_pool)
    _write_mastery(store, "run-rl-g1", [p.pid for p in parts_pool], p_hat=1.0)
    cfg = {
        "store_root": str(tmp_path),
        "mastery_run_ids": {"rl": "run-rl-g1"},
        "gate_model": "rl",
        "target_per_cell": 5,
        "min_per_cell": 3,
        "max_tries_per_cell": 60,
    }
    problems, stats = glue.build_joined(
        cfg,
        certifier=_AlwaysAcceptCertifier(),
        excluder=_AlwaysExcludeExcluder(),
        rng_seed=0,
        needs_ore=stub_needs_ore,
    )
    assert problems == []
    assert stats["drops"].get("excluded", 0) > 0


def test_build_joined_without_mastery_run_ids_uses_the_whole_pool(tmp_path: Path) -> None:
    """Documented fallback (module docstring / `build_joined` docstring): with no
    `mastery_run_ids`, every pool part is eligible."""
    store = Store(tmp_path)
    parts_pool = _fixture_pool()
    _write_parts_pool(store, parts_pool)
    cfg = {
        "store_root": str(tmp_path),
        "target_per_cell": 3,
        "min_per_cell": 2,
        "max_tries_per_cell": 60,
    }
    problems, stats = glue.build_joined(
        cfg, certifier=_AlwaysAcceptCertifier(), excluder=None, rng_seed=0, needs_ore=stub_needs_ore
    )
    assert stats["n_parts_eligible"] == len(parts_pool)
    assert problems


# ---------------------------------------------------------------------------- build_length_sweep

# Two real rows copied verbatim (as Python dicts, to keep every line short) from
# /scratch/cowifornia/nd-rl/ab3/sealed/matched_L6-14.jsonl (read-only source; embedded as data so
# this test is hermetic). The chaink2 row is a real instance of the plan §0 shortcut phenomenon:
# `constructed_lines` (the naive join witness) is 16, but the row's own certified
# `n_lines`/`minlen_proof` is 6 -- `build_length_sweep` must use the latter.
_ANDI_PROOF = (
    "N1 S : PR ; N2 Q : PR ; N3 R : PR ; "
    "N4 ( ( ( S v ( ~ ( ~ P ) ) ) v ( S > ( ~ S ) ) ) v S ) : ORI2 N1 ; "
    "N5 ( R v ( ~ Q ) ) : ORI1 N3 ; "
    "N6 ( ( ( ( S v ( ~ ( ~ P ) ) ) v ( S > ( ~ S ) ) ) v S ) & ( R v ( ~ Q ) ) ) "
    ": ANDI N4 N5 ; QED"
)
_ANDI_ROW = json.dumps(
    {
        "name": "ab3_matched_L6-14_24",
        "orig_name": "andi_33401_1734",
        "prompt": (
            "THM S , Q , R SEQ "
            "( ( ( ( S v ( ~ ( ~ P ) ) ) v ( S > ( ~ S ) ) ) v S ) & ( R v ( ~ Q ) ) ) PRF"
        ),
        "n_lines": 6,
        "L_true": 6,
        "source": "ab3_andi",
        "family": "andi",
        "label_kind": "exact",
        "n_prem": 3,
        "minlen_bound": 18,
        "minlen_proof": _ANDI_PROOF,
        "constructed_lines": 6,
        "constructed_proof": _ANDI_PROOF,
        "parts": ["ab2g_123534", "ab2g_26101"],
        "part_L": [2, 3],
    }
)
_CHAIN_MINLEN_PROOF = (
    "N1 ( ~ ( ( ~ ( R v Q ) ) & ( P v ( P v R ) ) ) ) : PR ; N2 Q : PR ; "
    "N3 ( ( ( ~ ( R v Q ) ) & ( P v ( P v R ) ) ) > ( S v ( Q v R ) ) ) : PR ; "
    "N4 ( ( R v P ) v S ) : PR ; N5 ( P v ( ~ Q ) ) : PR ; N6 ( R v Q ) : ORI2 N2 ; QED"
)
_CHAIN_CONSTRUCTED_PROOF = (
    "N1 ( ~ ( ( ~ ( R v Q ) ) & ( P v ( P v R ) ) ) ) : PR ; N2 Q : PR ; "
    "N3 ( ( ( ~ ( R v Q ) ) & ( P v ( P v R ) ) ) > ( S v ( Q v R ) ) ) : PR ; "
    "N4 ( ( R v P ) v S ) : PR ; N5 ( P v ( ~ Q ) ) : PR ; "
    "N6 | P : AS ; N7 | ( ~ Q ) : AS ; N8 | F : NEGE N2 N7 ; N9 | P : BOTE N8 ; "
    "N10 P : ORE N5 N6 N6 N7 N9 ; N11 ( P v ( P v R ) ) : ORI1 N10 ; "
    "N12 | ( ~ ( R v Q ) ) : AS ; N13 | ( ( ~ ( R v Q ) ) & ( P v ( P v R ) ) ) : ANDI N12 N11 ; "
    "N14 | F : NEGE N13 N1 ; N15 ( ~ ( ~ ( R v Q ) ) ) : NEGI N12 N14 ; "
    "N16 ( R v Q ) : DN N15 ; QED"
)
_CHAIN_ROW = json.dumps(
    {
        "name": "ab3_matched_L6-14_5",
        "orig_name": "chaink2_33212_5373",
        "prompt": (
            "THM ( ~ ( ( ~ ( R v Q ) ) & ( P v ( P v R ) ) ) ) , Q , "
            "( ( ( ~ ( R v Q ) ) & ( P v ( P v R ) ) ) > ( S v ( Q v R ) ) ) , "
            "( ( R v P ) v S ) , ( P v ( ~ Q ) ) SEQ ( R v Q ) PRF"
        ),
        "n_lines": 6,
        "L_true": 6,
        "source": "ab3_chain",
        "family": "chaink2",
        "label_kind": "exact",
        "n_prem": 5,
        "minlen_bound": 12,
        "minlen_proof": _CHAIN_MINLEN_PROOF,
        "constructed_lines": 16,
        "constructed_proof": _CHAIN_CONSTRUCTED_PROOF,
        "parts": ["ab2g_508482", "ab2g_460675", "ab2g_278808"],
        "part_L": [3, 7, 8],
    }
)


def test_build_length_sweep_uses_the_certified_minlen_not_the_constructed_witness(
    tmp_path: Path,
) -> None:
    path = tmp_path / "matched_sample.jsonl"
    path.write_text(_ANDI_ROW + "\n" + _CHAIN_ROW + "\n")
    problems, stats = glue.build_length_sweep({"path": str(path)})
    assert stats["n_items"] == 2
    by_op = {p.tags["op"]: p for p in problems}
    assert set(by_op) == {"AND", "CHAIN"}
    chain_item = by_op["CHAIN"]
    assert chain_item.length.L == 6  # the certified minlen, not constructed_lines == 16
    assert chain_item.ref is not None and chain_item.ref.lines == 6
    assert chain_item.tags["constructed_lines"] == 16
    assert chain_item.parts == ("ab2g_508482", "ab2g_460675", "ab2g_278808")
    ok, reason, nl = ND_VERIFY.verify_text(chain_item.sequent.prompt() + " " + chain_item.ref.body)
    assert ok and nl == 6, reason


def test_build_length_sweep_respects_excluder(tmp_path: Path) -> None:
    path = tmp_path / "matched_sample.jsonl"
    path.write_text(_ANDI_ROW + "\n" + _CHAIN_ROW + "\n")
    problems, stats = glue.build_length_sweep(
        {"path": str(path)}, excluder=_AlwaysExcludeExcluder()
    )
    assert problems == []
    assert stats["drops"]["excluded"] == 2
