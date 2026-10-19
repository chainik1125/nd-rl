"""`interp/orderings.py`'s hard requirements (`INTERP_SPEC.md` §1):

* every ordering it produces verifies with the vendored `nd_verify`, checked on the 60 fixture
  proofs (`_fixtures.py`) plus hand-built proofs with independent branches and (nested) boxes;
* `count_orderings` matches a from-scratch brute-force count of valid line permutations on small
  (<=8 line) proofs;
* `sample_orderings` still produces verifying orderings once a level exceeds the exact-DP cutoff.
"""

from __future__ import annotations

import itertools
import random
from typing import Any

import pytest

from code.ndbench.core.proof import split_lines
from code.ndbench.core.types import Sequent
from code.ndbench.interp.orderings import (
    MAX_EXACT_UNITS,
    count_orderings,
    enumerate_orderings,
    sample_orderings,
)
from code.ndbench.vendor import load_nd_verify

from ._fixtures import load_valid_proofs

nd_verify = load_nd_verify()


def _verify(seq: Sequent, body: str) -> tuple[bool, str]:
    ok, reason, _n_lines = nd_verify.verify_text(f"{seq.prompt()} {body}")
    return ok, reason


def _brute_force_count(seq: Sequent, body: str) -> int:
    """From-scratch (does not reuse `interp.orderings`) count of every permutation of `body`'s
    lines that, once renumbered N1.., still verifies -- an independent oracle for `count_orderings`.

    On top of `nd_verify`'s own verdict, this also rejects any permutation citing something at a
    *later* new index than the citing line itself. `nd_verify.verify`'s `line_citable` already
    enforces this explicitly (`1 <= j < i`) for ordinary line citations, but `box_citable` (used by
    `IMPI`/`NEGI`/`ORE`'s box arguments) does not check `s`/`e` against the citer's index at all --
    only that the box is not *currently open* and that its parent context is a prefix of the
    citer's, both of which a *forward*-referenced top-level box (parent context `()`) satisfies
    vacuously. That is a real leniency in the vendored verifier (confirmed empirically: without
    this extra check, brute force finds 24 "valid" permutations of the two-independent-boxes fixture
    below, most of them citing a box before its own `AS`/content lines), not part of the plan's
    definition of "the same up to order" ("keeps every line after every line it cites") -- so this
    oracle enforces the definition's own backward-reference requirement directly, rather than
    relying on `nd_verify` alone to reject forward references the way it does for plain lines.
    """
    lines = split_lines(body)
    prompt = seq.prompt()
    n_ok = 0
    for perm in itertools.permutations(lines):
        remap = {int(line.split()[0][1:]): i + 1 for i, line in enumerate(perm)}
        new_lines = []
        backward_only = True
        for new_idx, line in enumerate(perm, start=1):
            toks = line.split()
            own = int(toks[0][1:])
            assert remap[own] == new_idx
            new_toks = []
            for t in toks[1:]:
                if t[0] == "N" and t[1:].isdigit():
                    ref_new = remap[int(t[1:])]
                    if ref_new >= new_idx:
                        backward_only = False
                    new_toks.append(f"N{ref_new}")
                else:
                    new_toks.append(t)
            new_lines.append(f"N{new_idx} " + " ".join(new_toks))
        if not backward_only:
            continue
        candidate = f"{prompt} {' '.join(new_lines)} QED"
        ok, _reason, _n = nd_verify.verify_text(candidate)
        n_ok += int(ok)
    return n_ok


# --------------------------------------------------------------------------------- 60 fixtures


@pytest.mark.parametrize("row", load_valid_proofs(), ids=lambda r: r["name"])
def test_fixture_orderings_all_verify(row: dict[str, Any]) -> None:
    seq = Sequent.from_prompt(row["prompt"])
    body = row["proof"]
    # the fixture proof itself must verify (sanity: catches a broken fixture before blaming
    # orderings.py for it).
    ok, reason = _verify(seq, body)
    assert ok, f"{row['name']}: fixture proof itself does not verify ({reason})"

    n = count_orderings(body)
    if n is not None and n <= 2000:
        orderings = enumerate_orderings(body, cap=2000)
        assert len(orderings) == n
        assert len(set(orderings)) == n, "enumerate_orderings produced a duplicate"
        for cand in orderings:
            ok, reason = _verify(seq, cand)
            assert ok, f"{row['name']}: an enumerated ordering failed nd_verify ({reason})"
    else:
        rng = random.Random(0)
        for cand, _logq in sample_orderings(body, 20, rng):
            ok, reason = _verify(seq, cand)
            assert ok, f"{row['name']}: a sampled ordering failed nd_verify ({reason})"


@pytest.mark.parametrize(
    "row", [r for r in load_valid_proofs() if r["n_lines"] <= 8][:6], ids=lambda r: r["name"]
)
def test_fixture_small_proofs_count_matches_bruteforce(row: dict[str, Any]) -> None:
    seq = Sequent.from_prompt(row["prompt"])
    body = row["proof"]
    n = count_orderings(body)
    assert n is not None
    assert n == _brute_force_count(seq, body)


# --------------------------------------------------------- hand-built: independent branches


def test_independent_plain_branches_count_and_bruteforce() -> None:
    """Two ANDI lines that each depend on a shared premise but not on each other: exactly 2
    orderings (swap them), verified against brute force over all 6! permutations."""
    seq = Sequent(premises=("P", "Q", "R"), conclusion="( ( P & Q ) & ( Q & R ) )")
    body = (
        "N1 P : PR ; N2 Q : PR ; N3 R : PR ; "
        "N4 ( P & Q ) : ANDI N1 N2 ; N5 ( Q & R ) : ANDI N2 N3 ; "
        "N6 ( ( P & Q ) & ( Q & R ) ) : ANDI N4 N5 ; QED"
    )
    ok, reason = _verify(seq, body)
    assert ok, reason
    assert count_orderings(body) == 2 == _brute_force_count(seq, body)
    orderings = enumerate_orderings(body, cap=10)
    assert len(orderings) == 2
    for cand in orderings:
        ok, reason = _verify(seq, cand)
        assert ok, reason


def test_independent_boxes_count_and_bruteforce() -> None:
    """Two independent top-level boxes, each immediately discharged by its own `IMPI` line, combined
    by a final ANDI: the ROOT's 4 permutable units are `box1, N3 (discharges box1), box2, N4
    (discharges box2)`, with only `box1 < N3` and `box2 < N4` as constraints (the two branches are
    otherwise unrelated) -- 4!/(2!*2!) = 6 interleavings, over 7 lines total."""
    seq = Sequent(premises=(), conclusion="( ( P > P ) & ( Q > Q ) )")
    body = (
        "N1 | P : AS ; N2 | P : R N1 ; N3 ( P > P ) : IMPI N1 N2 ; "
        "N4 | Q : AS ; N5 | Q : R N4 ; N6 ( Q > Q ) : IMPI N4 N5 ; "
        "N7 ( ( P > P ) & ( Q > Q ) ) : ANDI N3 N6 ; QED"
    )
    ok, reason = _verify(seq, body)
    assert ok, reason
    assert count_orderings(body) == 6 == _brute_force_count(seq, body)
    orderings = enumerate_orderings(body, cap=10)
    assert len(orderings) == 6
    assert len(set(orderings)) == 6
    for cand in orderings:
        ok, reason = _verify(seq, cand)
        assert ok, reason


def test_nested_box_independent_branches_count_and_bruteforce() -> None:
    """A box whose own content is two independent 2-line branches (`R`/`ANDI` pairs) that both
    cite the box's own `AS` line (not each other): interleaving them gives 4!/(2!2!) = 6 orderings,
    all inside one box, exercising nested-container edges (a dependency promoted from a line
    directly onto its enclosing box's permutable set, not onto a top-level unit)."""
    seq = Sequent(premises=(), conclusion="( P > ( ( P & P ) & ( P & P ) ) )")
    body = (
        "N1 | P : AS ; "
        "N2 | P : R N1 ; N3 | ( P & P ) : ANDI N1 N2 ; "
        "N4 | P : R N1 ; N5 | ( P & P ) : ANDI N1 N4 ; "
        "N6 | ( ( P & P ) & ( P & P ) ) : ANDI N3 N5 ; "
        "N7 ( P > ( ( P & P ) & ( P & P ) ) ) : IMPI N1 N6 ; QED"
    )
    ok, reason = _verify(seq, body)
    assert ok, reason
    assert count_orderings(body) == 6 == _brute_force_count(seq, body)
    orderings = enumerate_orderings(body, cap=20)
    assert len(orderings) == 6
    assert len(set(orderings)) == 6
    for cand in orderings:
        ok, reason = _verify(seq, cand)
        assert ok, reason


def test_deeply_nested_citation_climbs_two_levels() -> None:
    """`N3` (depth 2) cites `N1` (the outer box's own `AS` line, two containers up): exercises
    `_climb` walking through more than one enclosing box. No reordering freedom (every line forms
    a straight dependency chain), so this is a correctness/crash regression, not a counting test."""
    seq = Sequent(premises=(), conclusion="( P > ( Q > P ) )")
    body = (
        "N1 | P : AS ; N2 | | Q : AS ; N3 | | P : R N1 ; "
        "N4 | ( Q > P ) : IMPI N2 N3 ; N5 ( P > ( Q > P ) ) : IMPI N1 N4 ; QED"
    )
    ok, reason = _verify(seq, body)
    assert ok, reason
    assert count_orderings(body) == 1
    (only,) = enumerate_orderings(body, cap=10)
    assert only == body


# ------------------------------------------------------------- exceeding the exact cutoff


def test_wide_independent_star_exceeds_exact_cutoff_but_still_samples_validly() -> None:
    """23 mutually-independent lines at one level (> `MAX_EXACT_UNITS`): `count_orderings` must
    give up (`None`) rather than attempt a 2**23 DP, and `sample_orderings` must still produce
    verifying orderings."""
    n_branches = MAX_EXACT_UNITS + 1
    branch_lines = " ".join(f"N{i} P : R N1 ;" for i in range(2, 2 + n_branches))
    body = f"N1 P : PR ; {branch_lines} N{2 + n_branches} P : R N2 ; QED"
    seq = Sequent(premises=("P",), conclusion="P")
    ok, reason = _verify(seq, body)
    assert ok, reason
    assert count_orderings(body) is None

    rng = random.Random(0)
    samples = sample_orderings(body, 25, rng)
    assert len(samples) == 25
    for cand, logq in samples:
        assert logq < 0.0  # a genuine product of 1/k factors, k > 1 somewhere
        ok, reason = _verify(seq, cand)
        assert ok, reason


def test_medium_chain_interleave_matches_hand_derivation() -> None:
    """Two independent 4-line chains (no freedom within a chain, free interleaving between them):
    C(8,4) = 70 orderings -- the "about 50-500 orderings" regime `likelihood.marginal_logprob`'s
    importance sampler is checked against (`test_interp_likelihood.py`), also checked here as an
    orderings.py case in its own right."""
    seq = Sequent(premises=("P", "Q"), conclusion="( P & Q )")
    # chain_a: N3 cites N1, then each cites its immediate predecessor (N4<-N3<-...).
    chain_a = "N3 P : R N1 ; N4 P : R N3 ; N5 P : R N4 ; N6 P : R N5 ;"
    # chain_b: N7 cites N2, independent of chain_a throughout.
    chain_b = "N7 Q : R N2 ; N8 Q : R N7 ; N9 Q : R N8 ; N10 Q : R N9 ;"
    body = f"N1 P : PR ; N2 Q : PR ; {chain_a} {chain_b} N11 ( P & Q ) : ANDI N6 N10 ; QED"
    ok, reason = _verify(seq, body)
    assert ok, reason
    assert count_orderings(body) == 70
    orderings = enumerate_orderings(body, cap=1000)
    assert len(orderings) == 70
    assert len(set(orderings)) == 70
    for cand in orderings:
        ok, reason = _verify(seq, cand)
        assert ok, reason
