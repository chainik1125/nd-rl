"""WP0 acceptance test: `NDJudge` agrees with `nd_verify` + `ndcheck` on 60 fixture proofs plus
200 programmatic corruptions (plan §3 WP0 row).

The ground truth is computed directly from the vendored `nd_verify`/`ndcheck` modules (loaded
independently of `judge.nd`'s own loader calls, though both resolve to the same vendored files),
so this is a real cross-check of `NDJudge`'s wiring, not a tautology.
"""

from __future__ import annotations

from typing import Any

from code.ndbench.core.types import Attempt, Sequent
from code.ndbench.judge.nd import NDJudge
from code.ndbench.vendor import load_absgpt_module, load_ndcheck

from ._fixtures import generate_corruptions, load_valid_proofs

judge = NDJudge()


def _expected(prompt: str, body: str) -> dict[str, Any]:
    ndcheck = load_ndcheck()
    return ndcheck.classify(prompt, body)


def _assert_matches_ground_truth(prompt: str, body: str) -> None:
    ndcheck = load_ndcheck()
    prune = load_absgpt_module("prune")
    expected = ndcheck.classify(prompt, body)
    seq = Sequent.from_prompt(prompt)
    attempt = Attempt(
        pid="x", i=0, body=body, native=None, n_tokens=len(body.split()), finished=True
    )
    (verdict,) = judge.judge(seq, [attempt], None)

    assert verdict.ok == expected["ok"]
    assert verdict.kind == expected["kind"]
    assert verdict.reason == expected["reason"]
    assert verdict.written == expected["n_lines"]
    assert verdict.start == expected["start"]
    expected_first_bad = expected["line"] if expected["kind"] == "bad_line" else None
    assert verdict.first_bad_line == expected_first_bad
    if verdict.ok:
        assert verdict.pruned == prune.pruned_length(prompt, body)
    else:
        assert verdict.pruned is None


def test_ndjudge_agrees_on_60_valid_fixture_proofs() -> None:
    proofs = load_valid_proofs()
    assert len(proofs) == 60
    for p in proofs:
        _assert_matches_ground_truth(p["prompt"], p["proof"])
        # These are minlen-verified proofs: nd_verify must accept every one.
        expected = _expected(p["prompt"], p["proof"])
        assert expected["ok"] is True
        assert expected["kind"] == "ok"


def test_ndjudge_agrees_on_200_corruptions() -> None:
    proofs = load_valid_proofs()
    corruptions = generate_corruptions(proofs, n=200, seed=20260930)
    assert len(corruptions) == 200
    saw_kinds: set[str] = set()
    for proof, corrupted_body in corruptions:
        _assert_matches_ground_truth(proof["prompt"], corrupted_body)
        saw_kinds.add(_expected(proof["prompt"], corrupted_body)["kind"])
    # Sanity: the corruption generator actually produces more than one failure mode, not just
    # one degenerate case every time.
    assert len(saw_kinds) >= 2


def test_ndjudge_handles_unparseable_marker() -> None:
    seq = Sequent(premises=("P",), conclusion="P")
    attempt = Attempt(
        pid="x",
        i=0,
        body="UNPARSEABLE decoder produced invalid tokens",
        native="garbage",
        n_tokens=3,
        finished=True,
    )
    (verdict,) = judge.judge(seq, [attempt], None)
    assert verdict.ok is False
    assert verdict.kind == "unparseable"
    assert verdict.written is None
    assert verdict.pruned is None
    assert verdict.first_bad_line is None
    assert verdict.start is None
    assert "decoder produced invalid tokens" in verdict.reason
