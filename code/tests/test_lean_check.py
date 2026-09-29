"""Tests for ``code.shared_code.lean_check``: the error-line mapping and grammar rejections run
everywhere; the Lean verdicts need Lean 4 (``--runslow``, skipped when Lean is not installed)."""

import json
from pathlib import Path

import pytest

from code.shared_code.lean_check import (
    Verdict,
    check_theorems,
    failing_theorems,
    find_lean,
    judge,
)
from code.shared_code.lean_seq import LeanSeqTokenizer

FIXTURE = Path(__file__).parent / "fixtures" / "lean_seq_golden.jsonl"
GOLDEN: list[dict[str, str]] = [
    json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()[1:]
]
TOK = LeanSeqTokenizer()
NO_LEAN = "/nonexistent/lean"


def ids_of(tokens: str) -> list[int]:
    return [TOK.stoi[t] for t in tokens.split()] + [TOK.eos_id]


def test_errors_map_to_theorems() -> None:
    out = (
        "c_0.lean:3:40: error: type mismatch\n"
        "c_0.lean:4:2: warning: unused variable\n"
        "c_0.lean:8:12: error: unknown identifier 'n9'\n"
    )
    assert failing_theorems(out, [3, 4, 5, 8]) == ({0, 3}, False)
    assert failing_theorems("c_0.lean:1:0: error: bad option\n", [3, 4]) == (set(), True)
    assert failing_theorems("", [3]) == (set(), False)
    # Lean 4.34 writes `sorry` in backticks; older versions used quotes.
    for sorry in ("`sorry`", "'sorry'"):
        out = f"c_0.lean:4:8: warning: declaration uses {sorry}\n"
        assert failing_theorems(out, [3, 4]) == ({1}, False)


def test_grammar_rejections_need_no_lean() -> None:
    prompt = "THM P SEQ P PRF"
    got = judge(
        TOK, [prompt, prompt], [ids_of("have n1 : P := n2 ; exact n1"), [TOK.stoi["exact"]]]
    )
    assert got == [
        Verdict(False, "lean_seq parse: unbound", "LEANPARSE unbound"),
        Verdict(False, "lean_seq parse: no-eos", "LEANPARSE no-eos"),
    ]


def test_input_checks() -> None:
    assert check_theorems([], lean=NO_LEAN) == []
    with pytest.raises(ValueError):
        check_theorems(["example : True := trivial"], lean=NO_LEAN)
    with pytest.raises(ValueError):
        judge(TOK, ["THM P SEQ P PRF"], [])


def _needs_lean() -> None:
    if find_lean() is None:
        pytest.skip("Lean 4 is not installed")


@pytest.mark.slow
def test_judge_accepts_every_golden_proof_and_rejects_swaps() -> None:
    _needs_lean()
    prompts = [rec["prompt"] for rec in GOLDEN]
    samples = [TOK.encode_proof(rec["proof"]) for rec in GOLDEN]
    # Negative control in the same batch: each proof against the next record's theorem, where the
    # statements differ, so a wrong error-line mapping would show up as a false accept or reject.
    swapped = [
        (prompts[(i + 1) % len(GOLDEN)], samples[i])
        for i in range(len(GOLDEN))
        if TOK.statement(prompts[(i + 1) % len(GOLDEN)]) != TOK.statement(prompts[i])
    ]
    verdicts = judge(TOK, prompts + [p for p, _ in swapped], samples + [s for _, s in swapped])
    assert [v.ok for v in verdicts[: len(GOLDEN)]] == [True] * len(GOLDEN)
    assert [v.nd for v in verdicts[: len(GOLDEN)]] == [rec["proof"] for rec in GOLDEN]
    assert len(swapped) > 50
    assert not any(v.ok for v in verdicts[len(GOLDEN) :])
    assert {v.reason for v in verdicts[len(GOLDEN) :]} == {"lean rejected"}


@pytest.mark.slow
def test_lean_decides_not_nd() -> None:
    _needs_lean()
    # ~P |- P > Q by BOTE on the ~P line: ND's BOTE needs a line that is literally F, but Lean
    # reads `n1.elim` as `Not.elim n1 : P → Q`, a valid proof (the fork's canonical example).
    prompts = ["THM ( ~ P ) SEQ ( P > Q ) PRF", "THM P SEQ Q PRF"]
    proofs = ["N1 ( ~ P ) : PR ; N2 ( P > Q ) : BOTE N1 ; QED", "N1 P : PR ; N2 Q : R N1 ; QED"]
    verdicts = judge(TOK, prompts, [TOK.encode_proof(p) for p in proofs])
    assert [v.ok for v in verdicts] == [True, False]


@pytest.mark.slow
def test_sorry_and_broken_theorems_fail_alone() -> None:
    _needs_lean()
    good = TOK.statement(GOLDEN[0]["prompt"]) + " " + GOLDEN[0]["text"]
    sorry = "theorem t : False := by exact sorry"
    broken = "theorem t : False := by exact ("
    got = check_theorems([good, sorry, good, broken, good], workers=1)
    assert got == [True, False, True, False, True]
