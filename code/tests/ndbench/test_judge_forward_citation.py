"""NDJudge rejects circular proofs that the vendored nd_verify accepts (forward box citations)."""

from code.ndbench.core.types import Attempt, Sequent
from code.ndbench.judge.nd import NDJudge, first_forward_citation
from code.ndbench.vendor import load_nd_verify

CIRCULAR = (
    "N1 ( P > Q ) : IMPI N2 N3 ; N2 | P : AS ; N3 | Q : IMPE N1 N2 ; N4 ( P > Q ) : R N1 ; QED"
)
NORMAL = "N1 | P : AS ; N2 ( P > P ) : IMPI N1 N1 ; QED"


def _attempt(body: str) -> Attempt:
    return Attempt(pid="t", i=0, body=body, native=None, n_tokens=0, finished=True)


def test_vendored_verifier_accepts_the_circular_proof() -> None:
    """Documents the bug the judge works around; if this ever fails, the vendored copy changed."""
    ok, _, _ = load_nd_verify().verify_text("THM SEQ ( P > Q ) PRF " + CIRCULAR)
    assert ok


def test_judge_rejects_forward_citation() -> None:
    (v,) = NDJudge().judge(Sequent((), "( P > Q )"), [_attempt(CIRCULAR)], None)
    assert not v.ok
    assert v.reason == "forward_citation"
    assert v.first_bad_line == 1


def test_judge_accepts_a_normal_proof() -> None:
    (v,) = NDJudge().judge(Sequent((), "( P > P )"), [_attempt(NORMAL)], None)
    assert v.ok
    assert first_forward_citation(NORMAL) is None
