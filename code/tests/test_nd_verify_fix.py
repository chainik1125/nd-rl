"""The shared nd_verify rejects forward (circular) box citations and otherwise is unchanged."""

import difflib
from pathlib import Path

import pytest

from code.shared_code.nd_verify import verify_text

ROOT = Path(__file__).resolve().parents[1]
FIXED = ROOT / "shared_code" / "nd_verify" / "verify.py"
FROZEN = ROOT / "experiments" / "current" / "r1_cot" / "vendor" / "nd_verify" / "verify.py"

VALID = [
    "THM ( P > Q ) , P SEQ Q PRF N1 ( P > Q ) : PR ; N2 P : PR ; N3 Q : IMPE N1 N2 ; QED",
    "THM SEQ ( P > P ) PRF N1 | P : AS ; N2 ( P > P ) : IMPI N1 N1 ; QED",
    "THM ( P v Q ) , ( ~ P ) SEQ Q PRF N1 ( P v Q ) : PR ; N2 ( ~ P ) : PR ; N3 | P : AS ; "
    "N4 | F : NEGE N3 N2 ; N5 | Q : BOTE N4 ; N6 | Q : AS ; N7 Q : ORE N1 N3 N5 N6 N6 ; QED",
    # same proof, starting at N12 (any start index >= 1 is allowed)
    "THM ( P > Q ) , P SEQ Q PRF N12 ( P > Q ) : PR ; N13 P : PR ; N14 Q : IMPE N12 N13 ; QED",
]

CIRCULAR = [
    # |- P > Q: line 1 discharges the later box, which uses line 1
    "THM SEQ ( P > Q ) PRF N1 ( P > Q ) : IMPI N2 N3 ; N2 | P : AS ; N3 | Q : IMPE N1 N2 ; "
    "N4 ( P > Q ) : R N1 ; QED",
    # forward box citation without circularity is also outside the spec
    "THM SEQ ( P > P ) PRF N1 ( P > P ) : IMPI N2 N2 ; N2 | P : AS ; N3 ( P > P ) : R N1 ; QED",
]


@pytest.mark.parametrize("text", VALID)
def test_valid_proofs_still_verify(text: str) -> None:
    assert verify_text(text)[0]


@pytest.mark.parametrize("text", CIRCULAR)
def test_forward_box_citations_are_rejected(text: str) -> None:
    assert not verify_text(text)[0]


def test_only_box_citable_changed_from_the_original() -> None:
    """The fix adds lines to `box_citable` and changes nothing else."""
    if not FROZEN.exists():
        pytest.skip("frozen r1_cot copy not present")
    diff = [
        d
        for d in difflib.unified_diff(
            FROZEN.read_text().splitlines(), FIXED.read_text().splitlines(), lineterm="", n=0
        )
        if d[:1] in "+-" and not d.startswith(("+++", "---"))
    ]
    assert diff, "expected the fix to differ from the original"
    assert all(d.startswith("+") for d in diff), diff
    assert any("if e >= i:" in d for d in diff)
