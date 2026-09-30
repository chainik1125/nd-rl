"""WP7 acceptance tests: `LeanJudge` (plan §3 WP7 row).

The primary acceptance criterion -- "LeanJudge is skipped or raises `NotAvailable` without `lean`"
-- is directly exercisable here: this environment has no Lean 4 install (plan §A2/§A "Lean is not
installed here"). The rendering/batching/dedup/error-parsing pipeline is additionally exercised
against a small *fake* `lean` executable (a local Python script this test writes to `tmp_path` and
marks executable) that mimics the real binary's argv/stdout/return-code contract closely enough for
`LeanJudge`'s own code to be tested without a real Lean 4 install -- not a network resource, and not
a claim about real Lean's actual error-message format.
"""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest

from code.ndbench.core.types import Attempt, Sequent
from code.ndbench.judge.base import NotAvailable
from code.ndbench.judge.lean import LeanJudge, _failing_theorems, find_lean

_SEQ = Sequent(premises=("( P v Q )", "( ~ P )"), conclusion="Q")
_GOOD_PROOF = (
    "N1 ( P v Q ) : PR ; N2 ( ~ P ) : PR ; N3 | P : AS ; N4 | F : NEGE N3 N2 ; "
    "N5 | Q : BOTE N4 ; N6 | Q : AS ; N7 Q : ORE N1 N3 N5 N6 N6 ; QED"
)

_FAKE_LEAN_SRC = f"""\
#!{sys.executable}
import os, sys

mode = os.environ.get("FAKE_LEAN_MODE", "accept")
path = sys.argv[-1]
with open(path) as f:
    text = f.read()

if mode == "accept":
    sys.exit(0)
if mode == "reject":
    # Point at the first theorem's line (right after the 2-line prelude), matching the real
    # `lean` error format `_ERROR_RE` parses: "<path>:<line>:<col>: error: ...".
    print(f"{{path}}:3:0: error: fake rejection")
    sys.exit(1)
if mode == "crash":
    sys.exit(137)
raise ValueError(f"unknown FAKE_LEAN_MODE {{mode!r}}")
"""


def _write_fake_lean(tmp_path: Path) -> str:
    script = tmp_path / "fake_lean"
    script.write_text(_FAKE_LEAN_SRC)
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(script)


def test_no_lean_binary_raises_not_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("code.ndbench.judge.lean.find_lean", lambda env_var="LEAN": None)
    with pytest.raises(NotAvailable):
        LeanJudge()


def test_find_lean_checks_env_var_first(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = tmp_path / "my_lean"
    fake.write_text("")
    monkeypatch.setenv("LEAN", str(fake))
    assert find_lean() == str(fake)


def test_failing_theorems_maps_error_lines_to_theorem_indices() -> None:
    output = "file.lean:5:2: error: bad\nfile.lean:12:0: error: also bad\n"
    bad, stray = _failing_theorems(output, starts=[3, 10, 20])
    assert bad == {0, 1}
    assert stray is False


def test_failing_theorems_flags_a_stray_error() -> None:
    output = "file.lean:1:0: error: prelude broke\n"
    bad, stray = _failing_theorems(output, starts=[3, 10])
    assert bad == set()
    assert stray is True


def test_unparseable_attempt_is_rejected_without_calling_lean(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    judge = LeanJudge(lean=_write_fake_lean(tmp_path))
    attempts = [
        Attempt(pid="p", i=0, body="UNPARSEABLE no-eos", native=None, n_tokens=1, finished=False)
    ]
    (v,) = judge.judge(_SEQ, attempts, ref_lines=None)
    assert v.ok is False
    assert v.kind == "unparseable"


def test_render_failure_is_rejected_without_calling_lean(tmp_path: Path) -> None:
    judge = LeanJudge(lean=_write_fake_lean(tmp_path))
    # 9 premises: more than lean_seq's MAX_PREMISES (8), so rendering the statement fails.
    seq = Sequent(premises=tuple("P" for _ in range(9)), conclusion="P")
    attempts = [
        Attempt(pid="p", i=0, body="N1 P : PR ; QED", native=None, n_tokens=1, finished=True)
    ]
    (v,) = judge.judge(seq, attempts, ref_lines=None)
    assert v.ok is False
    assert v.kind == "unparseable"


def test_fake_lean_accept_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FAKE_LEAN_MODE", "accept")
    judge = LeanJudge(lean=_write_fake_lean(tmp_path))
    attempts = [
        Attempt(pid="p", i=0, body=_GOOD_PROOF, native=None, n_tokens=1, finished=True),
    ]
    (v,) = judge.judge(_SEQ, attempts, ref_lines=None)
    assert v.ok is True
    assert v.reason == ""
    assert v.written == 7
    assert v.pruned is not None


def test_fake_lean_reject_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FAKE_LEAN_MODE", "reject")
    judge = LeanJudge(lean=_write_fake_lean(tmp_path))
    attempts = [
        Attempt(pid="p", i=0, body=_GOOD_PROOF, native=None, n_tokens=1, finished=True),
    ]
    (v,) = judge.judge(_SEQ, attempts, ref_lines=None)
    assert v.ok is False
    assert v.kind == "bad_line"
    assert v.reason == "lean rejected"


def test_fake_lean_crash_mode_with_single_theorem_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FAKE_LEAN_MODE", "crash")
    judge = LeanJudge(lean=_write_fake_lean(tmp_path))
    attempts = [
        Attempt(pid="p", i=0, body=_GOOD_PROOF, native=None, n_tokens=1, finished=True),
    ]
    (v,) = judge.judge(_SEQ, attempts, ref_lines=None)
    assert v.ok is False  # crashed with only one theorem in the batch -> rejected, not raised


def test_identical_bodies_are_deduplicated_into_one_lean_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two attempts with the same body render to the same source and should be judged as one
    `check_theorems` call (mirroring Dan's `judge()` dedup-by-source-text)."""
    monkeypatch.setenv("FAKE_LEAN_MODE", "accept")
    judge = LeanJudge(lean=_write_fake_lean(tmp_path))
    attempts = [
        Attempt(pid="p", i=0, body=_GOOD_PROOF, native=None, n_tokens=1, finished=True),
        Attempt(pid="p", i=1, body=_GOOD_PROOF, native=None, n_tokens=1, finished=True),
    ]
    verdicts = judge.judge(_SEQ, attempts, ref_lines=None)
    assert len(verdicts) == 2
    assert all(v.ok for v in verdicts)


def test_judge_id() -> None:
    assert LeanJudge.judge_id == "lean_v1"
