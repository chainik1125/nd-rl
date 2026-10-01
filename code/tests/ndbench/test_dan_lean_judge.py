"""WP-D2 acceptance tests: `LeanJudge` v2 (plan §2, `code/ndbench/judge/lean.py`).

Two groups:

- Fake-`lean`-binary tests (always run, CPU-only, no real Lean needed): the literal-native-text
  vs. `nd2lean.translate` routing (module docstring's decision order), one batched Lean process
  per group regardless of how many items/attempts, same-source dedup, the sqlite verdict cache,
  the version check, and every "rejected without calling Lean" short circuit. A small Python
  script (written to `tmp_path`, not a network resource) stands in for `lean`: it answers
  `--version` and otherwise logs the `.lean` file it was asked to check (so these tests can
  inspect *what* was sent to Lean) before accepting/rejecting/crashing per `$FAKE_LEAN_MODE`.
- Real-Lean tests (skipped if `~/.elan/bin/lean` is absent; present and `4.34.0` in this
  environment, so they run here): ~10 golden ND proofs accepted, ~10 corruptions rejected, and the
  premise-omitting proof Lean accepts that `nd_v2` rejects (plan §2's "Lean is a superset of
  nd_v2 on denotable proofs... where his models omit restating premises").
"""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

from code.ndbench.core.types import Attempt, Sequent
from code.ndbench.judge.base import NotAvailable
from code.ndbench.judge.lean import LeanJudge, find_lean, lean_version
from code.ndbench.judge.nd import NDJudge
from code.ndbench.vendor.danstate_loader import load_danstate

from ._fixtures import generate_corruptions, load_valid_proofs

_REAL_LEAN = find_lean()
_REAL_LEAN_OK = _REAL_LEAN is not None and lean_version(_REAL_LEAN) == "4.34.0"

_FAKE_LEAN_SRC = f"""\
#!{sys.executable}
import os, sys

if "--version" in sys.argv[1:]:
    print(os.environ.get(
        "FAKE_LEAN_VERSION",
        "Lean (version 4.34.0, x86_64-unknown-linux-gnu, commit deadbeef, Release)",
    ))
    sys.exit(0)

mode = os.environ.get("FAKE_LEAN_MODE", "accept")
path = sys.argv[-1]
with open(path) as f:
    text = f.read()

log = os.environ.get("FAKE_LEAN_LOG")
if log:
    with open(log, "a") as lf:
        lf.write("=====\\n" + text)

if mode == "accept":
    sys.exit(0)
if mode == "reject":
    # Matches the real error format `_ERROR_RE` parses: "<path>:<line>:<col>: error: ...".
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


def _native(lean_text: str) -> str:
    return json.dumps({"lean": lean_text, "base": 0, "steps": 1, "end": "done"})


def _invocation_count(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    return log_path.read_text().count("=====\n")


# --------------------------------------------------------------------------- fake-lean tests


def test_no_lean_binary_raises_not_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("code.ndbench.judge.lean.find_lean", lambda env_var="LEAN": None)
    with pytest.raises(NotAvailable):
        LeanJudge()


def test_wrong_lean_version_raises_not_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FAKE_LEAN_VERSION", "Lean (version 4.9.9, x86_64-unknown-linux-gnu)")
    with pytest.raises(NotAvailable, match="4.34.0"):
        LeanJudge(lean=_write_fake_lean(tmp_path))


def test_lean_version_parses_real_lean_format(tmp_path: Path) -> None:
    assert lean_version(_write_fake_lean(tmp_path)) == "4.34.0"


def test_lean_version_returns_none_for_a_nonexistent_binary(tmp_path: Path) -> None:
    assert lean_version(str(tmp_path / "does_not_exist")) is None


def test_unparseable_attempt_is_rejected_without_calling_lean(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log = tmp_path / "log.txt"
    monkeypatch.setenv("FAKE_LEAN_LOG", str(log))
    judge = LeanJudge(lean=_write_fake_lean(tmp_path))
    attempts = [
        Attempt(pid="p", i=0, body="UNPARSEABLE no-eos", native=None, n_tokens=1, finished=False)
    ]
    (v,) = judge.judge(Sequent(premises=("P",), conclusion="P"), attempts, ref_lines=None)
    assert v.ok is False
    assert v.kind == "unparseable"
    assert _invocation_count(log) == 0


def test_statement_render_failure_with_native_lean_text_is_rejected_without_lean(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """9 premises: more than `lean_tok.MAXH` (8), so building the statement (needed only on the
    literal-native-text path) raises before any Lean call."""
    log = tmp_path / "log.txt"
    monkeypatch.setenv("FAKE_LEAN_LOG", str(log))
    judge = LeanJudge(lean=_write_fake_lean(tmp_path))
    seq = Sequent(premises=tuple("P" for _ in range(9)), conclusion="P")
    attempts = [
        Attempt(
            pid="p",
            i=0,
            body="N1 P : PR ; QED",
            native=_native("exact h1"),
            n_tokens=1,
            finished=True,
        )
    ]
    (v,) = judge.judge(seq, attempts, ref_lines=None)
    assert v.ok is False
    assert v.kind == "unparseable"
    assert v.reason.startswith("statement:")
    assert _invocation_count(log) == 0


def test_untranslatable_body_without_native_lean_text_is_rejected_without_lean(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log = tmp_path / "log.txt"
    monkeypatch.setenv("FAKE_LEAN_LOG", str(log))
    judge = LeanJudge(lean=_write_fake_lean(tmp_path))
    attempts = [
        Attempt(pid="p", i=0, body="not a proof at all", native=None, n_tokens=1, finished=True)
    ]
    (v,) = judge.judge(Sequent(premises=("P",), conclusion="P"), attempts, ref_lines=None)
    assert v.ok is False
    assert v.kind == "unparseable"
    assert v.reason.startswith("nd2lean:")
    assert _invocation_count(log) == 0


def test_literal_native_lean_text_is_checked_directly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When `native["lean"]` is present, the checked source is `statement(prompt) + " " +
    native["lean"]` -- `Attempt.body` is ignored entirely (set here to something that would
    translate to a completely different theorem, to prove the literal path, not the fallback
    path, was used)."""
    monkeypatch.setenv("FAKE_LEAN_MODE", "accept")
    log = tmp_path / "log.txt"
    monkeypatch.setenv("FAKE_LEAN_LOG", str(log))
    judge = LeanJudge(lean=_write_fake_lean(tmp_path))
    seq = Sequent(premises=("P",), conclusion="P")
    attempts = [
        Attempt(
            pid="p",
            i=0,
            body="N1 Q : PR ; QED",  # a different (and alone, unrelated) body
            native=_native("have n1 : P := h1 ; exact n1"),
            n_tokens=1,
            finished=True,
        )
    ]
    (v,) = judge.judge(seq, attempts, ref_lines=None)
    assert v.ok is True
    ds = load_danstate()
    expected_statement = " ".join(ds.lean_tok.prompt_tokens(seq.prompt()))
    assert expected_statement == "theorem t ( P Q R S : Prop ) ( h1 : P ) : P := by"
    checked = log.read_text()
    # "theorem t " -> "theorem t0 " is `_run_chunk`'s own per-batch renaming.
    expected_renamed = expected_statement.replace("theorem t ", "theorem t0 ", 1)
    assert f"{expected_renamed} have n1 : P := h1 ; exact n1" in checked
    # the fallback-path rendering of the (unrelated) body never appears.
    assert "theorem t0 (P Q R S : Prop)" not in checked  # nd2lean's own spacing convention


def test_translate_fallback_used_without_native_lean_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FAKE_LEAN_MODE", "accept")
    log = tmp_path / "log.txt"
    monkeypatch.setenv("FAKE_LEAN_LOG", str(log))
    judge = LeanJudge(lean=_write_fake_lean(tmp_path))
    seq = Sequent(premises=("P",), conclusion="P")
    attempts = [
        Attempt(pid="p", i=0, body="N1 P : PR ; QED", native=None, n_tokens=1, finished=True)
    ]
    (v,) = judge.judge(seq, attempts, ref_lines=None)
    assert v.ok is True
    ds = load_danstate()
    expected_source = ds.nd2lean.translate(seq.prompt(), attempts[0].body, require_all_pr=False)
    assert expected_source.replace("theorem t ", "theorem t0 ", 1) in log.read_text()


def test_one_batched_lean_call_across_a_whole_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """5 items x 3 attempts each, every attempt a distinct literal text: one `judge_many` call
    across every item starts exactly one `lean` process (well under the 400-source chunk size),
    not one per item (module docstring's "Batching"/`judge.base.Judge.judge_many`'s rationale)."""
    monkeypatch.setenv("FAKE_LEAN_MODE", "accept")
    log = tmp_path / "log.txt"
    monkeypatch.setenv("FAKE_LEAN_LOG", str(log))
    judge = LeanJudge(lean=_write_fake_lean(tmp_path))
    items = []
    for i in range(5):
        seq = Sequent(premises=("P",), conclusion="P")
        attempts = [
            Attempt(
                pid=f"p{i}",
                i=j,
                body="N1 Q : PR ; QED",
                native=_native(f"have n1 : P := h1 ; exact h1 -- {i}-{j}"),
                n_tokens=1,
                finished=True,
            )
            for j in range(3)
        ]
        items.append((seq, attempts, None))
    verdicts = judge.judge_many(items)
    assert len(verdicts) == 5
    assert all(v.ok for group in verdicts for v in group)
    assert _invocation_count(log) == 1


def test_identical_sources_are_deduplicated_into_one_theorem(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FAKE_LEAN_MODE", "accept")
    log = tmp_path / "log.txt"
    monkeypatch.setenv("FAKE_LEAN_LOG", str(log))
    judge = LeanJudge(lean=_write_fake_lean(tmp_path))
    seq = Sequent(premises=("P",), conclusion="P")
    same_native = _native("have n1 : P := h1 ; exact n1")
    attempts = [
        Attempt(pid="p", i=k, body="N1 Q : PR ; QED", native=same_native, n_tokens=1, finished=True)
        for k in range(3)
    ]
    verdicts = judge.judge(seq, attempts, ref_lines=None)
    assert len(verdicts) == 3
    assert all(v.ok for v in verdicts)
    assert log.read_text().count("have n1 : P := h1 ; exact n1") == 1


def test_cache_hit_skips_the_second_lean_invocation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FAKE_LEAN_MODE", "accept")
    log = tmp_path / "log.txt"
    monkeypatch.setenv("FAKE_LEAN_LOG", str(log))
    fake = _write_fake_lean(tmp_path)
    cache_dir = tmp_path / "cache"

    seq = Sequent(premises=("P",), conclusion="P")
    attempts = [
        Attempt(
            pid="p",
            i=0,
            body="N1 Q : PR ; QED",
            native=_native("have n1 : P := h1 ; exact n1"),
            n_tokens=1,
            finished=True,
        )
    ]

    judge1 = LeanJudge(lean=fake, cache_dir=cache_dir)
    (v1,) = judge1.judge(seq, attempts, ref_lines=None)
    assert v1.ok is True
    assert _invocation_count(log) == 1

    # A fresh LeanJudge instance pointed at the same cache_dir: the cache is on disk, not
    # in-process memoisation, so this must still hit it.
    judge2 = LeanJudge(lean=fake, cache_dir=cache_dir)
    (v2,) = judge2.judge(seq, attempts, ref_lines=None)
    assert v2.ok is True
    assert _invocation_count(log) == 1  # unchanged: no second lean process was started
    assert (cache_dir / "lean_v2" / "verdicts.sqlite").exists()


def test_fake_lean_reject_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FAKE_LEAN_MODE", "reject")
    judge = LeanJudge(lean=_write_fake_lean(tmp_path))
    seq = Sequent(premises=("P",), conclusion="P")
    attempts = [
        Attempt(
            pid="p",
            i=0,
            body="N1 P : PR ; QED",
            native=_native("have n1 : P := h1 ; exact n1"),
            n_tokens=1,
            finished=True,
        )
    ]
    (v,) = judge.judge(seq, attempts, ref_lines=None)
    assert v.ok is False
    assert v.kind == "bad_line"
    assert v.reason == "lean rejected"
    assert v.pruned is None
    assert v.written == 1


def test_fake_lean_crash_mode_with_single_theorem_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FAKE_LEAN_MODE", "crash")
    judge = LeanJudge(lean=_write_fake_lean(tmp_path))
    seq = Sequent(premises=("P",), conclusion="P")
    attempts = [
        Attempt(
            pid="p",
            i=0,
            body="N1 P : PR ; QED",
            native=_native("have n1 : P := h1 ; exact n1"),
            n_tokens=1,
            finished=True,
        )
    ]
    (v,) = judge.judge(seq, attempts, ref_lines=None)
    assert v.ok is False  # crashed with only one theorem in the batch -> rejected, not raised


def test_judge_id() -> None:
    assert LeanJudge.judge_id == "lean_v2"


# ----------------------------------------------------------------------------- real-lean tests

pytestmark_real_lean = pytest.mark.skipif(
    not _REAL_LEAN_OK, reason="no lean 4.34.0 binary found ($LEAN, PATH, ~/.elan/bin/lean)"
)


@pytestmark_real_lean
def test_real_lean_accepts_ten_golden_proofs(tmp_path: Path) -> None:
    proofs = sorted(load_valid_proofs(), key=lambda p: p["n_lines"])[:10]
    judge = LeanJudge(lean=_REAL_LEAN, cache_dir=tmp_path / "cache")
    items = []
    for p in proofs:
        seq = Sequent.from_prompt(p["prompt"])
        a = Attempt(pid=p["name"], i=0, body=p["proof"], native=None, n_tokens=1, finished=True)
        items.append((seq, [a], None))
    verdicts = judge.judge_many(items)
    for p, (v,) in zip(proofs, verdicts, strict=True):
        assert v.ok is True, f"{p['name']}: {v.reason}"
        assert v.pruned is not None


@pytestmark_real_lean
def test_real_lean_rejects_ten_corruptions(tmp_path: Path) -> None:
    proofs = load_valid_proofs()
    corruptions = generate_corruptions(proofs, 10, seed=0)
    judge = LeanJudge(lean=_REAL_LEAN, cache_dir=tmp_path / "cache")
    items = []
    for fixture, body in corruptions:
        seq = Sequent.from_prompt(fixture["prompt"])
        a = Attempt(pid=fixture["name"], i=0, body=body, native=None, n_tokens=1, finished=True)
        items.append((seq, [a], None))
    verdicts = judge.judge_many(items)
    for (fixture, _body), (v,) in zip(corruptions, verdicts, strict=True):
        assert v.ok is False, f"{fixture['name']}: corruption unexpectedly accepted"


@pytestmark_real_lean
def test_real_lean_accepts_premise_omitting_proof_that_nd_v2_rejects(tmp_path: Path) -> None:
    """Dan's headline Lean-vs-nd_v2 disagreement class (plan §2: "41.7% of Lean-only accepts"
    omit restating a premise): a proof that only restates a *prefix* of the declared premises
    still proves the conclusion, which `nd2lean.translate(..., require_all_pr=False)` allows and
    Lean accepts (the unused hypothesis is just an unused `by`-block parameter), while `nd_verify`
    (`judge.nd.NDJudge`) requires the `PR` block to match every declared premise exactly."""
    seq = Sequent(premises=("P", "Q"), conclusion="P")
    body = "N1 P : PR ; QED"  # only restates the first premise; never mentions Q at all
    attempts = [Attempt(pid="p", i=0, body=body, native=None, n_tokens=1, finished=True)]

    (nd_verdict,) = NDJudge().judge(seq, attempts, ref_lines=None)
    assert nd_verdict.ok is False

    judge = LeanJudge(lean=_REAL_LEAN, cache_dir=tmp_path / "cache")
    (lean_verdict,) = judge.judge(seq, attempts, ref_lines=None)
    assert lean_verdict.ok is True


@pytestmark_real_lean
def test_real_lean_literal_text_accept_and_reject(tmp_path: Path) -> None:
    judge = LeanJudge(lean=_REAL_LEAN, cache_dir=tmp_path / "cache")
    seq = Sequent(premises=("P",), conclusion="P")

    ok_attempt = Attempt(
        pid="p",
        i=0,
        body="N1 Q : PR ; QED",
        native=_native("have n1 : P := h1 ; exact n1"),
        n_tokens=1,
        finished=True,
    )
    (v_ok,) = judge.judge(seq, [ok_attempt], ref_lines=None)
    assert v_ok.ok is True

    bad_attempt = Attempt(
        pid="p",
        i=0,
        body="N1 Q : PR ; QED",
        # n2 is never defined: Lean reports "unknown identifier".
        native=_native("have n1 : P := h1 ; exact n2"),
        n_tokens=1,
        finished=True,
    )
    (v_bad,) = judge.judge(seq, [bad_attempt], ref_lines=None)
    assert v_bad.ok is False
