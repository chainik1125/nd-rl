"""`MinlenCertifier` (plan §3 WP2 row): reproduces `minlen.py`'s own selftest cases and at least 6
rows of plan §0's verification table, plus the `ub_lb`/`reference`/cache/witness-verification
paths that table alone does not exercise (those are cheap to hit directly via monkeypatching
rather than hunting for a real theorem whose search genuinely times out)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from code.ndbench.core.types import Sequent
from code.ndbench.logic import certify as certify_mod
from code.ndbench.logic.certify import MAX_PROCS, MinlenCertifier

# minlen.py's own `selftest()` cases: (prompt, expected min_lines_ub within bound 8).
_MINLEN_SELFTEST_CASES: list[tuple[str, int | None]] = [
    ("THM ( P > Q ) , P SEQ Q PRF", 3),
    ("THM SEQ ( P > P ) PRF", 2),
    ("THM ( ~ ( ~ P ) ) SEQ P PRF", 2),
    ("THM ( P > Q ) , ( ~ Q ) SEQ ( ~ P ) PRF", 6),
    ("THM ( P v Q ) , ( ~ P ) SEQ Q PRF", 7),
    ("THM ( P > Q ) SEQ ( ( ~ Q ) > ( ~ P ) ) PRF", 7),
    ("THM P , ( ~ P ) SEQ Q PRF", 4),
    ("THM ( P & Q ) SEQ ( Q & P ) PRF", 4),
    ("THM ( P v Q ) SEQ ( Q v P ) PRF", 6),
    ("THM ( ( P > Q ) > P ) SEQ P PRF", None),  # Peirce: unreachable within bound 8
    ("THM SEQ ( P v ( ~ P ) ) PRF", None),  # excluded middle: unreachable within bound 8
    ("THM ( P > ( Q > R ) ) SEQ ( ( P & Q ) > R ) PRF", 7),
    ("THM ( ( P & Q ) > R ) SEQ ( P > ( Q > R ) ) PRF", 7),
]


def test_reproduces_minlen_selftest_cases() -> None:
    certifier = MinlenCertifier(procs=4)
    seqs = [Sequent.from_prompt(prompt) for prompt, _ in _MINLEN_SELFTEST_CASES]
    labels = certifier.certify(seqs, [None] * len(seqs), bound=8, time_s=30.0)
    for (prompt, expected), label in zip(_MINLEN_SELFTEST_CASES, labels, strict=True):
        assert expected == label.L, f"{prompt}: expected {expected}, got {label.L}"
        assert label.method == "minlen_exact"
        assert label.timeout is False


def _not_chain(k: int, atom: str = "P") -> str:
    s = atom
    for _ in range(k):
        s = f"( ~ {s} )"
    return s


def test_reproduces_six_plan_table_rows() -> None:
    """DN-elim `~^{2n}P |- P` (predicted `n+1`, exact n=1..5) plus the chain-glue shortcut
    `(P&Q),(Q>R) |- R` (naive 7, minlen 4) -- 6 distinct rows of plan §0's verification table."""
    certifier = MinlenCertifier(procs=4)

    dn_elim_seqs = [Sequent(premises=(_not_chain(2 * n),), conclusion="P") for n in range(1, 6)]
    dn_elim_labels = certifier.certify(dn_elim_seqs, [None] * 5, bound=10, time_s=30.0)
    for n, label in zip(range(1, 6), dn_elim_labels, strict=True):
        assert n + 1 == label.L, f"DN-elim n={n}: expected {n + 1}, got {label.L}"
        assert label.method == "minlen_exact"

    chain_glue_seq = Sequent(premises=("( P & Q )", "( Q > R )"), conclusion="R")
    (chain_label,) = certifier.certify([chain_glue_seq], [None], bound=8, time_s=30.0)
    assert chain_label.L == 4
    assert chain_label.method == "minlen_exact"


def test_no_witness_bound_exhausted_without_timeout() -> None:
    """Peirce's law genuinely has no proof within bound 8 in minlen's search space; `certify`
    reports `L=None` without a timeout (an exhaustion, not a search-time cutoff)."""
    certifier = MinlenCertifier(procs=1)
    seq = Sequent.from_prompt("THM ( ( P > Q ) > P ) SEQ P PRF")
    (label,) = certifier.certify([seq], [None], bound=8, time_s=30.0)
    assert label.L is None
    assert label.method == "minlen_exact"
    assert label.timeout is False


def test_witness_verification_failure_raises() -> None:
    certifier = MinlenCertifier(procs=1)
    seq = Sequent(premises=("P",), conclusion="P")
    bogus_witness = "N1 P : PR ; N2 Q : R N1 ; QED"  # does not verify (Q is not P)
    with pytest.raises(ValueError):
        certifier.certify([seq], [bogus_witness], bound=8, time_s=5.0)


def test_ub_lb_path_via_monkeypatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a witness whose direct `minlen(bound=L_ub)` search times out, but the cheaper
    "exhaust below" check (`bound=L_ub - 1`) resolves cleanly -- the `ub_lb` method (plan §2)."""
    calls: list[int] = []

    def fake_run_minlen(prompt: str, bound: int, time_s: float) -> dict[str, Any]:
        del prompt, time_s
        calls.append(bound)
        if bound == 5:  # L_ub itself: times out
            return {"min_lines_ub": None, "proof": None, "timeout": True, "bound": bound}
        if bound == 4:  # L_ub - 1: exhausts cleanly, confirming nothing shorter exists
            return {"min_lines_ub": None, "proof": None, "timeout": False, "bound": bound}
        raise AssertionError(f"unexpected bound {bound}")

    monkeypatch.setattr(certify_mod, "_run_minlen", fake_run_minlen)
    certifier = MinlenCertifier(procs=1)
    seq = Sequent(premises=("P",), conclusion="P")
    # A 5-line witness for the trivial "P |- P" (the actual body's content does not matter to the
    # monkeypatched _run_minlen; only nd_verify's own line count from _verify_witness does).
    witness = "N1 P : PR ; N2 P : R N1 ; N3 P : R N1 ; N4 P : R N1 ; N5 P : R N1 ; QED"
    (label,) = certifier.certify([seq], [witness], bound=99, time_s=5.0)
    assert label.method == "ub_lb"
    assert label.L == 5
    assert label.L_lb == 5
    assert label.timeout is False
    assert calls == [5, 4]


def test_reference_fallback_when_both_witness_searches_time_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def always_timeout(prompt: str, bound: int, time_s: float) -> dict[str, Any]:
        del prompt, bound, time_s
        return {"min_lines_ub": None, "proof": None, "timeout": True, "bound": 0}

    monkeypatch.setattr(certify_mod, "_run_minlen", always_timeout)
    certifier = MinlenCertifier(procs=1)
    seq = Sequent(premises=("P",), conclusion="P")
    witness = "N1 P : PR ; N2 P : R N1 ; QED"
    (label,) = certifier.certify([seq], [witness], bound=99, time_s=0.001)
    assert label.method == "reference"
    assert label.L is None
    assert label.L_ub == 2
    assert label.timeout is True


def test_reference_fallback_when_witness_outside_search_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The witness verifies, but minlen's restricted search space never matches it (exhausts
    without timing out) -- plausible for a constructed proof using hypotheses outside minlen's
    subformula closure. Falls back to `"reference"`, not `"ub_lb"`."""

    def exhausts_cleanly(prompt: str, bound: int, time_s: float) -> dict[str, Any]:
        del prompt, bound, time_s
        return {"min_lines_ub": None, "proof": None, "timeout": False, "bound": 0}

    monkeypatch.setattr(certify_mod, "_run_minlen", exhausts_cleanly)
    certifier = MinlenCertifier(procs=1)
    seq = Sequent(premises=("P",), conclusion="P")
    witness = "N1 P : PR ; N2 P : R N1 ; QED"
    (label,) = certifier.certify([seq], [witness], bound=99, time_s=5.0)
    assert label.method == "reference"
    assert label.L is None
    assert label.L_ub == 2


def test_cache_by_key_avoids_recomputation(tmp_path: Path) -> None:
    certifier = MinlenCertifier(procs=1, cache_dir=tmp_path)
    seq = Sequent(premises=("( P > Q )", "P"), conclusion="Q")
    (first,) = certifier.certify([seq], [None], bound=6, time_s=10.0)
    assert first.L == 3

    # A second certifier reading the same cache_dir must not need to search at all: monkeypatch
    # the only place a real minlen call could originate from and confirm it is never reached.
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("minlen should not run again for an already-cached key")

    import code.ndbench.logic.certify as mod

    original = mod._run_minlen
    mod._run_minlen = boom  # type: ignore[assignment]
    try:
        certifier2 = MinlenCertifier(procs=1, cache_dir=tmp_path)
        (second,) = certifier2.certify([seq], [None], bound=6, time_s=10.0)
    finally:
        mod._run_minlen = original  # type: ignore[assignment]
    assert second == first
    assert (tmp_path / "minlen_labels.jsonl").exists()


def test_duplicate_keys_within_one_call_computed_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    real = certify_mod._certify_one

    def counting(payload: Any) -> Any:
        nonlocal calls
        calls += 1
        return real(payload)

    monkeypatch.setattr(certify_mod, "_certify_one", counting)
    certifier = MinlenCertifier(procs=1)
    seq_a = Sequent(premises=("P",), conclusion="P")
    seq_b = Sequent(premises=("Q",), conclusion="Q")  # same canonical key as seq_a
    assert seq_a.key() == seq_b.key()
    labels = certifier.certify([seq_a, seq_b], [None, None], bound=6, time_s=10.0)
    assert calls == 1
    assert labels[0] == labels[1]


def test_procs_clamped_to_max() -> None:
    certifier = MinlenCertifier(procs=1000)
    assert certifier.procs == MAX_PROCS


def test_witnesses_length_mismatch_raises() -> None:
    certifier = MinlenCertifier(procs=1)
    seq = Sequent(premises=("P",), conclusion="P")
    with pytest.raises(ValueError):
        certifier.certify([seq], [], bound=6, time_s=5.0)
