"""`suites/prefix.py` (plan §2 B6 / WP4): prefix indices are computed correctly and deduplicated,
every emitted `PrefixSpec` body is N1-based and ends at a line boundary, sources can be loaded from
a raw (`ab3/sealed`-schema) jsonl file, from a `Store` suite, or injected directly, bucket
assignment (with tolerance) and the plan §2 dedup policy (drop, not just flag) are applied, and
short buckets are reported."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from code.ndbench.core import proof as core_proof
from code.ndbench.core.io import utc_now_iso
from code.ndbench.core.types import (
    LengthLabel,
    Problem,
    ReferenceProof,
    Sequent,
    SuiteManifest,
)
from code.ndbench.store import Store
from code.ndbench.suites import prefix
from code.ndbench.vendor import load_nd_verify

from ._fixtures import load_valid_proofs

ND_VERIFY = load_nd_verify()


def _mk_problem(pid: str, seq: Sequent, body: str, length: int) -> Problem:
    ref = ReferenceProof(
        body=body, lines=length, pruned=core_proof.pruned_length(body), source="minlen"
    )
    label = LengthLabel(L=length, L_ub=length, L_lb=length, method="minlen_exact", bound=None)
    return Problem(
        pid=pid, suite="parts_pool", version="v1", sequent=seq, length=label, ref=ref, tags={}
    )


def _fixture_pool() -> list[Problem]:
    rows = load_valid_proofs()
    return [
        _mk_problem(f"fx{i}", Sequent.from_prompt(r["prompt"]), r["proof"], r["n_lines"])
        for i, r in enumerate(rows)
    ]


# --------------------------------------------------------------------------- _prefix_indices


def test_prefix_indices_are_sorted_unique_and_in_range() -> None:
    for ln, n_prem in [(10, 2), (14, 3), (18, 5), (5, 1), (3, 3)]:
        js = prefix._prefix_indices(ln, n_prem)
        assert js == sorted(set(js))
        assert all(0 <= j < ln for j in js)
        assert 0 in js
        assert (ln - 1) in js
        if n_prem < ln:
            assert n_prem in js


def test_prefix_indices_match_the_plan_formula() -> None:
    ln, n_prem = 40, 4
    js = prefix._prefix_indices(ln, n_prem)
    expected = {0, n_prem, 10, 20, 30, 37, 38, 39}  # ceil(40/4)=10, ceil(40/2)=20, ceil(3*40/4)=30
    assert set(js) == expected


# --------------------------------------------------------------------------- loaders


_RAW_ROW_EXACT = (
    '{"name": "r1", "prompt": "THM P , ( P > Q ) SEQ Q PRF", '
    '"minlen_proof": "N1 P : PR ; N2 ( P > Q ) : PR ; N3 Q : IMPE N2 N1 ; QED", '
    '"n_lines": 3, "label_kind": "exact", "minlen_bound": 8}'
)
_RAW_ROW_UB = (
    '{"name": "r2", "prompt": "THM R SEQ R PRF", '
    '"minlen_proof": "N1 R : PR ; N2 R : R N1 ; QED", '
    '"n_lines": 2, "label_kind": "ub_cert14", "minlen_bound": 14}'
)
# A real ab3/sealed `ub_cert14` row has `minlen_proof: null` (no exact minlen search completed);
# `n_lines` there is the *constructed* proof's own length (see `_load_raw_source`'s docstring).
_RAW_ROW_UB_NULL_MINLEN = (
    '{"name": "r3", "prompt": "THM ( P & Q ) SEQ ( Q & P ) PRF", '
    '"minlen_proof": null, '
    '"constructed_proof": '
    '"N1 ( P & Q ) : PR ; N2 Q : ANDE2 N1 ; N3 P : ANDE1 N1 ; N4 ( Q & P ) : ANDI N2 N3 ; QED", '
    '"n_lines": 4, "label_kind": "ub_cert14", "minlen_bound": 14}'
)


def test_load_raw_source_maps_label_kind_to_method(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    path.write_text(_RAW_ROW_EXACT + "\n" + _RAW_ROW_UB + "\n")
    problems = prefix._load_raw_source({"path": str(path)})
    assert len(problems) == 2
    by_name = {p.pid: p for p in problems}
    exact = by_name["raw_r1"]
    assert exact.length.method == "minlen_exact" and exact.length.L == 3 and exact.length.L_lb == 3
    ub = by_name["raw_r2"]
    assert ub.length.method == "ub_lb" and ub.length.L == 2 and ub.length.L_lb is None
    for p in problems:
        assert p.ref is not None
        ok, reason, nl = ND_VERIFY.verify_text(p.sequent.prompt() + " " + p.ref.body)
        assert ok and nl == p.ref.lines, (p.pid, reason)


def test_load_raw_source_falls_back_to_constructed_proof_when_minlen_proof_is_null(
    tmp_path: Path,
) -> None:
    path = tmp_path / "raw.jsonl"
    path.write_text(_RAW_ROW_UB_NULL_MINLEN + "\n")
    problems = prefix._load_raw_source({"path": str(path)})
    assert len(problems) == 1
    p = problems[0]
    assert p.ref is not None and p.ref.lines == 4
    assert p.ref.body.startswith("N1 ( P & Q ) : PR")
    ok, reason, nl = ND_VERIFY.verify_text(p.sequent.prompt() + " " + p.ref.body)
    assert ok and nl == 4, reason


def _manifest(n: int) -> SuiteManifest:
    return SuiteManifest(
        suite="glue",
        version="v1",
        builder="test",
        config={},
        seed=0,
        created_utc=utc_now_iso(),
        git_commit="test",
        n_items=n,
        items_sha256="x",
        excl_sources={},
        stats={},
    )


def test_load_store_source(tmp_path: Path) -> None:
    store = Store(tmp_path)
    problems = _fixture_pool()[:5]
    store.write_suite(problems, _manifest(len(problems)))
    loaded = prefix._load_store_source({"suite": "glue", "version": "v1"}, str(tmp_path))
    assert [p.pid for p in loaded] == [p.pid for p in problems]


# --------------------------------------------------------------------------- build()


def test_build_with_injected_problems_buckets_and_prefixes() -> None:
    pool = _fixture_pool()
    cfg = {
        "injected_problems": pool,
        "buckets": [2, 4, 6, 8, 10],
        "tolerance": 1,
        "per_bucket": 3,
        "version": "v1",
    }
    problems, stats = prefix.build(cfg, certifier=None, excluder=None, rng_seed=0)
    assert problems
    assert set(stats["bucket_counts"]) == {2, 4, 6, 8, 10}
    assert sum(stats["bucket_counts"].values()) <= 5 * 3

    by_src: dict[str, list[Problem]] = {}
    for p in problems:
        assert p.suite == "prefix"
        assert p.prefix is not None
        assert {"j", "L", "src_pid", "bucket"} <= set(p.tags)
        by_src.setdefault(str(p.tags["src_pid"]), []).append(p)

    src_by_pid = {p.pid: p for p in pool}
    for src_pid, items in by_src.items():
        src = src_by_pid[src_pid]
        assert src.ref is not None
        ln = src.ref.lines
        n_prem = len(src.sequent.premises)
        expected_js = set(prefix._prefix_indices(ln, n_prem))
        assert {cast(int, p.tags["j"]) for p in items} <= expected_js
        for p in items:
            j = cast(int, p.tags["j"])
            assert p.prefix is not None
            assert p.prefix.j == j
            expected_body = core_proof.take_prefix(src.ref.body, j, new_start=1)
            assert p.prefix.body == expected_body
            # ends at a line boundary, N1-based (or empty at j=0)
            if p.prefix.body:
                toks = p.prefix.body.split()
                assert toks[0] == "N1"
                assert toks[-1] == ";"
                assert core_proof.line_count(p.prefix.body) == j
            else:
                assert j == 0


def test_build_respects_excluder_drop_policy() -> None:
    class _AlwaysExclude:
        def status(self, seq: Sequent) -> tuple[bool, tuple[str, ...]]:
            del seq
            return False, ("some_eval_set",)  # eval-tier flag only -> still DROPPED for prefix

    pool = _fixture_pool()[:5]
    cfg = {
        "injected_problems": pool,
        "buckets": [p_.ref.lines for p_ in pool if p_.ref],
        "tolerance": 0,
    }
    problems, stats = prefix.build(cfg, certifier=None, excluder=_AlwaysExclude(), rng_seed=0)
    assert problems == []
    assert stats["drops"]["excluded"] == len(pool)


def test_build_dedups_identical_sequents() -> None:
    pool = _fixture_pool()[:3]
    duplicated = pool + pool  # same pids and sequents, listed twice
    cfg = {
        "injected_problems": duplicated,
        "buckets": [p_.ref.lines for p_ in pool if p_.ref],
        "tolerance": 0,
    }
    problems, stats = prefix.build(cfg, certifier=None, excluder=None, rng_seed=0)
    assert stats["drops"]["duplicate"] == len(pool)
    seen_src = {cast(str, p.tags["src_pid"]) for p in problems}
    assert len(seen_src) == len(pool)


def test_build_reports_short_buckets() -> None:
    pool = _fixture_pool()[:2]
    cfg = {
        "injected_problems": pool,
        "buckets": [999],  # far from any fixture length -> stays empty
        "tolerance": 0,
        "per_bucket": 50,
    }
    problems, stats = prefix.build(cfg, certifier=None, excluder=None, rng_seed=0)
    assert problems == []
    assert stats["bucket_counts"] == {999: 0}
    assert stats["short_buckets"] == {999: 0}


def test_build_default_buckets_pick_up_l10_fixtures() -> None:
    pool = _fixture_pool()
    cfg = {"injected_problems": pool, "per_bucket": 50}  # DEFAULT_BUCKETS, tolerance=0
    problems, stats = prefix.build(cfg, certifier=None, excluder=None, rng_seed=0)
    assert set(stats["bucket_counts"]) == set(prefix.DEFAULT_BUCKETS)
    assert stats["bucket_counts"][10] > 0
    for b in (14, 18, 24, 30, 40):
        assert stats["bucket_counts"][b] == 0
    assert stats["short_buckets"]
    assert all(p.tags["bucket"] == 10 for p in problems)
