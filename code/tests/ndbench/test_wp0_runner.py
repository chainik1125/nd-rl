"""WP0 acceptance test: `runner.run` with `FakeAdapter` gives the expected `n`/`c`, adaptive
stopping, and deterministic seeds (plan §3 WP0 row)."""

from __future__ import annotations

from pathlib import Path

import pytest

from code.ndbench.adapters.fake import FakeAdapter
from code.ndbench.core.io import git_commit_sha, sha256_str, utc_now_iso
from code.ndbench.core.proof import pruned_length
from code.ndbench.core.types import LengthLabel, Problem, ReferenceProof, Sequent, SuiteManifest
from code.ndbench.judge.nd import NDJudge
from code.ndbench.runner.adaptive import AdaptiveRule
from code.ndbench.runner.runner import RunSpec, _item_seed, run
from code.ndbench.store import Store

from ._fixtures import load_valid_proofs


def _problems() -> list[Problem]:
    proofs = [p for p in load_valid_proofs() if p["n_lines"] <= 4][:2]
    assert len(proofs) == 2
    out = []
    for i, p in enumerate(proofs):
        seq = Sequent.from_prompt(p["prompt"])
        out.append(
            Problem(
                pid=f"item{i}",
                suite="demo",
                version="v1",
                sequent=seq,
                length=LengthLabel(
                    L=p["n_lines"], L_ub=None, L_lb=None, method="minlen_exact", bound=None
                ),
                ref=ReferenceProof(
                    body=p["proof"],
                    lines=p["n_lines"],
                    pruned=pruned_length(p["proof"]),
                    source="minlen",
                ),
                tags={},
            )
        )
    return out


def _write_suite(store: Store, problems: list[Problem], seed: int = 0) -> None:
    manifest = SuiteManifest(
        suite="demo",
        version="v1",
        builder="test",
        config={},
        seed=seed,
        created_utc=utc_now_iso(),
        git_commit=git_commit_sha(),
        n_items=len(problems),
        items_sha256=sha256_str("x"),
        excl_sources={},
        stats={},
    )
    store.write_suite(problems, manifest)


def _refs(problems: list[Problem]) -> dict[str, str]:
    return {p.pid: p.ref.body for p in problems if p.ref is not None}


def test_adaptive_stop_early_when_p_is_one(tmp_path: Path) -> None:
    """p=1.0: every draw is the reference proof, so the item hits `stop_c` inside its very first
    chunk and never draws a second one -- an exact, non-random expected outcome."""
    problems = _problems()
    store = Store(tmp_path)
    _write_suite(store, problems)
    adapter = FakeAdapter(refs=_refs(problems), p=1.0)
    spec = RunSpec(
        suite="demo",
        version="v1",
        model="fake",
        k=999,  # unused in adaptive mode
        adaptive=AdaptiveRule(chunk=5, n_max=50, stop_c=3),
        seed=1,
    )
    summary = run(spec, adapter, NDJudge(), store)
    assert summary["status"] == "complete"
    assert summary["n"] == 5 * len(problems)  # exactly one chunk per item
    assert summary["c"] == 5 * len(problems)
    assert summary["n_chunks"] == 1
    ref_lines_by_pid = {pr.pid: pr.ref.lines for pr in problems if pr.ref is not None}
    for item in store.items(summary["run_id"]):
        assert item.n == 5
        assert item.c == 5
        assert item.first_success == 1
        assert item.ok_written == [ref_lines_by_pid[item.pid]] * 5


def test_adaptive_runs_to_n_max_when_p_is_zero(tmp_path: Path) -> None:
    """p=0.0: every draw is corrupted, `stop_c` is never reached, so the item draws chunks until
    `n_max` is exhausted -- again an exact expected outcome."""
    problems = _problems()
    store = Store(tmp_path)
    _write_suite(store, problems)
    adapter = FakeAdapter(refs=_refs(problems), p=0.0)
    spec = RunSpec(
        suite="demo",
        version="v1",
        model="fake",
        k=999,
        adaptive=AdaptiveRule(chunk=5, n_max=50, stop_c=3),
        seed=2,
    )
    summary = run(spec, adapter, NDJudge(), store)
    assert summary["n"] == 50 * len(problems)
    assert summary["c"] == 0
    assert summary["n_chunks"] == 10
    for item in store.items(summary["run_id"]):
        assert item.n == 50
        assert item.c == 0
        assert item.first_success is None
        assert set(item.fail_kinds) == {"no_qed"}  # FakeAdapter's default corruption drops QED
        assert sum(item.fail_kinds.values()) == 50


def test_fixed_k_matches_independently_replayed_adapter_and_judge(tmp_path: Path) -> None:
    """Non-adaptive run: replay the exact same `SampleRequest` the runner builds (same
    `_item_seed`) against a *separate* `FakeAdapter`/`NDJudge` instance and check the recorded
    `ItemResult` matches -- an end-to-end check of the runner's wiring, not just of `FakeAdapter`
    or `NDJudge` in isolation."""
    from code.ndbench.core.types import SampleRequest

    problems = _problems()
    store = Store(tmp_path)
    _write_suite(store, problems)
    k = 40
    p = 0.4
    seed = 7
    adapter = FakeAdapter(refs=_refs(problems), p=p)
    spec = RunSpec(suite="demo", version="v1", model="fake", k=k, temperature=0.8, seed=seed)
    summary = run(spec, adapter, NDJudge(), store)
    assert summary["status"] == "complete"
    assert summary["n"] == k * len(problems)

    replay_adapter = FakeAdapter(refs=_refs(problems), p=p)
    replay_judge = NDJudge()
    by_pid = {item.pid: item for item in store.items(summary["run_id"])}
    for problem in problems:
        req = SampleRequest(
            pid=problem.pid,
            sequent=problem.sequent,
            n=k,
            temperature=0.8,
            max_new_tokens=max(1, 1024 - adapter.token_count(problem.sequent)),
            seed=_item_seed(seed, problem.pid, 0),
        )
        assert problem.ref is not None
        (attempts,) = replay_adapter.sample([req])
        verdicts = replay_judge.judge(problem.sequent, attempts, problem.ref.lines)
        expected_c = sum(v.ok for v in verdicts)
        assert by_pid[problem.pid].n == k
        assert by_pid[problem.pid].c == expected_c
        assert 0 < expected_c < k  # p=0.4 with k=40: sanity that this isn't a degenerate 0/all case


def test_same_seed_is_fully_deterministic_across_independent_stores(tmp_path: Path) -> None:
    problems = _problems()
    store1 = Store(tmp_path / "store1")
    store2 = Store(tmp_path / "store2")
    _write_suite(store1, problems)
    _write_suite(store2, problems)
    spec = RunSpec(suite="demo", version="v1", model="fake", k=64, seed=123)

    summary1 = run(spec, FakeAdapter(refs=_refs(problems), p=0.5), NDJudge(), store1)
    summary2 = run(spec, FakeAdapter(refs=_refs(problems), p=0.5), NDJudge(), store2)

    assert summary1["run_id"] == summary2["run_id"]
    assert summary1["n"] == summary2["n"]
    assert summary1["c"] == summary2["c"]

    items1 = {i.pid: i for i in store1.items(summary1["run_id"])}
    items2 = {i.pid: i for i in store2.items(summary2["run_id"])}
    assert items1.keys() == items2.keys()
    for pid in items1:
        assert items1[pid].n == items2[pid].n
        assert items1[pid].c == items2[pid].c
        assert items1[pid].ok_written == items2[pid].ok_written
        assert items1[pid].fail_kinds == items2[pid].fail_kinds


def test_different_seeds_usually_differ(tmp_path: Path) -> None:
    problems = _problems()
    store = Store(tmp_path)
    _write_suite(store, problems)
    spec_a = RunSpec(suite="demo", version="v1", model="fake", k=200, seed=1)
    spec_b = RunSpec(suite="demo", version="v1", model="fake", k=200, seed=2)
    summary_a = run(spec_a, FakeAdapter(refs=_refs(problems), p=0.5), NDJudge(), store)
    summary_b = run(spec_b, FakeAdapter(refs=_refs(problems), p=0.5), NDJudge(), store)
    assert summary_a["run_id"] != summary_b["run_id"]
    assert summary_a["c"] != summary_b["c"]  # astronomically unlikely to tie at n=400 draws


def test_missing_pid_raises_keyerror(tmp_path: Path) -> None:
    problems = _problems()
    store = Store(tmp_path)
    _write_suite(store, problems)
    adapter = FakeAdapter(refs={}, p=1.0)  # no refs at all
    spec = RunSpec(suite="demo", version="v1", model="fake", k=1, seed=0)
    with pytest.raises(KeyError):
        run(spec, adapter, NDJudge(), store)
