"""WP-D2 acceptance tests: `rejudge` (plan §2 item 3) and the `judge_id` filters added to
`analysis._common` (plan §2 item 5), on a synthetic `Store` -- no real model, no GPU.

A single `Problem` whose fixed sample body is a *premise-omitting* proof is used throughout:
`nd_verify`/`judge.nd.NDJudge` requires the `PR` block to match every declared premise exactly
(`code/ndbench/vendor/nd_verify/verify.py:158-160`), so a proof that only restates a prefix of a
2-premise sequent's premises is rejected outright, while `judge.lean.LeanJudge` -- via
`nd2lean.translate(..., require_all_pr=False)` -- accepts it (the unused second premise is just an
unused hypothesis). This is the same disagreement class `test_dan_lean_judge.py`'s
`test_real_lean_accepts_premise_omitting_proof_that_nd_v2_rejects` exercises directly on the two
judges; here it drives an actual `run()` + `rejudge` round trip through the `Store`/CLI.
"""

from __future__ import annotations

from pathlib import Path

from code.ndbench.adapters.fake import FakeAdapter
from code.ndbench.analysis._common import all_run_ids, latest_run_id, models_present
from code.ndbench.cli import main
from code.ndbench.core.io import git_commit_sha, sha256_str, utc_now_iso
from code.ndbench.core.types import LengthLabel, Problem, Sequent, SuiteManifest
from code.ndbench.judge.nd import NDJudge
from code.ndbench.runner.runner import RunSpec, run
from code.ndbench.store import Store

_SUITE = "rejudge_demo"
_VERSION = "v1"


def _write_suite(store: Store, problems: list[Problem]) -> None:
    manifest = SuiteManifest(
        suite=_SUITE,
        version=_VERSION,
        builder="test",
        config={},
        seed=0,
        created_utc=utc_now_iso(),
        git_commit=git_commit_sha(),
        n_items=len(problems),
        items_sha256=sha256_str(f"{_SUITE}-{_VERSION}-{len(problems)}"),
        excl_sources={},
        stats={},
    )
    store.write_suite(problems, manifest)


def _problem(pid: str, premises: tuple[str, ...], conclusion: str) -> Problem:
    return Problem(
        pid=pid,
        suite=_SUITE,
        version=_VERSION,
        sequent=Sequent(premises=premises, conclusion=conclusion),
        length=LengthLabel(L=1, L_ub=None, L_lb=None, method="minlen_exact", bound=None),
        ref=None,
        tags={},
    )


def test_rejudge_with_the_same_judge_reproduces_identical_counts(tmp_path: Path) -> None:
    """A proof `nd_v2` happily accepts: rejudging under `nd` again against the same samples must
    reproduce exactly the same `n`/`c`, item by item -- `nd_verify` is deterministic."""
    problems = [_problem("item0", ("P",), "P")]
    store = Store(tmp_path)
    _write_suite(store, problems)
    adapter = FakeAdapter(refs={"item0": "N1 P : PR ; QED"}, p=1.0)
    spec = RunSpec(suite=_SUITE, version=_VERSION, model="fake", k=5, seed=0)
    summary = run(spec, adapter, NDJudge(), store)
    assert summary["status"] == "complete"
    assert summary["n"] == 5
    assert summary["c"] == 5  # nd_v2 accepts the full-premise proof every time

    rc = main(["rejudge", "--run-id", summary["run_id"], "--judge", "nd", "--out", str(tmp_path)])
    assert rc == 0

    derived_id = f"{summary['run_id']}~nd_v2"
    derived_items = store.items(derived_id)
    assert len(derived_items) == 1
    assert derived_items[0].n == 5
    assert derived_items[0].c == 5  # agrees with the original run exactly, item by item

    derived_run = store.runs(run_id=derived_id)[0]
    assert derived_run["derived_from"] == summary["run_id"]
    assert derived_run["judge_id"] == "nd_v2"
    assert derived_run["suite"] == _SUITE
    assert derived_run["version"] == _VERSION
    assert derived_run["model"] == "fake"
    assert derived_run["status"] == "complete"

    # the original run's own rows are untouched.
    original_items = store.items(summary["run_id"])
    assert original_items[0].n == 5 and original_items[0].c == 5


def test_rejudge_with_a_different_judge_gives_different_counts(tmp_path: Path) -> None:
    """A premise-omitting proof (module docstring): `nd_v2` rejects every attempt, `lean_v2`
    accepts every attempt -- `rejudge --judge lean` must flip the counts, not reproduce them, and
    the `judge_id` filters (plan §2 item 5) must tell the two runs apart afterwards."""
    problems = [_problem("item0", ("P", "Q"), "P")]
    store = Store(tmp_path)
    _write_suite(store, problems)
    adapter = FakeAdapter(refs={"item0": "N1 P : PR ; QED"}, p=1.0)
    spec = RunSpec(suite=_SUITE, version=_VERSION, model="fake", k=3, seed=0)
    summary = run(spec, adapter, NDJudge(), store)
    assert summary["n"] == 3
    assert summary["c"] == 0  # nd_v2: the PR block omits the declared second premise

    rc = main(["rejudge", "--run-id", summary["run_id"], "--judge", "lean", "--out", str(tmp_path)])
    assert rc == 0

    derived_id = f"{summary['run_id']}~lean_v2"
    derived_items = store.items(derived_id)
    assert derived_items[0].n == 3
    assert derived_items[0].c == 3  # lean_v2: the unused second premise doesn't block acceptance

    # the sqlite verdict cache was created under this store's own root, not somewhere ad hoc.
    assert (tmp_path / "cache" / "lean_v2" / "verdicts.sqlite").exists()

    # judge_id filters (plan §2 item 5): each judge's own run is found only under its own filter,
    # and the original run is untouched by any of this.
    original_items = store.items(summary["run_id"])
    assert original_items[0].n == 3 and original_items[0].c == 0

    assert (
        latest_run_id(store, suite=_SUITE, model="fake", version=_VERSION) == summary["run_id"]
    )  # default judge_id filter ("nd_v2") still finds the original run
    assert (
        latest_run_id(store, suite=_SUITE, model="fake", version=_VERSION, judge_id="nd_v2")
        == summary["run_id"]
    )
    assert (
        latest_run_id(store, suite=_SUITE, model="fake", version=_VERSION, judge_id="lean_v2")
        == derived_id
    )
    assert latest_run_id(store, suite=_SUITE, model="fake", version=_VERSION, judge_id="x") is None
    assert set(all_run_ids(store, suite=_SUITE, model="fake", version=_VERSION, judge_id=None)) == {
        summary["run_id"],
        derived_id,
    }
    assert models_present(store, suite=_SUITE, version=_VERSION, judge_id="lean_v2") == ["fake"]
    assert models_present(store, suite=_SUITE, version=_VERSION, judge_id="nd_v2") == ["fake"]


def test_rejudge_unknown_run_id_fails_cleanly(tmp_path: Path) -> None:
    store = Store(tmp_path)
    del store  # just ensures the Store directories exist; rejudge itself constructs its own
    rc = main(["rejudge", "--run-id", "does-not-exist", "--judge", "nd", "--out", str(tmp_path)])
    assert rc != 0


def test_new_derived_run_raises_for_a_missing_source_run(tmp_path: Path) -> None:
    store = Store(tmp_path)
    try:
        store.new_derived_run(derived_from="nope", judge_id="nd_v2")
    except ValueError as e:
        assert "nope" in str(e)
    else:
        raise AssertionError("expected ValueError")
