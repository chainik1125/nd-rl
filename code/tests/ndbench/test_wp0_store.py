"""`Store`: suite round trip, run event folding + `latest`, item chunk merging, sample gzip
append, and the GPU ledger (plan §3 WP0 row: "store append/merge/latest")."""

from __future__ import annotations

import json
from pathlib import Path

from code.ndbench.core.io import git_commit_sha, read_jsonl, sha256_str, utc_now_iso
from code.ndbench.core.types import ItemResult, LengthLabel, Problem, Sequent, SuiteManifest
from code.ndbench.store import Store


def _manifest(n_items: int) -> SuiteManifest:
    return SuiteManifest(
        suite="demo",
        version="v1",
        builder="test",
        config={"k": 1},
        seed=0,
        created_utc=utc_now_iso(),
        git_commit=git_commit_sha(),
        n_items=n_items,
        items_sha256=sha256_str("x"),
        excl_sources={},
        stats={},
    )


def _problem(pid: str) -> Problem:
    return Problem(
        pid=pid,
        suite="demo",
        version="v1",
        sequent=Sequent(premises=("P",), conclusion="P"),
        length=LengthLabel(L=1, L_ub=None, L_lb=None, method="minlen_exact", bound=None),
        ref=None,
        tags={"x": 1},
    )


def test_write_and_load_suite_roundtrip(tmp_path: Path) -> None:
    store = Store(tmp_path)
    problems = [_problem("a"), _problem("b")]
    store.write_suite(problems, _manifest(len(problems)))
    loaded = store.load_suite("demo", "v1")
    assert loaded == problems
    manifest_path = tmp_path / "suites" / "demo" / "v1" / "manifest.json"
    assert manifest_path.exists()
    assert json.loads(manifest_path.read_text())["n_items"] == 2


def test_new_run_is_stable_given_the_same_config(tmp_path: Path) -> None:
    from code.ndbench.runner.runner import RunSpec

    store = Store(tmp_path)
    spec = RunSpec(suite="demo", version="v1", model="fake", k=10, seed=0)
    identity = {"label": "fake", "p": 0.5}
    run_id_1 = store.new_run(spec, identity, "nd_v1")
    run_id_2 = store.new_run(spec, identity, "nd_v1")
    assert run_id_1 == run_id_2

    other_identity = {"label": "fake", "p": 0.9}
    run_id_3 = store.new_run(spec, other_identity, "nd_v1")
    assert run_id_3 != run_id_1


def test_runs_and_latest_fold_start_and_finish_events(tmp_path: Path) -> None:
    from code.ndbench.runner.runner import RunSpec

    store = Store(tmp_path)
    spec = RunSpec(suite="demo", version="v1", model="fake", k=10, seed=0)
    identity = {"label": "fake"}
    key = store.config_key(spec, identity, "nd_v1")
    run_id = store.new_run(spec, identity, "nd_v1")

    running = store.runs(run_id=run_id)
    assert len(running) == 1
    assert running[0]["status"] == "running"

    store.finish_run(run_id, "complete", 12.5)
    finished = store.runs(run_id=run_id)
    assert len(finished) == 1
    assert finished[0]["status"] == "complete"
    assert finished[0]["gpu_s"] == 12.5
    assert finished[0]["suite"] == "demo"  # fields from the start event are preserved

    assert store.latest("demo", "fake", key) == run_id
    assert store.latest("demo", "fake", "not-a-real-key") is None


def test_append_items_merges_chunks(tmp_path: Path) -> None:
    store = Store(tmp_path)
    run_id = "r1"
    chunk0 = ItemResult(
        run_id=run_id,
        pid="a",
        chunk=0,
        n=5,
        c=2,
        first_success=3,
        ok_written=[4, 5],
        ok_pruned=[4, 4],
        ok_tokens=[10, 11],
        ok_starts=[1, 1],
        fail_kinds={"no_qed": 3},
        fail_lines=[],
        long_enough=1,
        tokens_mean=9.0,
        hit_max=0,
    )
    chunk1 = ItemResult(
        run_id=run_id,
        pid="a",
        chunk=1,
        n=5,
        c=1,
        first_success=2,
        ok_written=[4],
        ok_pruned=[4],
        ok_tokens=[10],
        ok_starts=[1],
        fail_kinds={"bad_line": 4},
        fail_lines=[3, 3, 3, 3],
        long_enough=0,
        tokens_mean=8.0,
        hit_max=1,
    )
    store.append_items(run_id, [chunk0])
    store.append_items(run_id, [chunk1])

    (merged,) = store.items(run_id)
    assert merged.n == 10
    assert merged.c == 3
    assert merged.first_success == 3  # local index 3 within chunk 0, chunk 0 had no prior n
    assert merged.ok_written == [4, 5, 4]
    assert merged.fail_kinds == {"no_qed": 3, "bad_line": 4}
    assert merged.fail_lines == [3, 3, 3, 3]
    assert merged.long_enough == 1
    assert merged.hit_max == 1
    assert merged.tokens_mean == (9.0 * 5 + 8.0 * 5) / 10
    assert merged.chunk == -1  # merged-across-chunks sentinel


def test_append_items_first_success_offset_by_prior_chunks(tmp_path: Path) -> None:
    store = Store(tmp_path)
    run_id = "r2"
    chunk0 = ItemResult(
        run_id=run_id,
        pid="a",
        chunk=0,
        n=5,
        c=0,
        first_success=None,
        ok_written=[],
        ok_pruned=[],
        ok_tokens=[],
        ok_starts=[],
        fail_kinds={"no_qed": 5},
        fail_lines=[],
        long_enough=0,
        tokens_mean=1.0,
        hit_max=0,
    )
    chunk1 = ItemResult(
        run_id=run_id,
        pid="a",
        chunk=1,
        n=5,
        c=1,
        first_success=2,
        ok_written=[3],
        ok_pruned=[3],
        ok_tokens=[9],
        ok_starts=[1],
        fail_kinds={"no_qed": 4},
        fail_lines=[],
        long_enough=0,
        tokens_mean=1.0,
        hit_max=0,
    )
    store.append_items(run_id, [chunk0, chunk1])
    (merged,) = store.items(run_id)
    assert (
        merged.first_success == 5 + 2
    )  # 5 draws in chunk 0 (all failed) + local index 2 in chunk 1


def test_append_samples_multiple_chunks_readable_as_one_gz_stream(tmp_path: Path) -> None:
    store = Store(tmp_path)
    run_id = "r3"
    store.append_samples(run_id, [{"i": 0, "ok": True}])
    store.append_samples(run_id, [{"i": 1, "ok": False}, {"i": 2, "ok": True}])
    path = store.samples_dir / f"{run_id}.jsonl.gz"
    rows = list(read_jsonl(path))
    assert [r["i"] for r in rows] == [0, 1, 2]


def test_ledger(tmp_path: Path) -> None:
    store = Store(tmp_path)
    assert store.ledger_total_s() == 0.0
    store.ledger_append({"job": "G0", "gpu_s": 120.0})
    store.ledger_append({"job": "G1", "gpu_s": 480.0})
    store.ledger_append({"job": "note-only"})  # no gpu_s: should count as 0, not error
    assert store.ledger_total_s() == 600.0


# ------------------------------------------------------ effective config_key / invalid runs (fix)


def test_config_key_fills_in_the_max_new_policy_default(tmp_path: Path) -> None:
    """Fix: rerunning a completed config silently merged into the old run because `config_key`
    hashed only explicit `spec.options`, so an implicit runner default (e.g. `max_new_policy`)
    never showed up in the key. An implicit default must hash the same as the same value set
    explicitly, and a genuinely different value must hash differently."""
    from code.ndbench.runner.runner import RunSpec

    store = Store(tmp_path)
    identity = {"label": "fake"}
    spec_implicit = RunSpec(suite="demo", version="v1", model="fake", k=10, seed=0)
    spec_explicit_ref3x = RunSpec(
        suite="demo", version="v1", model="fake", k=10, seed=0, options={"max_new_policy": "ref3x"}
    )
    spec_explicit_full = RunSpec(
        suite="demo", version="v1", model="fake", k=10, seed=0, options={"max_new_policy": "full"}
    )
    key_implicit = store.config_key(spec_implicit, identity, "nd_v1")
    key_ref3x = store.config_key(spec_explicit_ref3x, identity, "nd_v1")
    key_full = store.config_key(spec_explicit_full, identity, "nd_v1")
    assert key_implicit == key_ref3x
    assert key_implicit != key_full


def test_config_key_ignores_items_per_call(tmp_path: Path) -> None:
    """`items_per_call` only affects scheduling (how items are grouped into `adapter.sample`
    calls), never what gets sampled, so it must not fork the config key -- resume/skip-if-complete
    must still match across two runs that only differ in this option."""
    from code.ndbench.runner.runner import RunSpec

    store = Store(tmp_path)
    identity = {"label": "fake"}
    spec_a = RunSpec(
        suite="demo", version="v1", model="fake", k=10, seed=0, options={"items_per_call": 64}
    )
    spec_b = RunSpec(
        suite="demo", version="v1", model="fake", k=10, seed=0, options={"items_per_call": 512}
    )
    spec_default = RunSpec(suite="demo", version="v1", model="fake", k=10, seed=0)
    key_a = store.config_key(spec_a, identity, "nd_v1")
    key_b = store.config_key(spec_b, identity, "nd_v1")
    key_default = store.config_key(spec_default, identity, "nd_v1")
    assert key_a == key_b == key_default


def test_new_run_accepts_an_explicit_run_id_override(tmp_path: Path) -> None:
    from code.ndbench.runner.runner import RunSpec

    store = Store(tmp_path)
    spec = RunSpec(suite="demo", version="v1", model="fake", k=10, seed=0)
    identity = {"label": "fake"}
    base_run_id = store.new_run(spec, identity, "nd_v1")
    rerun_id = store.new_run(spec, identity, "nd_v1", run_id=f"{base_run_id}-r2")
    assert rerun_id == f"{base_run_id}-r2"
    (base_record,) = store.runs(run_id=base_run_id)
    (rerun_record,) = store.runs(run_id=rerun_id)
    # both events carry the same config_key (they're the same logical config), just different ids.
    assert base_record["config_key"] == rerun_record["config_key"]


def test_latest_excludes_an_invalid_run(tmp_path: Path) -> None:
    from code.ndbench.runner.runner import RunSpec

    store = Store(tmp_path)
    spec = RunSpec(suite="demo", version="v1", model="fake", k=10, seed=0)
    identity = {"label": "fake"}
    key = store.config_key(spec, identity, "nd_v1")
    run_id = store.new_run(spec, identity, "nd_v1")
    store.finish_run(run_id, "invalid", 1.0)
    assert store.latest("demo", "fake", key) is None


def test_latest_falls_back_to_an_older_valid_run_when_the_newest_is_invalid(
    tmp_path: Path,
) -> None:
    from code.ndbench.runner.runner import RunSpec

    store = Store(tmp_path)
    spec = RunSpec(suite="demo", version="v1", model="fake", k=10, seed=0)
    identity = {"label": "fake"}
    key = store.config_key(spec, identity, "nd_v1")
    base_run_id = store.new_run(spec, identity, "nd_v1")
    store.finish_run(base_run_id, "complete", 1.0)
    rerun_id = store.new_run(spec, identity, "nd_v1", run_id=f"{base_run_id}-r2")
    store.finish_run(rerun_id, "invalid", 1.0)
    # `rerun_id` is chronologically newer but invalid, so `latest` falls back to `base_run_id`.
    assert store.latest("demo", "fake", key) == base_run_id
