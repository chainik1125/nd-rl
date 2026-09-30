"""`analysis/_common.py`'s shared run-selection helpers exclude any run whose status isn't
`"complete"`/`"partial"` by default -- in particular `"invalid"` (fix list: rerunning a completed
config silently merges into the old run; a run corrupted by that bug, or any other reason, is
marked `"invalid"` and every analysis view built on these helpers must then ignore it)."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from code.ndbench.analysis._common import all_run_ids, latest_run_id, models_present, run_records
from code.ndbench.store import Store

from .test_wp6_helpers import start_run


def test_run_records_excludes_an_invalid_run_by_default(tmp_path: Path) -> None:
    store = Store(tmp_path)
    complete_id = start_run(store, suite="demo", version="v1", model="base", k=1, seed=0)
    invalid_id = start_run(
        store, suite="demo", version="v1", model="base", k=1, seed=1, status="invalid"
    )
    ids = {cast(str, r["run_id"]) for r in run_records(store, suite="demo", model="base")}
    assert complete_id in ids
    assert invalid_id not in ids


def test_latest_run_id_skips_an_invalid_run(tmp_path: Path) -> None:
    store = Store(tmp_path)
    complete_id = start_run(store, suite="demo", version="v1", model="base", k=1, seed=0)
    # A later, invalid run for the same (suite, model) must not shadow the still-valid one.
    start_run(store, suite="demo", version="v1", model="base", k=1, seed=1, status="invalid")
    assert latest_run_id(store, suite="demo", model="base") == complete_id


def test_all_run_ids_excludes_an_invalid_run(tmp_path: Path) -> None:
    store = Store(tmp_path)
    complete_id = start_run(store, suite="demo", version="v1", model="base", k=1, seed=0)
    invalid_id = start_run(
        store, suite="demo", version="v1", model="base", k=1, seed=1, status="invalid"
    )
    ids = all_run_ids(store, suite="demo", model="base")
    assert complete_id in ids
    assert invalid_id not in ids


def test_models_present_ignores_a_model_with_only_an_invalid_run(tmp_path: Path) -> None:
    store = Store(tmp_path)
    start_run(store, suite="demo", version="v1", model="base", k=1, seed=0)
    start_run(store, suite="demo", version="v1", model="rl", k=1, seed=0, status="invalid")
    assert models_present(store, suite="demo") == ["base"]


def test_an_all_invalid_config_key_has_no_live_runs_at_all(tmp_path: Path) -> None:
    """Every run for a (suite, model) being invalid must not fall back to showing one anyway."""
    store = Store(tmp_path)
    start_run(store, suite="demo", version="v1", model="base", k=1, seed=0, status="invalid")
    assert run_records(store, suite="demo", model="base") == []
    assert latest_run_id(store, suite="demo", model="base") is None
    assert all_run_ids(store, suite="demo", model="base") == []
    assert models_present(store, suite="demo") == []
