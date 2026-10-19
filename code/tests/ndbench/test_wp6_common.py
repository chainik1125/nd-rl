"""`analysis/_common.py`'s shared run-selection helpers exclude any run whose status isn't
`"complete"`/`"partial"` by default -- in particular `"invalid"` (fix list: rerunning a completed
config silently merges into the old run; a run corrupted by that bug, or any other reason, is
marked `"invalid"` and every analysis view built on these helpers must then ignore it)."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import cast

import numpy as np

from code.ndbench.analysis._common import (
    all_run_ids,
    default_version,
    irls_binomial,
    latest_run_id,
    models_present,
    run_records,
)
from code.ndbench.store import Store

from .test_wp6_helpers import make_problem, manifest, start_run


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


def test_latest_run_id_prefers_the_larger_k_run_even_if_it_finished_earlier(tmp_path: Path) -> None:
    """WP8 integration fix list item 3: `textbook_dev`'s real store carries both the Yue benchmark's
    big adaptive run (e.g. k=4096) and a small k=8 "unsteered baseline" measurement for the interp
    steering/interpolation sweep (INTERP_SPEC.md's eval suites include `textbook_dev`) against the
    very same (suite, model, version) -- and the small baseline finishes *later*. A caller wanting
    "the real run" must not get shadowed by recency alone."""
    store = Store(tmp_path)
    big_id = start_run(store, suite="textbook_dev", version="v1", model="base", k=4096, seed=0)
    small_id = start_run(store, suite="textbook_dev", version="v1", model="base", k=8, seed=1)
    assert small_id != big_id
    assert latest_run_id(store, suite="textbook_dev", model="base") == big_id


def test_latest_run_id_still_prefers_recency_among_equal_k_reruns(tmp_path: Path) -> None:
    """A `--rerun` at the *same* k as an earlier attempt must still win on recency (unchanged
    resume/rerun behaviour)."""
    store = Store(tmp_path)
    first_id = start_run(store, suite="demo", version="v1", model="base", k=32, seed=0)
    second_id = start_run(store, suite="demo", version="v1", model="base", k=32, seed=1)
    assert first_id != second_id
    assert latest_run_id(store, suite="demo", model="base") == second_id


def test_an_all_invalid_config_key_has_no_live_runs_at_all(tmp_path: Path) -> None:
    """Every run for a (suite, model) being invalid must not fall back to showing one anyway."""
    store = Store(tmp_path)
    start_run(store, suite="demo", version="v1", model="base", k=1, seed=0, status="invalid")
    assert run_records(store, suite="demo", model="base") == []
    assert latest_run_id(store, suite="demo", model="base") is None
    assert all_run_ids(store, suite="demo", model="base") == []
    assert models_present(store, suite="demo") == []


def _write_suite_version(store: Store, suite: str, version: str) -> None:
    problems = [make_problem(f"{suite}-{version}-p0", suite, version, length=5)]
    store.write_suite(problems, manifest(suite, version, len(problems)))


def test_default_version_prefers_the_version_with_more_model_coverage(tmp_path: Path) -> None:
    """Fix list item 1: a suite can have an unrun demo version alongside the version the real runs
    target (e.g. padding's 132-item v1 demo vs. the 51-base v2 base/rl were actually run on) --
    `default_version` must pick whichever version actually has runs, not an arbitrary `"v1"`."""
    store = Store(tmp_path)
    _write_suite_version(store, "demo", "v1")
    _write_suite_version(store, "demo", "v2")
    start_run(store, suite="demo", version="v2", model="base", k=1)
    start_run(store, suite="demo", version="v2", model="rl", k=1)
    assert default_version(store, suite="demo") == "v2"
    assert default_version(store, suite="demo", models=["base", "rl"]) == "v2"


def test_default_version_falls_back_to_the_only_version_with_no_runs_at_all(tmp_path: Path) -> None:
    store = Store(tmp_path)
    _write_suite_version(store, "demo", "v1")
    assert default_version(store, suite="demo") == "v1"


def test_default_version_none_when_the_suite_does_not_exist(tmp_path: Path) -> None:
    store = Store(tmp_path)
    assert default_version(store, suite="nope") is None


def test_default_version_ignores_an_invalid_run_when_scoring_versions(tmp_path: Path) -> None:
    """A version whose only run is `invalid` must not outscore a version with a real run."""
    store = Store(tmp_path)
    _write_suite_version(store, "demo", "v1")
    _write_suite_version(store, "demo", "v2")
    start_run(store, suite="demo", version="v1", model="base", k=1, status="invalid")
    start_run(store, suite="demo", version="v2", model="base", k=1)
    assert default_version(store, suite="demo") == "v2"


def test_irls_binomial_stays_finite_under_quasi_complete_separation() -> None:
    """Fix list item 3 (via `metrics.logit_slope`'s sibling): `prefix.py`'s fitted logit
    `p = a + b*(L-j) + c*j` hit this on the real store when a model's prefix-success rate is near-0
    almost everywhere (a `base`-model intercept at se=1431.97 before the `ridge` penalty). A
    perfectly-separating two-covariate design must still return finite, bounded coefficients."""
    # intercept, lines_left, j (collinear, lines_left + j == 7 -- also exercises near-singularity).
    design = np.array([[1.0, x, 7.0 - x] for x in range(8)])
    n = np.array([32.0] * 8)
    c = np.array([32.0, 32.0, 32.0, 32.0, 0.0, 0.0, 0.0, 0.0])  # a perfect step at x=4.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        beta, se = irls_binomial(design, n, c)
    assert np.all(np.isfinite(beta))
    assert np.all(np.isfinite(se))
    assert np.all(np.abs(beta) < 50)
    assert np.all(se < 50)
    assert math.isfinite(beta[1])  # the "lines_left" coefficient in particular is well-formed.
