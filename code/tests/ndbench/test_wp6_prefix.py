"""WP6 acceptance test for `analysis/prefix.py` (prefix completion)."""

from __future__ import annotations

import math
from pathlib import Path

from code.ndbench.analysis import prefix
from code.ndbench.store import Store

from .test_wp6_helpers import make_item, make_problem, manifest, start_run


def _build_store(tmp_path: Path) -> Store:
    store = Store(tmp_path)
    problems = [
        make_problem("pfx_a", "prefix", "v1", tags={"j": 2, "L": 10}),
        make_problem("pfx_b", "prefix", "v1", tags={"j": 5, "L": 10}),
        make_problem("pfx_c", "prefix", "v1", tags={"j": 2, "L": 14}),
    ]
    store.write_suite(problems, manifest("prefix", "v1", len(problems)))
    run_id = start_run(store, suite="prefix", version="v1", model="base", k=10)
    store.append_items(
        run_id,
        [
            make_item(run_id, "pfx_a", 10, 6, ok_written=[10] * 6, fail_lines=[3, 3, 4, 10]),
            make_item(run_id, "pfx_b", 10, 5, ok_written=[10] * 5, fail_lines=[6, 7, 8, 9, 10]),
            make_item(run_id, "pfx_c", 10, 3, ok_written=[14] * 3, fail_lines=[5] * 7),
        ],
    )
    return store


def test_prefix_view_nonempty(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    data = prefix.view(store, {})
    assert data["empty"] is False
    base = data["models"]["base"]
    assert base["n_items"] == 3

    by_j = base["success_by_j"]
    assert abs(by_j["2.0"]["pass1"] - 9 / 20) < 1e-9  # (6+3) / (10+10)
    assert by_j["2.0"]["n_items"] == 2
    assert by_j["5.0"]["pass1"] == 0.5

    by_ll = base["success_by_lines_left"]
    assert by_ll["8.0"]["pass1"] == 0.6  # L - j = 10 - 2
    assert by_ll["5.0"]["pass1"] == 0.5  # 10 - 5
    assert by_ll["12.0"]["pass1"] == 0.3  # 14 - 2

    by_l = base["success_by_L"]
    assert abs(by_l["10.0"]["pass1"] - 11 / 20) < 1e-9
    assert by_l["14.0"]["pass1"] == 0.3

    hazard = base["hazard"]
    assert hazard  # non-empty
    assert all({"line", "at_risk", "failed", "hazard"} <= set(row) for row in hazard)

    fitted = base["fitted_logit"]
    assert set(fitted) == {"intercept", "lines_left", "j"}
    for coef in fitted.values():
        assert math.isfinite(coef["coef"])
        assert math.isfinite(coef["se"])


def test_prefix_empty_store(tmp_path: Path) -> None:
    store = Store(tmp_path)
    data = prefix.view(store, {})
    assert data["empty"] is True
