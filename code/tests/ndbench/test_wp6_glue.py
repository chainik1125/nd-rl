"""WP6 acceptance test for `analysis/glue.py` (the glued/compositionality benchmark)."""

from __future__ import annotations

from pathlib import Path

from code.ndbench.analysis import glue
from code.ndbench.store import Store

from .test_wp6_helpers import make_item, make_problem, manifest, start_run


def _build_store(tmp_path: Path) -> Store:
    store = Store(tmp_path)

    parts = [
        make_problem("part_a", "parts_pool", "v1", length=3, ref_lines=3),
        make_problem("part_b", "parts_pool", "v1", length=4, ref_lines=4),
    ]
    store.write_suite(parts, manifest("parts_pool", "v1", len(parts)))
    parts_run = start_run(store, suite="parts_pool", version="v1", model="base", k=32)
    store.append_items(
        parts_run,
        [make_item(parts_run, "part_a", 32, 30), make_item(parts_run, "part_b", 32, 25)],
    )

    joined = [
        make_problem(
            "joined1",
            "glue",
            "v1",
            length=8,
            tags={"op": "AND", "k": 2, "part_len": 3, "cell": "AND_k2"},
            parts=("part_a", "part_b"),
        )
    ]
    store.write_suite(joined, manifest("glue", "v1", len(joined)))
    joined_run = start_run(store, suite="glue", version="v1", model="base", k=32)
    # fail_lines: 2 -> part_a's range [1,3]; 5 -> part_b's range [4,7]; 8 -> past both -> "glue".
    store.append_items(
        joined_run,
        [make_item(joined_run, "joined1", 32, 16, long_enough=20, fail_lines=[2, 5, 8])],
    )

    control = [make_problem("ctrl1", "gen_ctrl", "v1", length=8)]
    store.write_suite(control, manifest("gen_ctrl", "v1", len(control)))
    ctrl_run = start_run(store, suite="gen_ctrl", version="v1", model="base", k=32)
    store.append_items(ctrl_run, [make_item(ctrl_run, "ctrl1", 32, 10)])

    return store


def test_glue_view_nonempty(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    data = glue.view(store, {"B": 100, "seed": 0})
    assert data["empty"] is False
    cell = data["cells"]["AND_k2"]["base"]
    assert cell["n_joined_items"] == 1
    assert cell["n_mastery_groups"] == 1  # both parts mastered (phat >= 0.5)

    comp = cell["compositionality"]
    assert comp is not None
    assert comp["n_used"] == 1
    # R = p_joined / (p_a * p_b) = 0.5 / (0.9375 * 0.78125)
    assert abs(comp["R"] - (0.5 / (0.9375 * 0.78125))) < 1e-9
    # R_min = p_joined / min(p_a, p_b) = 0.5 / 0.78125
    assert abs(comp["R_min"] - (0.5 / 0.78125)) < 1e-9

    assert abs(cell["joined_pass1"] - 0.5) < 1e-9
    assert abs(cell["frac_long_enough"] - 20 / 32) < 1e-9

    fail_loc = cell["failure_location"]
    assert fail_loc == {"part_a": 1, "part_b": 1, "glue": 1}

    ctrl = cell["control_pass1"]
    assert ctrl is not None
    assert ctrl["n_items"] == 1
    assert abs(ctrl["pass1"] - 10 / 32) < 1e-9


def test_glue_empty_store(tmp_path: Path) -> None:
    store = Store(tmp_path)
    data = glue.view(store, {})
    assert data["empty"] is True
