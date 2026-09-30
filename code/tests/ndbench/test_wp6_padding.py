"""WP6 acceptance test for `analysis/padding.py` (free lines / padding)."""

from __future__ import annotations

from pathlib import Path

from code.ndbench.analysis import padding
from code.ndbench.store import Store

from .test_wp6_helpers import make_item, make_problem, manifest, start_run


def _build_store(tmp_path: Path) -> Store:
    store = Store(tmp_path)
    padded = [
        make_problem(
            f"pad{m}", "padding", "v1", length=5 + m, tags={"pad_type": "P1-shared", "m": m}
        )
        for m in range(3)
    ]
    store.write_suite(padded, manifest("padding", "v1", len(padded)))
    run_id = start_run(store, suite="padding", version="v1", model="base", k=32)
    store.append_items(
        run_id,
        [
            make_item(run_id, "pad0", 32, 32),
            make_item(run_id, "pad1", 32, 16),
            make_item(run_id, "pad2", 32, 8),
        ],
    )

    natural = [make_problem(f"nat{5 + i}", "gen_ctrl", "v1", length=5 + i) for i in range(3)]
    store.write_suite(natural, manifest("gen_ctrl", "v1", len(natural)))
    nat_run = start_run(store, suite="gen_ctrl", version="v1", model="base", k=32)
    store.append_items(
        nat_run,
        [
            make_item(nat_run, "nat5", 32, 20),
            make_item(nat_run, "nat6", 32, 10),
            make_item(nat_run, "nat7", 32, 4),
        ],
    )
    return store


def test_padding_view_nonempty(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    data = padding.view(store, {})
    assert data["empty"] is False
    pad_type = data["pad_types"]["P1-shared"]["base"]
    assert pad_type["n_items"] == 3
    assert pad_type["padded_curve_vs_W"] == [
        {"W": 5.0, "pass1": 1.0, "n_items": 1},
        {"W": 6.0, "pass1": 0.5, "n_items": 1},
        {"W": 7.0, "pass1": 0.25, "n_items": 1},
    ]
    assert pad_type["natural_curve_vs_W"] == [
        {"W": 5.0, "pass1": 0.625, "n_items": 1},
        {"W": 6.0, "pass1": 0.3125, "n_items": 1},
        {"W": 7.0, "pass1": 0.125, "n_items": 1},
    ]
    assert pad_type["free_line_cost_per_line"] is not None


def test_padding_empty_store(tmp_path: Path) -> None:
    store = Store(tmp_path)
    data = padding.view(store, {})
    assert data["empty"] is True
