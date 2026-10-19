"""WP6 acceptance test for `analysis/subst.py` (law x substitution size)."""

from __future__ import annotations

from pathlib import Path

from code.ndbench.analysis import subst
from code.ndbench.store import Store

from .test_wp6_helpers import make_item, make_problem, manifest, start_run


def _build_store(tmp_path: Path) -> Store:
    store = Store(tmp_path)
    laws = [
        make_problem(
            "law_s0",
            "laws",
            "v1",
            length=7,
            tags={"schema": "contraposition", "s": 0, "tokens": 60},
        ),
        make_problem(
            "law_s2",
            "laws",
            "v1",
            length=7,
            tags={"schema": "contraposition", "s": 2, "tokens": 160},
        ),
        make_problem(
            "law_s4",
            "laws",
            "v1",
            length=7,
            tags={"schema": "contraposition", "s": 4, "tokens": 260},
        ),
    ]
    store.write_suite(laws, manifest("laws", "v1", len(laws)))
    run_id = start_run(store, suite="laws", version="v1", model="base", k=32)
    store.append_items(
        run_id,
        [
            make_item(run_id, "law_s0", 32, 32, tokens_mean=50.0),
            make_item(run_id, "law_s2", 32, 16, tokens_mean=150.0),
            make_item(run_id, "law_s4", 32, 8, tokens_mean=250.0),
        ],
    )

    textbook = [
        make_problem("tb1", "textbook_law", "v1", length=7, tags={"schema": "contraposition"})
    ]
    store.write_suite(textbook, manifest("textbook_law", "v1", len(textbook)))
    tb_run = start_run(store, suite="textbook_law", version="v1", model="base", k=32)
    store.append_items(tb_run, [make_item(tb_run, "tb1", 32, 20)])
    return store


def test_subst_view_nonempty(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    data = subst.view(store, {})
    assert data["empty"] is False
    schema = data["schemas"]["contraposition"]["base"]
    assert schema["n_items"] == 3
    assert schema["curve_vs_s"] == [
        {"s": 0, "pass1": 1.0, "n_items": 1},
        {"s": 2, "pass1": 0.5, "n_items": 1},
        {"s": 4, "pass1": 0.25, "n_items": 1},
    ]
    assert abs(schema["controlled_pass1_overall"] - (1.0 + 0.5 + 0.25) / 3) < 1e-9
    assert schema["logit_slope_per_100_tokens"] is not None

    cmp = schema["textbook_law_comparison"]
    assert cmp is not None
    assert cmp["n_items"] == 1
    assert abs(cmp["pass1"] - 20 / 32) < 1e-9


def test_subst_empty_store(tmp_path: Path) -> None:
    store = Store(tmp_path)
    data = subst.view(store, {})
    assert data["empty"] is True


def test_subst_token_slope_uses_item_tokens_not_generated_length(tmp_path: Path) -> None:
    # Audit 2026-10-01: the x of the token slope is the item's own size, so changing the model's
    # generated length (tokens_mean) must not move the scatter points.
    store = _build_store(tmp_path)
    data = subst.view(store, {"B": 50, "seed": 0})
    pts = data["schemas"]["contraposition"]["base"]["points_vs_tokens"]
    assert sorted(p["tokens"] for p in pts) == [60.0, 160.0, 260.0]
    assert [row["s"] for row in data["schemas"]["contraposition"]["base"]["curve_vs_s"]] == [
        0,
        2,
        4,
    ]
