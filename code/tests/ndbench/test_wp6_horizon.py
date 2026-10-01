"""WP6 acceptance test for `analysis/horizon.py` (the "families" dashboard page)."""

from __future__ import annotations

from pathlib import Path

from code.ndbench.analysis import horizon
from code.ndbench.store import Store

from .test_wp6_helpers import make_item, make_problem, manifest, start_run


def _build_store(tmp_path: Path) -> Store:
    store = Store(tmp_path)
    problems = [
        make_problem(
            f"and_tower_n{n}",
            "families",
            "v1",
            length=n + 2,
            tags={"family": "and_tower", "n": n, "tokens": 40 + 10 * n},
        )
        for n in range(1, 5)
    ]
    store.write_suite(problems, manifest("families", "v1", len(problems)))
    run_id = start_run(store, suite="families", version="v1", model="base", k=32)
    # pass@8 == 1.0 for n=1..3 (c == n), 0.0 for n=4 (c == 0): n* at (k=8, tau=0.5) is exactly 3.
    items = [
        make_item(run_id, f"and_tower_n{n}", 32, 32 if n < 4 else 0, tokens_mean=40.0 + 10 * n)
        for n in range(1, 5)
    ]
    store.append_items(run_id, items)
    return store


def test_horizon_view_nonempty_and_n_star(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    data = horizon.view(store, {"B": 100, "seed": 0})
    assert data["empty"] is False
    fam = data["families"]["and_tower"]["base"]
    assert fam["n_items"] == 4
    n_star = fam["n_star"]["k8_tau0.5"]
    assert n_star["n_star"] == 3
    assert n_star["curve"] == [(1, 1.0), (2, 1.0), (3, 1.0), (4, 0.0)]
    assert "1" in fam["curves_by_k"]
    assert "8" in fam["curves_by_k"]
    assert "32" in fam["curves_by_k"]
    # a clean step function -> a well-defined (if extreme) logit slope, not None.
    assert fam["logit_slope_per_line"] is not None
    assert fam["logit_slope_per_100_tokens"] is not None


def test_horizon_deterministic_bootstrap_under_seed(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    a = horizon.view(store, {"B": 50, "seed": 3})
    b = horizon.view(store, {"B": 50, "seed": 3})
    assert a == b


def test_horizon_empty_store(tmp_path: Path) -> None:
    store = Store(tmp_path)
    data = horizon.view(store, {})
    assert data["empty"] is True


def test_horizon_splits_iltp_style_by_kind(tmp_path: Path) -> None:
    """`suites/families.py`'s "ILTP-style" family packs four structurally distinct sub-schemas
    under one `family` tag with *overlapping* `n` ranges, distinguished only by `tags["kind"]`
    (e.g. real `n=1` mixes syj204_style/syj203_style/de_bruijn_style items) -- grouping by `family`
    alone would pool pass@k across unrelated schemas at the same nominal `n`."""
    store = Store(tmp_path)
    problems = [
        make_problem(
            "iltp_a",
            "families",
            "v1",
            length=4,
            tags={"family": "ILTP-style", "n": 1, "kind": "syj204_style"},
        ),
        make_problem(
            "iltp_b",
            "families",
            "v1",
            length=8,
            tags={"family": "ILTP-style", "n": 1, "kind": "syj203_style"},
        ),
        # a plain (non-ILTP) family has no "kind" tag at all -- must be unaffected.
        make_problem("plain_a", "families", "v1", length=3, tags={"family": "and_tower", "n": 1}),
    ]
    store.write_suite(problems, manifest("families", "v1", len(problems)))
    run_id = start_run(store, suite="families", version="v1", model="base", k=8)
    store.append_items(
        run_id,
        [
            make_item(run_id, "iltp_a", 8, 8),
            make_item(run_id, "iltp_b", 8, 0),
            make_item(run_id, "plain_a", 8, 8),
        ],
    )
    data = horizon.view(store, {"B": 20, "seed": 0})
    assert "ILTP-style (syj204_style)" in data["families"]
    assert "ILTP-style (syj203_style)" in data["families"]
    assert "ILTP-style" not in data["families"]  # the unsplit, kind-blind key must not appear.
    assert "and_tower" in data["families"]  # a plain family (no "kind" tag) is unaffected.
    # each sub-schema's own n=1 item result is not pooled with the other sub-schema's.
    syj204 = data["families"]["ILTP-style (syj204_style)"]["base"]
    assert syj204["curves_by_k"]["1"] == [(1, 1.0)]
