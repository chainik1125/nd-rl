"""WP6 acceptance tests for `analysis/yue.py` (plan §3 WP6 row): every view is non-empty on a
synthetic store, and "a Yue crossover is detected on synthetic curves"."""

from __future__ import annotations

from pathlib import Path

from code.ndbench.analysis import yue
from code.ndbench.analysis._common import first_chunk_by_pid
from code.ndbench.store import Store

from .test_wp6_helpers import make_item, make_problem, manifest, start_run


def _build_crossover_store(tmp_path: Path) -> Store:
    """40 `textbook_dev` items, `n=256` each (a single chunk): 'base' has a uniform p=0.1 on every
    item (mean p = pass@1 = 0.1, but `1-(1-0.1)^k` climbs steadily); 'rl' has p=0.9 on 10 items and
    p=0.0 on the other 30 (pass@1 = 0.225, higher than base's, but the aggregate plateaus near
    0.25 as k grows since 30/40 items are never solved). Hand-computed: base's curve overtakes
    rl's between k=2 (0.19 < 0.2475) and k=4 (0.344 > 0.250), i.e. `k* == 4` on the default
    `ks = 2**0..2**12` grid -- see the arithmetic in the WP6 report.
    """
    store = Store(tmp_path)
    problems = [make_problem(f"p{i}", "textbook_dev", "v1", length=8 + (i % 5)) for i in range(40)]
    store.write_suite(problems, manifest("textbook_dev", "v1", len(problems)))

    base_run = start_run(store, suite="textbook_dev", version="v1", model="base", k=256)
    rl_run = start_run(store, suite="textbook_dev", version="v1", model="rl", k=256)
    base_items = [make_item(base_run, f"p{i}", 256, round(0.1 * 256)) for i in range(40)]
    rl_items = [
        make_item(rl_run, f"p{i}", 256, round(0.9 * 256) if i < 10 else 0) for i in range(40)
    ]
    store.append_items(base_run, base_items)
    store.append_items(rl_run, rl_items)
    return store


def test_yue_view_nonempty_and_crossover_detected(tmp_path: Path) -> None:
    store = _build_crossover_store(tmp_path)
    data = yue.view(store, {"B": 200, "seed": 0})
    assert data["empty"] is False
    sets = data["sets"]
    assert "textbook_dev" in sets
    dev = sets["textbook_dev"]
    assert set(dev["models"]) == {"base", "rl"}
    assert dev["n_items"] == {"base": 40, "rl": 40}
    assert len(dev["curves"]["base"]) == len(yue.KS)
    assert dev["pair"] == ["base", "rl"]

    cross = dev["crossover"]
    assert cross is not None
    assert cross["k_star"] == 4
    assert cross["n_common"] == 40
    # the bootstrap distribution is non-empty and concentrated near the observed crossover.
    assert cross["n_boot_found"] > 0
    lo, hi = cross["ci"]
    assert lo is not None and hi is not None
    assert lo <= hi

    # base (p=0.1, n=256) is "ever solved" on all 40 items (c = round(0.1*256) = 26 > 0); rl only
    # ever solves its 10 "easy" (p=0.9) items, so rl's solved set is a strict subset of base's.
    venn = dev["venn"]
    assert venn is not None
    assert venn["n"] == 40
    assert venn["label_a"] == "base"
    assert venn["label_b"] == "rl"
    assert venn["only_b"] == 0
    assert venn["both"] == 10
    assert venn["only_a"] == 30
    assert venn["neither"] == 0
    assert venn["b_subset_of_a"] is True

    coverage = dev["coverage"]
    assert coverage["base"]["n_items"] == 40
    assert 0.0 <= coverage["base"]["coverage"] <= 1.0
    assert coverage["rl"]["coverage"] < 1.0  # 30/40 rl items never solved (p=0)

    assert dev["per_length"]["base"]  # problems carried a `length.L`, so per-length is populated


def test_yue_deterministic_bootstrap_under_seed(tmp_path: Path) -> None:
    store = _build_crossover_store(tmp_path)
    a = yue.view(store, {"B": 100, "seed": 42})
    b = yue.view(store, {"B": 100, "seed": 42})
    assert a["sets"]["textbook_dev"]["curves"] == b["sets"]["textbook_dev"]["curves"]
    assert a["sets"]["textbook_dev"]["crossover"] == b["sets"]["textbook_dev"]["crossover"]

    c = yue.view(store, {"B": 100, "seed": 7})
    # different seeds are allowed to (but need not) differ; at minimum the call must not crash and
    # must still find the same *observed* (non-bootstrapped) crossover.
    assert c["sets"]["textbook_dev"]["crossover"]["k_star"] == 4


def test_yue_empty_store_reports_no_data(tmp_path: Path) -> None:
    store = Store(tmp_path)
    data = yue.view(store, {})
    assert data["empty"] is True
    assert "reason" in data


def test_yue_legacy_consistency(tmp_path: Path) -> None:
    store = _build_crossover_store(tmp_path)
    # a distinct `seed` keeps this run's `run_id` from colliding with the "complete" run of the
    # same suite/model/k that `_build_crossover_store` already registered.
    legacy_run = start_run(
        store, suite="textbook_dev", version="v1", model="base", k=256, seed=999, status="legacy"
    )
    # legacy agrees with the current first-chunk c/n on every item (same p=0.1, same n).
    store.append_items(
        legacy_run, [make_item(legacy_run, f"p{i}", 256, round(0.1 * 256)) for i in range(40)]
    )
    data = yue.view(store, {"B": 50})
    legacy = data["legacy_consistency"]["textbook_dev"]["base"]
    assert legacy["n_common"] == 40
    assert legacy["agree_rate"] == 1.0
    assert legacy["mean_abs_diff"] == 0.0


def test_first_chunk_by_pid_reads_only_the_lowest_chunk(tmp_path: Path) -> None:
    store = Store(tmp_path)
    run_id = "r1"
    store.append_items(run_id, [make_item(run_id, "a", 256, 3, chunk=0)])
    store.append_items(run_id, [make_item(run_id, "a", 256, 1, chunk=1)])
    fc = first_chunk_by_pid(store, run_id)
    assert fc == {"a": {"n": 256, "c": 3}}
    # store.items() merges both chunks, unlike first_chunk_by_pid.
    (merged,) = store.items(run_id)
    assert merged.n == 512
    assert merged.c == 4
