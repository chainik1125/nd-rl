"""Acceptance tests for `dash/summary.py` (the single-page "one chart per benchmark" dashboard):

- builds from a synthetic store with every one of the 12 charts populated;
- builds from an empty store, where every chart shows a "no data yet" panel;
- exactly 12 chart cards either way;
- no external URLs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from code.ndbench.analysis import interp as interp_view
from code.ndbench.core.types import JSON
from code.ndbench.dash import summary as dash_summary
from code.ndbench.edl import run_edl
from code.ndbench.interp import eval_suites
from code.ndbench.interp._store import new_interp_run, write_interp_rows
from code.ndbench.store import Store

from .test_interp_dash import _likelihood_row  # pyright: ignore[reportPrivateUsage]
from .test_wp5_fixtures import ConstantNllAdapter
from .test_wp6_dash import build_full_demo_store
from .test_wp6_edl import _toy_problems  # pyright: ignore[reportPrivateUsage]
from .test_wp6_helpers import make_item, make_problem, manifest, start_run

_N_CHARTS = 12


def _count_chart_cards(html: str) -> int:
    return len(re.findall(r'<figure class="card chart-card', html))


# --------------------------------------------------------------------------------- empty store


def test_build_summary_on_empty_store_shows_no_data_yet(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    out_dir = tmp_path / "dash_summary"
    path = dash_summary.build_summary(store, out_dir)
    assert path == out_dir / "summary.html"
    html = path.read_text()

    assert "<title>ND Capability Benchmarks</title>" in html
    assert _count_chart_cards(html) == _N_CHARTS
    assert html.count("no data yet") == _N_CHARTS
    assert "http://" not in html
    assert "https://" not in html


# ------------------------------------------------------------------------------ populated store


def _add_rl_run(
    store: Store, *, suite: str, version: str, pid_ns: list[tuple[str, int, int]]
) -> None:
    """A second (`"rl"`) run on a suite `build_full_demo_store` already wrote, reusing its own
    pids -- every summary chart compares (at least) base vs rl, so each benchmark the demo store
    covers needs an `"rl"` run alongside its existing `"base"` one for its chart to render real
    content rather than a single-series/partial one."""
    run_id = start_run(store, suite=suite, version=version, model="rl", k=32)
    store.append_items(run_id, [make_item(run_id, pid, n, c) for pid, n, c in pid_ns])


def _populate_families(store: Store) -> None:
    """Replaces `build_full_demo_store`'s minimal (1-item-per-`n`) `families` suite with one that
    clears chart 2's `_MIN_ITEMS_PER_N` (5) threshold, and exercises all three of its horizon
    outcomes: base clears n=1,2 then fails at n=3 (an ordinary finite horizon), rl fails at every
    tested n ("fails at smallest size" / `n_star=None`), armC clears every tested n (a censored,
    ">=" horizon -- the true horizon could be larger than what was tested)."""
    probs = [
        make_problem(
            f"and_tower_n{n}_{i}",
            "families",
            "v1",
            length=n + 2,
            tags={"family": "and_tower", "n": n},
        )
        for n in (1, 2, 3)
        for i in range(5)
    ]
    store.write_suite(probs, manifest("families", "v1", len(probs)))
    for model, cs in (("base", (30, 28, 0)), ("rl", (0, 0, 0)), ("armC", (30, 30, 30))):
        run_id = start_run(store, suite="families", version="v1", model=model, k=32)
        store.append_items(
            run_id,
            [
                make_item(run_id, f"and_tower_n{n}_{i}", 32, c)
                for n, c in zip((1, 2, 3), cs, strict=True)
                for i in range(5)
            ],
        )


def _populate_skills(store: Store) -> None:
    """Replaces `build_full_demo_store`'s 2-item `skill_pool` suite with >=20 items per (length,
    skill) bucket -- chart 4's new `_MIN_SKILL_CELL_ITEMS` threshold -- across all three of its
    buckets: "no special skill" (L=8), "needs case split" (L=9), "needs classical" (L=10)."""
    probs = (
        [make_problem(f"none_{i}", "skill_pool", "v1", length=8) for i in range(20)]
        + [
            make_problem(f"case_{i}", "skill_pool", "v1", length=9, tags={"case_split": True})
            for i in range(20)
        ]
        + [
            make_problem(f"classical_{i}", "skill_pool", "v1", length=10, tags={"classical": True})
            for i in range(20)
        ]
    )
    store.write_suite(probs, manifest("skill_pool", "v1", len(probs)))
    for model, c_none, c_case, c_classical in (("base", 10, 1, 2), ("rl", 16, 0, 1)):
        run_id = start_run(store, suite="skill_pool", version="v1", model=model, k=32)
        store.append_items(
            run_id,
            [make_item(run_id, f"none_{i}", 32, c_none) for i in range(20)]
            + [make_item(run_id, f"case_{i}", 32, c_case) for i in range(20)]
            + [make_item(run_id, f"classical_{i}", 32, c_classical) for i in range(20)],
        )


def _populate_laws(store: Store) -> None:
    """Replaces `build_full_demo_store`'s 3-item `laws` suite with >=20 items per s -- chart 5's
    `_MIN_ITEMS_PER_BIN` threshold -- at s=0/2/4 for one schema, plus one seen-in-training item
    the view must exclude."""
    from dataclasses import replace

    probs = [
        make_problem(f"law_s{s}_{i}", "laws", "v1", length=7, tags={"schema": "mt", "s": s})
        for s in (0, 2, 4)
        for i in range(20)
    ]
    seen = replace(
        make_problem("law_seen", "laws", "v1", length=7, tags={"schema": "mt", "s": 0}),
        seen_train=True,
    )
    store.write_suite([*probs, seen], manifest("laws", "v1", len(probs) + 1))
    for model, cs in (("base", (20, 10, 1)), ("rl", (28, 14, 2))):
        run_id = start_run(store, suite="laws", version="v1", model=model, k=32)
        store.append_items(
            run_id,
            [
                make_item(run_id, f"law_s{s}_{i}", 32, c)
                for s, c in zip((0, 2, 4), cs, strict=True)
                for i in range(20)
            ]
            + [make_item(run_id, "law_seen", 32, 32)],
        )


def _populate_prefix(store: Store) -> None:
    """Replaces `build_full_demo_store`'s 2-item `prefix` suite with >=20 items per lines-left
    bucket -- chart 7's new `_MIN_PREFIX_BIN_ITEMS` threshold -- at lines_left=1 (j=4, L=5) and
    lines_left=3 (j=2, L=5)."""
    probs = [make_problem(f"ll1_{i}", "prefix", "v1", tags={"j": 4, "L": 5}) for i in range(20)] + [
        make_problem(f"ll3_{i}", "prefix", "v1", tags={"j": 2, "L": 5}) for i in range(20)
    ]
    store.write_suite(probs, manifest("prefix", "v1", len(probs)))
    for model, c_ll1, c_ll3 in (("base", 2, 0), ("rl", 10, 3)):
        run_id = start_run(store, suite="prefix", version="v1", model=model, k=10)
        store.append_items(
            run_id,
            [make_item(run_id, f"ll1_{i}", 10, c_ll1) for i in range(20)]
            + [make_item(run_id, f"ll3_{i}", 10, c_ll3) for i in range(20)],
        )


def _populate_glue_operators(store: Store) -> None:
    """Extends `build_full_demo_store`'s single `op=AND` glue item with one item per remaining
    operator (`CHAIN`, `CASE-dilemma`, `BOX`) -- chart 3 pools R per operator across
    `{"AND", "CHAIN", "CASE", "BOX"}`, so a synthetic store exercising only `AND` would leave 3 of
    its 4 bars untested."""
    joined = [
        make_problem(
            "joined1",
            "glue",
            "v1",
            length=8,
            tags={"op": "AND", "k": 2, "part_len": 3, "cell": "AND_k2"},
            parts=("part_a", "part_b"),
        ),
        make_problem(
            "joined2",
            "glue",
            "v1",
            length=6,
            tags={"op": "CHAIN", "k": 2, "part_len": 3, "cell": "CHAIN_k2"},
            parts=("part_a", "part_b"),
        ),
        make_problem(
            "joined3",
            "glue",
            "v1",
            length=9,
            tags={"op": "CASE-dilemma", "part_len": 3, "cell": "CASE_dilemma"},
            parts=("part_a", "part_b"),
        ),
        make_problem(
            "joined4",
            "glue",
            "v1",
            length=5,
            tags={"op": "BOX", "part_len": 3, "cell": "BOX_d1"},
            parts=("part_a", "part_b"),
        ),
    ]
    store.write_suite(joined, manifest("glue", "v1", len(joined)))
    for model, cs in (("base", (16, 10, 0, 20)), ("rl", (24, 18, 0, 28))):
        run_id = start_run(store, suite="glue", version="v1", model=model, k=32)
        store.append_items(
            run_id,
            [
                make_item(run_id, pid, 32, c)
                for pid, c in zip(("joined1", "joined2", "joined3", "joined4"), cs, strict=True)
            ],
        )


def _populate_likelihood(store: Store) -> None:
    """Rows in the real `interp likelihood` row schema (module docstring's `_likelihood_row`),
    for both of chart 9's groups: `base_verified` pids resolved back to `parts_pool` (`part_a`/
    `part_b`, written by `build_full_demo_store`) and `rl_verified` pids (the crux set,
    `p_hat_base=0.0`) resolved back to `gen_ctrl` (`ctrl1`/`nat5`) -- exactly the join
    `dash/summary.py:_likelihood_crux_data` performs via the run's own recorded
    `cfg["model_samples"]`."""
    rows = [
        _likelihood_row("part_a", "base_verified", -2.0, -1.5, None),
        _likelihood_row("part_b", "base_verified", -3.0, -2.0, None),
        _likelihood_row("ctrl1", "rl_verified", -40.0, -8.0, 0.0),
        _likelihood_row("nat5", "rl_verified", -30.0, -6.0, 0.0),
    ]
    # `_likelihood_row(pid, ...)` builds `"pid": pid` directly; chart 9 expects the
    # `f"{origin}:{bare_pid}"` display-pid convention `interp/likelihood.py:run_likelihood` uses.
    for row, origin_pid in zip(
        rows,
        ("base_verified:part_a", "base_verified:part_b", "rl_verified:ctrl1", "rl_verified:nat5"),
        strict=True,
    ):
        row["pid"] = origin_pid
    cfg: dict[str, JSON] = {
        "model_samples": {
            "base": {"suite": "parts_pool", "version": "v1"},
            "rl": {"suite": "gen_ctrl", "version": "v1"},
        }
    }
    run_id = new_interp_run(store, "likelihood", cfg)
    write_interp_rows(store, "likelihood", run_id, rows)


def _populate_steering(store: Store) -> None:
    """`steer:<layer>:<alpha>` runs on `textbook_dev` (one of `view_steering`'s default pooled
    suites, already written with `dev0..dev3` by `build_full_demo_store`) plus the RL `-delta`
    control (`steer:<layer>:-1`, `view_steering`'s own convention for "steer rl, not base")."""
    for model, c in (("steer:0:0.25", 2), ("steer:0:1.0", 6), ("steer:0:-1", 18)):
        run_id = start_run(store, suite="textbook_dev", version="v1", model=model, k=32)
        store.append_items(run_id, [make_item(run_id, f"dev{i}", 32, c) for i in range(4)])


def _populate_interpolation(store: Store) -> None:
    """Fresh `interp_eval_dev`/`interp_eval_sealed`/`interp_eval_k128` suites with `base`/`rl`/
    `interp:0.5` runs -- `view_interpolation`'s own eval suites, which the rest of the demo store
    has no data for at all (mirrors `test_interp_dash.py`'s own interpolation fixture)."""
    dev = [make_problem(f"d{i}", eval_suites.DEV_GEN_SUITE, "v1", length=8) for i in range(3)]
    sealed = [
        make_problem(f"s{i}", eval_suites.SEALED_GEN_SUITE, "v1", length=12) for i in range(3)
    ]
    k128 = [make_problem(f"k{i}", eval_suites.K128_SUITE, "v1", length=10) for i in range(3)]
    store.write_suite(dev, manifest(eval_suites.DEV_GEN_SUITE, "v1", len(dev)))
    store.write_suite(sealed, manifest(eval_suites.SEALED_GEN_SUITE, "v1", len(sealed)))
    store.write_suite(k128, manifest(eval_suites.K128_SUITE, "v1", len(k128)))
    for model, c1, c128 in (("base", 4, 40), ("interp:0.5", 8, 60), ("rl", 12, 90)):
        r1 = start_run(store, suite=eval_suites.DEV_GEN_SUITE, version="v1", model=model, k=32)
        store.append_items(r1, [make_item(r1, f"d{i}", 32, c1) for i in range(3)])
        r2 = start_run(store, suite=eval_suites.SEALED_GEN_SUITE, version="v1", model=model, k=32)
        store.append_items(r2, [make_item(r2, f"s{i}", 32, c1) for i in range(3)])
        r3 = start_run(store, suite=eval_suites.K128_SUITE, version="v1", model=model, k=128)
        store.append_items(r3, [make_item(r3, f"k{i}", 128, c128) for i in range(3)])


def _populate_probes(store: Store) -> None:
    rows = []
    for model, base_acc in (("base", 0.4), ("rl", 0.8)):
        for i, layer in enumerate(("emb", "0", "1", "2", "3")):
            rows.append(
                {
                    "model": model,
                    "layer": layer,
                    "n_train_examples": 100,
                    "n_ood_examples": 20,
                    "next_rule": {
                        "l2": 0.01,
                        "train_acc": 0.9,
                        "val_acc": base_acc + i * 0.02,
                        "test_acc": base_acc + i * 0.02,
                        "n_train": 80,
                        "n_val": 10,
                        "n_test": 10,
                    },
                    "lines_to_go": {
                        "l2": 0.1,
                        "train_r2": 0.9,
                        "val_r2": 0.5,
                        "test_r2": 0.5,
                        "test_bin_acc": 0.6,
                        "n_train": 80,
                        "n_val": 10,
                        "n_test": 10,
                    },
                    "cited_back": None,
                    "mdl_next_rule": {
                        "code_length_bits": 100.0,
                        "uniform_bits": 200.0,
                        "compression": 2.0,
                        "n": 80,
                    },
                    "mdl_lines_to_go_binned": {
                        "code_length_bits": 90.0,
                        "uniform_bits": 150.0,
                        "compression": 1.6,
                        "n": 80,
                    },
                    "length_generalisation": {
                        "empty": False,
                        "l2": 0.1,
                        "in_distribution_val_r2": 0.5,
                        "ood_r2": 0.3,
                        "n_train": 80,
                        "n_ood": 20,
                    },
                }
            )
    run_id = new_interp_run(store, "probes", {})
    write_interp_rows(store, "probes", run_id, rows)


def build_fully_populated_store(store: Store) -> None:
    """`build_full_demo_store` (the pre-existing 8-benchmark + legacy-EDL synthetic store) plus
    everything this page's 4 extra interp charts, and every pooled/"base vs rl" chart, need to
    render real content rather than a single-series or "no data yet" card."""
    build_full_demo_store(store)

    # Chart 2 (families): replaces the demo store's 1-item-per-n families suite with one that
    # clears the new `_MIN_ITEMS_PER_N` threshold (see `_populate_families`'s own docstring).
    _populate_families(store)

    # Charts that only need a second ("rl") run alongside the demo store's existing suites/pids.
    _add_rl_run(
        store, suite="parts_pool", version="v1", pid_ns=[("part_a", 32, 28), ("part_b", 32, 22)]
    )
    # Chart 4 (skills): replaces the demo store's 2-item skill_pool suite (see
    # `_populate_skills`'s own docstring).
    _populate_skills(store)
    _populate_laws(store)
    _add_rl_run(store, suite="textbook_law", version="v1", pid_ns=[("tb1", 32, 4)])
    _add_rl_run(
        store,
        suite="padding",
        version="v1",
        pid_ns=[("pad0", 32, 30), ("pad1", 32, 20), ("pad2", 32, 8)],
    )
    _add_rl_run(
        store,
        suite="gen_ctrl",
        version="v1",
        pid_ns=[("ctrl1", 32, 14), ("nat5", 32, 24), ("nat6", 32, 18), ("nat7", 32, 10)],
    )
    # Chart 7 (prefix): replaces the demo store's 2-item prefix suite (see `_populate_prefix`'s
    # own docstring).
    _populate_prefix(store)

    # Chart 3 (glue): a joined item per operator, not just `AND`.
    _populate_glue_operators(store)

    # Chart 8 (EDL): `edl2-*` run ids specifically (chart 8 ignores plain `edl-*`, including the
    # `edl-base-8-4` row `build_full_demo_store` already wrote).
    edl_problems = _toy_problems(8, 4)
    store.append_edl(
        "edl2-base-8-4",
        run_edl(
            ConstantNllAdapter(c=2.0, label="base"),
            edl_problems,
            lrs=(1e-4,),
            orders=1,
            n=8,
            m=4,
            ckpts=(4, 8),
        ),
    )
    store.append_edl(
        "edl2-rl-8-4",
        run_edl(
            ConstantNllAdapter(c=1.0, label="rl"),
            edl_problems,
            lrs=(1e-4,),
            orders=1,
            n=8,
            m=4,
            ckpts=(4, 8),
        ),
    )

    # Charts 9, 10, 11, 12: the four interp measures (no demo-store equivalent at all).
    _populate_likelihood(store)
    _populate_steering(store)
    _populate_interpolation(store)
    _populate_probes(store)


def test_build_summary_from_a_fully_populated_store(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    build_fully_populated_store(store)

    out_dir = tmp_path / "dash_summary"
    path = dash_summary.build_summary(store, out_dir)
    html = path.read_text()

    assert "<title>ND Capability Benchmarks</title>" in html
    assert _count_chart_cards(html) == _N_CHARTS
    assert "no data yet" not in html, "every one of the 12 charts should have real data"
    assert "http://" not in html
    assert "https://" not in html

    # a spot check that the computed, per-chart takeaway text (not a table) is actually present.
    assert "ahead of" in html or "cross over" in html  # chart 1 (yue)
    assert "EDL" in html  # chart 8


def test_build_summary_is_idempotent_and_overwrites(tmp_path: Path) -> None:
    """Calling `build_summary` twice into the same `out_dir` (e.g. a re-run of the real build)
    just redeploys `summary.html` in place, as any other dashboard page does."""
    store = Store(tmp_path / "store")
    build_fully_populated_store(store)
    out_dir = tmp_path / "dash_summary"
    path1 = dash_summary.build_summary(store, out_dir)
    path2 = dash_summary.build_summary(store, out_dir)
    assert path1 == path2
    assert path2.exists()


# -------------------------------------------------------------------- chart 2: min-items filter


def test_families_chart_filters_thin_cells_and_labels_none_and_censored(tmp_path: Path) -> None:
    """A (family, n) cell with fewer than `_MIN_ITEMS_PER_N` (5) items must be excluded from the
    horizon computation entirely -- not just plotted as unreliable -- and `n_star=None` vs. a
    censored (">=", never dropped below 50% in the tested range) horizon must be labelled, not
    conflated (the pre-fix bug the review caught: `n_star=None` was being read as "infinite")."""
    store = Store(tmp_path / "store")
    # and_tower: 5 items per n (qualifies). thin_family: 1 item per n (never qualifies -> dropped
    # entirely, including from the "families" row count, and named in the caption's drop note).
    probs = [
        make_problem(
            f"and_tower_n{n}_{i}",
            "families",
            "v1",
            length=n + 2,
            tags={"family": "and_tower", "n": n},
        )
        for n in (1, 2, 3)
        for i in range(5)
    ] + [
        make_problem(
            f"thin_n{n}", "families", "v1", length=n + 1, tags={"family": "thin_family", "n": n}
        )
        for n in (1, 2)
    ]
    store.write_suite(probs, manifest("families", "v1", len(probs)))
    for model, cs in (("base", (30, 28, 0)), ("rl", (0, 0, 0)), ("armC", (30, 30, 30))):
        run_id = start_run(store, suite="families", version="v1", model=model, k=32)
        items = [
            make_item(run_id, f"and_tower_n{n}_{i}", 32, c)
            for n, c in zip((1, 2, 3), cs, strict=True)
            for i in range(5)
        ]
        items += [make_item(run_id, f"thin_n{n}", 32, 32) for n in (1, 2)]
        store.append_items(run_id, items)

    html = dash_summary._chart_2_families(store)  # noqa: SLF001
    data = dash_summary._data_2_families(store)  # noqa: SLF001
    # rl fails at every *qualifying* size (n=1, the smallest with >=5 items) -> "fails at smallest
    # size" (value None, drawn as a hollow marker at 0 and explained in the caption), not
    # "infinite"/omitted.
    assert ("rl", None) in [(n, v) for r in data.rows for n, v, _c, _z in r.points]
    assert data.opts["none_label"] == "fails at smallest size"
    assert "fails at the smallest unseen size" in html
    # armC clears every qualifying size tested (n=1..3) -> a censored ">=" horizon, not a plain one
    # (the figure draws it as an arrow; the data says so directly).
    assert [(n, z) for r in data.rows for n, _v, _c, z in r.points if z] == [("armC", True)]
    assert "&gt;= 5" in html  # armC's n*=3 -> written lines n+2=5, html-escaped in the table
    # thin_family never has a qualifying cell (1 item per n < 5) -> dropped, named in the caption.
    assert "thin_family" in html
    assert "and_tower" in html


# -------------------------------------------------------------------- chart 8: null control


def _edl_row(cls: str, label: str, lr: float, edl: float, n: int = 8) -> dict[str, Any]:
    """A hand-built row in `edl.run_edl`'s real schema (mirrors `test_interp_dash.py`'s
    `_likelihood_row` pattern): enough fields for `analysis.edl.view` to pick a `best_lr` (by
    smallest mean `online_nats` sum) and report a final `EDL`, without needing a real online
    training run."""
    online = [edl / n] * n
    return {
        "class": cls,
        "model_identity": {"label": label},
        "lr": lr,
        "order": 0,
        "n": n,
        "m": n,
        "seed": 0,
        "online_nats": online,
        "L_test_0": online[0],
        "L_test_n": online[-1],
        "L_test_by_checkpoint": {},
        "EDL_by_checkpoint": {},
        "EDL": edl,
        "EDL_per_example": edl / n,
        "start_excluded_online_nats": None,
        "L_test_0_start_excluded": None,
        "L_test_n_start_excluded": None,
        "EDL_start_excluded": None,
        "pass8_before": None,
        "pass8_after": None,
    }


def test_edl_chart_excludes_null_control_from_the_lower_count(tmp_path: Path) -> None:
    """`heldout_cap6` (the null control, EDL ~= 0) must be reported (a last row, labelled "null
    control") but excluded from the "rl lower on N/M classes" denominator -- a negative control
    class with nothing to learn says nothing about whether rl "elicits capability faster"."""
    store = Store(tmp_path / "store")
    store.append_edl(
        "edl2-base-8-8",
        [
            _edl_row("pattern_a", "base", 1e-4, 10.0),
            _edl_row("pattern_b", "base", 1e-4, 5.0),
            _edl_row("heldout_cap6", "base", 1e-4, -0.1),  # slightly negative, as real noise does
        ],
    )
    store.append_edl(
        "edl2-rl-8-8",
        [
            _edl_row("pattern_a", "rl", 1e-4, 2.0),  # rl lower here
            _edl_row("pattern_b", "rl", 1e-4, 8.0),  # rl higher here
            _edl_row("heldout_cap6", "rl", 1e-4, 0.3),
        ],
    )
    html = dash_summary._chart_8_edl(store)  # noqa: SLF001
    assert "lower on 1/2 classes" in html  # only pattern_a, out of the 2 non-null classes
    assert "heldout_cap6 (null control)" in html
    assert "&lt;= 0" in html  # base's negative value on the null control, html-escaped in the table


# -------------------------------------------------------------------- chart 10: steering bars


def test_steering_chart_shows_four_comparison_bars(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    probs = [make_problem(f"dev{i}", "textbook_dev", "v1", length=8) for i in range(4)]
    store.write_suite(probs, manifest("textbook_dev", "v1", len(probs)))
    for model, c in (
        ("base", 1),
        ("steer:0:0.5", 2),
        ("steer:0:1.0", 3),
        ("steer:0:-1", 5),
        ("rl", 10),
    ):
        run_id = start_run(store, suite="textbook_dev", version="v1", model=model, k=32)
        store.append_items(run_id, [make_item(run_id, f"dev{i}", 32, c) for i in range(4)])

    html = dash_summary._chart_10_steering(store)  # noqa: SLF001
    assert "base (unsteered)" in html
    assert "base + delta" in html
    assert "rl - delta" in html
    assert "the best of 2 (layer, alpha) settings" in html  # 2 base+delta configs here
    assert "costs" in html and "points" in html


# --------------------------------------------------------------------------- helper unit tests


def test_edl_friendly_label_handles_the_edl2_prefix() -> None:
    """The real store's `edl2-*` run ids are exactly what `analysis/edl.py`'s own
    `_label_from_run_id` fails to parse (it only recognises a literal `edl-` prefix, module
    docstring) -- `_edl_friendly_label` must still resolve `"base"`/`"rl"` from the run id even
    when `identity` carries no `"label"` key (the production `AbsGptAdapter`'s own identity shape).
    """
    assert dash_summary._edl_friendly_label({}, "edl2-base-64-64") == "base"  # noqa: SLF001
    assert dash_summary._edl_friendly_label({}, "edl2-rl-64-64") == "rl"  # noqa: SLF001
    # identity's own "label" still wins over any run-id parsing.
    assert (
        dash_summary._edl_friendly_label({"label": "armC"}, "edl2-base-64-64")  # noqa: SLF001
        == "armC"
    )
    # no "label" and no parseable run id: falls back to the checkpoint path.
    assert (
        dash_summary._edl_friendly_label(  # noqa: SLF001
            {"ckpt": "/x/y/pretrain.pt"}, "not-a-recognised-run-id"
        )
        == "/x/y/pretrain.pt"
    )


def test_pooled_glue_operator_colours_are_fixed_per_model() -> None:
    assert dash_summary.MODEL_COLOR["base"] != dash_summary.MODEL_COLOR["rl"]
    assert dash_summary._mcolor("base") == dash_summary.MODEL_COLOR["base"]  # noqa: SLF001
    assert dash_summary._mcolor("rl") == dash_summary.MODEL_COLOR["rl"]  # noqa: SLF001


def test_steering_pools_textbook_dev_by_default() -> None:
    """`_populate_steering` (above) puts its `steer:*` runs on `textbook_dev` specifically because
    it is one of `view_steering`'s default pooled suites; pin that assumption directly so a future
    change to that default list fails here, not as a silently-empty chart 10."""
    assert "textbook_dev" in interp_view.DEFAULT_STEERING_SUITES


def test_summary_footer_links_detail_pages_only_when_asked(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    plain = dash_summary.build_summary(store, tmp_path / "a").read_text()
    linked = dash_summary.build_summary(store, tmp_path / "b", detail_href="detail/index.html")
    assert 'href="detail/index.html"' not in plain
    assert 'href="detail/index.html"' in linked.read_text()
