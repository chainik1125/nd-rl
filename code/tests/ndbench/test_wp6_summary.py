"""Acceptance tests for `dash/summary.py` (the single-page "one chart per benchmark" dashboard):

- builds from a synthetic store with every one of the 12 charts populated;
- builds from an empty store, where every chart shows a "no data yet" panel;
- exactly 12 chart cards either way;
- no external URLs.
"""

from __future__ import annotations

import re
from pathlib import Path

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

    # Charts that only need a second ("rl") run alongside the demo store's existing suites/pids.
    _add_rl_run(
        store,
        suite="families",
        version="v1",
        # all-zero rl: families' n* should come back `None` for rl (exercises chart 2's "none"
        # dot-plot branch) while base keeps its own (non-`None`) horizon from the demo store.
        pid_ns=[("and_tower_n1", 32, 0), ("and_tower_n2", 32, 0), ("and_tower_n3", 32, 0)],
    )
    _add_rl_run(
        store, suite="parts_pool", version="v1", pid_ns=[("part_a", 32, 28), ("part_b", 32, 22)]
    )
    _add_rl_run(store, suite="skill_pool", version="v1", pid_ns=[("i1", 32, 24), ("i2", 32, 2)])
    _add_rl_run(
        store,
        suite="laws",
        version="v1",
        pid_ns=[("law_s0", 32, 28), ("law_s2", 32, 14), ("law_s4", 32, 2)],
    )
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
    _add_rl_run(store, suite="prefix", version="v1", pid_ns=[("pfx_a", 10, 8), ("pfx_b", 10, 7)])

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
