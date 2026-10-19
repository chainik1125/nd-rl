"""Regression tests for the 2026-10-01 audit of the "State-env models" (Dan) dashboard section.

Each test pins one confirmed problem: the "ours" panels must reproduce the audited paper figures
(families, laws, glue, padding, prefix), each Dan panel must use the same estimator with Dan's
own training tiers and judge, fz and t1 must be told apart, the judge-agreement view must pool
every run of a model, Table D1 must show Lean for ours too, and Dan's runs (nd_v2 rejudges,
state-env probes runs) must not leak into the main base/rl summary page. CPU only, synthetic
stores, a few seconds in all.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from code.ndbench.analysis import judges as judges_view
from code.ndbench.analysis import models_registry as mr
from code.ndbench.analysis import subst as subst_view
from code.ndbench.dash import dan_pages, summary
from code.ndbench.dash.chart_data import ChartData
from code.ndbench.figures import plots
from code.ndbench.figures import style as S
from code.ndbench.interp._store import new_interp_run, write_interp_rows
from code.ndbench.logic.exclusion import seen_sidecar_path
from code.ndbench.store import Store

from .test_wp6_helpers import make_item, make_problem, manifest, start_run
from .test_wp6_summary import build_fully_populated_store

REGISTRY = mr.load_registry()


def _sidecar(store: Store, suite: str, version: str, family: str, seen: dict[str, bool]) -> None:
    path = seen_sidecar_path(store.root, suite, version, family)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"suite": suite, "version": version, "seen": seen}))


@pytest.fixture(scope="module")
def synth(tmp_path_factory: pytest.TempPathFactory) -> Store:
    store = Store(tmp_path_factory.mktemp("dan_audit") / "store")
    build_fully_populated_store(store)
    return store


# ======================================================== "ours" panels = the paper figures


def _ours(cd: ChartData) -> ChartData:
    return cd.extra["panels"]["ours"]


def test_ours_families_panel_is_figure_3(synth: Store) -> None:
    fig3 = summary._data_2_families(synth)  # noqa: SLF001
    d2 = _ours(dan_pages._data_dan_families(synth, REGISTRY))  # noqa: SLF001
    want = {
        r.label: [(m, v, z) for m, v, _c, z in r.points if m in ("base", "rl")] for r in fig3.rows
    }
    got = {r.label: [(m, v, z) for m, v, _c, z in r.points] for r in d2.rows}
    assert got == want
    # written lines (item length at n*), not n*: base clears n=2, whose items are 4 lines long
    assert got["and_tower"][0] == ("base", 4.0, False)
    assert got["and_tower"][1] == ("rl", None, False)


def test_ours_laws_panel_is_figure_5(synth: Store) -> None:
    fig5 = {s.name: s.pts for s in summary._data_5_subst(synth).series}  # noqa: SLF001
    d3 = {s.name: s.pts for s in _ours(dan_pages._data_dan_laws(synth, REGISTRY)).series}  # noqa: SLF001
    assert d3 == fig5


def test_ours_glue_panel_is_figure_4(synth: Store) -> None:
    fig4 = [(b.name, b.value, b.lo, b.hi) for b in summary._data_3_glue(synth).bars]  # noqa: SLF001
    d4_panel = _ours(dan_pages._data_dan_glue(synth, REGISTRY))  # noqa: SLF001
    d4 = [(b.name, b.value, b.lo, b.hi) for b in d4_panel.bars]
    assert sorted(d4, key=str) == sorted(fig4, key=str)
    assert d4_panel.opts["y_fmt"] == "f3"  # R is a ratio, not drawn on a percent axis


def test_ours_padding_panel_is_figure_6(synth: Store) -> None:
    fig6 = {s.name: s.pts for s in summary._data_6_padding(synth).series}  # noqa: SLF001
    d5 = {s.name: s.pts for s in _ours(dan_pages._data_dan_padding(synth, REGISTRY)).series}  # noqa: SLF001
    assert d5 == {"base": fig6["base padded"], "rl": fig6["rl padded"]}


def test_ours_prefix_panel_is_figure_7(synth: Store) -> None:
    fig7 = {s.name: s.pts for s in summary._data_7_prefix(synth).series}  # noqa: SLF001
    d6 = dan_pages._data_dan_prefix_state(synth, REGISTRY)  # noqa: SLF001
    assert {s.name: s.pts for s in _ours(d6).series} == fig7


# ============================================================ Dan panels: estimator + tiers


def test_dan_families_uses_written_lines_dans_tiers_and_min_items(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    probs = [
        replace(
            make_problem(
                f"f_n{n}_{i}", "families", "v1", length=2 * n + 1, tags={"family": "fam", "n": n}
            ),
            seen_train=(n == 2),  # our tiers: must not apply to Dan's model
        )
        for n in (1, 2, 3)
        for i in range(5)
    ] + [
        make_problem(f"f_thin_{i}", "families", "v1", length=9, tags={"family": "fam", "n": 4})
        for i in range(2)
    ]
    store.write_suite(probs, manifest("families", "v1", len(probs)))
    run_id = start_run(
        store, suite="families", version="v1", model="dan12-t1-s0", k=16, judge_id="lean_v2"
    )
    cs = {1: 16, 2: 16, 3: 0}
    store.append_items(
        run_id,
        [make_item(run_id, f"f_n{n}_{i}", 16, cs[n]) for n in (1, 2, 3) for i in range(5)]
        + [make_item(run_id, f"f_thin_{i}", 16, 16) for i in range(2)],
    )
    # Dan's own tiers mark n=1 seen: his unseen horizon starts at n=2 and stops there.
    _sidecar(store, "families", "v1", "dan12", {p.pid: p.tags["n"] == 1 for p in probs})
    cd = dan_pages._data_dan_families(store, REGISTRY)  # noqa: SLF001
    rows = cd.extra["panels"]["dan12"].rows
    assert [(r.label, r.points) for r in rows] == [
        ("fam", [("dan12-t1-s0", 5.0, "var(--s1)", False)])  # n*=2 -> 5 written lines
    ]
    assert cd.extra["panels"]["ours"].rows == []  # no "ours" runs, no zero bars
    assert cd.opts["legend"] == ["dan12-t1-s0"]  # models without a run are not drawn


def test_dan_laws_pools_per_item_with_each_models_own_tiers(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    probs = [
        replace(
            make_problem(f"l_{sch}_s{s}_{i}", "laws", "v1", length=7, tags={"schema": sch, "s": s}),
            seen_train=(s == 2 and i == 0),
        )
        for sch, n_items in (("a", 30), ("b", 10))
        for s in (0, 2)
        for i in range(n_items if s == 2 else 1)
    ]
    store.write_suite(probs, manifest("laws", "v1", len(probs)))
    for model, judge in (("rl", "nd_v2"), ("dan12-t1-s0", "lean_v2")):
        run_id = start_run(store, suite="laws", version="v1", model=model, k=4, judge_id=judge)
        store.append_items(
            run_id,
            [make_item(run_id, p.pid, 4, 4 if p.tags["schema"] == "a" else 0) for p in probs],
        )
    # Dan's tiers: nothing of ours, but every schema-b s=2 item is seen.
    _sidecar(
        store,
        "laws",
        "v1",
        "dan12",
        {p.pid: p.tags["schema"] == "b" and p.tags["s"] == 2 for p in probs},
    )
    cd = dan_pages._data_dan_laws(store, REGISTRY)  # noqa: SLF001
    ours = {s.name: s.pts for s in cd.extra["panels"]["ours"].series}
    dan = {s.name: s.pts for s in cd.extra["panels"]["dan12"].series}
    # s=0 (2 items) is dropped; s=2 pooled per item: rl 29 of a (solved) + 9 of b (not) = 29/38
    assert ours == {"rl": [(2.0, pytest.approx(29 / 38))]}
    # Dan keeps a's seen-by-us item and drops all of b: 30/30
    assert dan == {"dan12-t1-s0": [(2.0, pytest.approx(1.0))]}
    data = subst_view.view(store, {"models": ["dan12-t1-s0"], "judge_id": "lean_v2"})
    assert data["schemas"]["a"]["dan12-t1-s0"]["n_seen_excluded"] == 10


def test_dan_padding_counts_each_unpadded_theorem_once(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    probs = [
        make_problem(
            f"pad_{t}_m0", "padding", "v1", length=3, tags={"pad_type": t, "m": 0, "base_pid": "b0"}
        )
        for t in ("P1-shared", "P1-disjoint", "P2")
    ] + [
        make_problem(
            "pad_other_m0",
            "padding",
            "v1",
            length=3,
            tags={"pad_type": "P2", "m": 0, "base_pid": "b1"},
        )
    ]
    store.write_suite(probs, manifest("padding", "v1", len(probs)))
    run_id = start_run(
        store, suite="padding", version="v1", model="dan12-t1-s0", k=4, judge_id="lean_v2"
    )
    store.append_items(
        run_id,
        [make_item(run_id, f"pad_{t}_m0", 4, 4) for t in ("P1-shared", "P1-disjoint", "P2")]
        + [make_item(run_id, "pad_other_m0", 4, 0)],
    )
    cd = dan_pages._data_dan_padding(store, REGISTRY)  # noqa: SLF001
    (series,) = cd.extra["panels"]["dan12"].series
    assert series.pts == [(3.0, pytest.approx(0.5))]  # b0 once (1.0), b1 (0.0); not 3:1


def test_dan_prefix_drops_bins_with_fewer_than_20_items(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    probs = [
        make_problem(
            f"s_{j}_{i}", "prefix_state", "v1", tags={"j": j, "L": 5, "actions_left": 6 - j}
        )
        for j, n_items in ((4, 20), (2, 5))
        for i in range(n_items)
    ]
    store.write_suite(probs, manifest("prefix_state", "v1", len(probs)))
    run_id = start_run(
        store, suite="prefix_state", version="v1", model="dan12-t1-s0", k=4, judge_id="lean_v2"
    )
    store.append_items(run_id, [make_item(run_id, p.pid, 4, 2) for p in probs])
    cd = dan_pages._data_dan_prefix_state(store, REGISTRY)  # noqa: SLF001
    (series,) = cd.extra["panels"]["dan12"].series
    assert series.pts == [(1.0, 0.5)]  # lines left 3 (5 items) omitted


def test_dan_glue_pools_case_and_needs_no_parts_pool_run(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    joined = [
        make_problem(f"j_{op}", "glue", "v1", length=8, tags={"op": op}, parts=("pa", "pb"))
        for op in ("CASE-dilemma", "CASE-same")
    ]
    parts = [
        make_problem(f"j_{op}__p{i}", "glue_parts_renamed", "v1", length=3)
        for op in ("CASE-dilemma", "CASE-same")
        for i in range(2)
    ]
    store.write_suite(joined, manifest("glue", "v1", len(joined)))
    store.write_suite(parts, manifest("glue_parts_renamed", "v1", len(parts)))
    run_id = start_run(
        store, suite="glue", version="v1", model="dan12-t1-s0", k=4, judge_id="lean_v2"
    )
    store.append_items(
        run_id, [make_item(run_id, "j_CASE-dilemma", 4, 4), make_item(run_id, "j_CASE-same", 4, 0)]
    )
    rid = start_run(
        store,
        suite="glue_parts_renamed",
        version="v1",
        model="dan12-t1-s0",
        k=4,
        judge_id="lean_v2",
    )
    store.append_items(rid, [make_item(rid, p.pid, 4, 4) for p in parts])
    cd = dan_pages._data_dan_glue(store, REGISTRY)  # noqa: SLF001
    bars = [(b.name, b.value) for b in cd.extra["panels"]["dan12"].bars]
    assert bars == [("CASE: dan12-t1-s0", pytest.approx(0.5))]  # (1 + 0) / (1 + 1)


# ====================================================================== styles and figures


def test_fz_and_t1_are_drawn_differently() -> None:
    fz, t1 = S.model_style("dan12-fz-s0"), S.model_style("dan12-t1-s0")
    assert fz.color != t1.color  # pretrained darker shade, RL full hue
    assert fz.color == S.shade(t1.color, S.ROLE_SHADE["base"])
    assert fz.hatch is None and t1.hatch  # bars: solid vs hatched, readable in greyscale
    assert fz.marker == t1.marker  # one marker per family
    assert S.model_style("dan12-fz-s1").linestyle != fz.linestyle  # seeds: line style


def test_dan_panels_share_one_y_scale() -> None:
    def panel(name: str, top: float) -> ChartData:
        from code.ndbench.dash import primitives as P

        return ChartData(
            "dan_laws", "t", "line", series=[P.Series(name, [(1.0, top), (2.0, 0.0)], "")]
        )

    cd = ChartData(
        "dan_laws",
        "t",
        "line",
        opts={"legend": ["rl", "dan12-t1-s0"]},
        extra={"panels": {"ours": panel("rl", 0.1), "dan12": panel("dan12-t1-s0", 0.8)}},
    )
    with S.paper_style():
        fig = plots.render(cd)
        assert fig is not None
        tops = {ax.get_ylim()[1] for ax in fig.axes}
        S.close(fig)
    assert len(tops) == 1 and tops.pop() >= 0.8


# ========================================================================= judge agreement


def _sample(store: Store, run_id: str, i: int, ok: bool, reason: str = "") -> None:
    store.append_samples(
        run_id,
        [
            {
                "pid": "t0",
                "chunk": 0,
                "i": i,
                "body": "",
                "native": None,
                "n_tokens": 1,
                "finished": True,
                "ok": ok,
                "reason": reason,
                "kind": "ok" if ok else "bad_line",
                "written": 1,
                "pruned": None,
                "first_bad_line": None,
            }
        ],
    )


def test_judges_view_pools_every_run_of_a_model(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    for suite, lean_only in (("s_a", 1), ("s_b", 0)):
        lean = start_run(
            store, suite=suite, version="v1", model="dan12-t1-s0", k=2, judge_id="lean_v2"
        )
        nd = store.new_derived_run(derived_from=lean, judge_id="nd_v2")
        store.finish_run(nd, "complete", 0.0)
        _sample(store, lean, 0, True)
        _sample(store, nd, 0, not lean_only, "premise")
        _sample(store, lean, 1, False)
        _sample(store, nd, 1, False)
    info = judges_view.view(store, {"models": ["dan12-t1-s0"]})["models"]["dan12-t1-s0"]
    assert info["n_pairs"] == 2 and info["n_attempts"] == 4
    assert info["lean_only_count"] == 1 and info["agree_count"] == 3
    assert info["lean_only_rate"] == pytest.approx(0.25)
    cd = dan_pages._data_dan_judges(store, REGISTRY)  # noqa: SLF001
    assert [(b.name, b.value) for b in cd.bars] == [("dan12-t1-s0", pytest.approx(0.25))]
    assert "4 attempts" in cd.caption and "1 lean-only" in cd.caption


# ================================================================================ Table D1


def test_crossfamily_table_shows_lean_for_ours_too(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    probs = [make_problem("t0", "textbook_dev", "v1", length=2)]
    store.write_suite(probs, manifest("textbook_dev", "v1", 1))
    for model, judge, c in (("base", "nd_v2", 64), ("base", "lean_v2", 128)):
        run_id = start_run(
            store, suite="textbook_dev", version="v1", model=model, k=256, judge_id=judge
        )
        store.append_items(run_id, [make_item(run_id, "t0", 256, c)])
    result = dan_pages.dan_crossfamily_table(store, REGISTRY)
    assert result is not None
    headers, rows, _caption = result
    assert headers[2] == "pass@1, lean_v2 (nd_v2)"
    base = next(r for r in rows if r[:2] == ["ours", "base"])
    assert base[2] == "50.0% (25.0%)"


# ================================================= main page: Dan's runs must not leak in


def test_main_probes_chart_ignores_newer_dan_probes_runs(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    build_fully_populated_store(store)
    before = summary._data_12_probes(store)  # noqa: SLF001
    run_id = new_interp_run(store, "probes", {"base_model": "dan12-fz-s0"})
    write_interp_rows(
        store,
        "probes",
        run_id,
        [{"model": m, "layer": 0, "next_rule": {"test_acc": 0.5}} for m in ("fz", "t1")],
    )
    after = summary._data_12_probes(store)  # noqa: SLF001
    assert after.kind != "missing"
    assert [(s.name, s.pts) for s in after.series] == [(s.name, s.pts) for s in before.series]


def test_main_laws_caption_ignores_dan_nd_v2_rejudges(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    build_fully_populated_store(store)
    before = summary._data_5_subst(store)  # noqa: SLF001
    run_id = start_run(store, suite="laws", version="v1", model="dan12-fz-s0", k=4)  # nd_v2
    store.append_items(
        run_id, [make_item(run_id, p.pid, 4, 4) for p in store.load_suite("laws", "v1")]
    )
    after = summary._data_5_subst(store)  # noqa: SLF001
    assert after.caption == before.caption
    assert [(s.name, s.pts) for s in after.series] == [(s.name, s.pts) for s in before.series]


def test_crossfamily_tex_escapes_headers() -> None:
    tex = dan_pages._crossfamily_tex(["pass@1, lean_v2 (nd_v2)"], [["4.1%"]], "lean_v2")  # noqa: SLF001
    assert r"pass@1, lean\_v2 (nd\_v2) \\" in tex
    assert "lean_v2" not in tex.replace(r"lean\_v2", "")
