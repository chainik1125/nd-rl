"""Tests for the Dan-only results page (`dash --summary --dan-only`, `dash/dan_only.py`,
`figures/dan_only.py`, `analysis/dan_views.py`).

On a synthetic store holding both our base/rl/armC data (`build_fully_populated_store`) and
Dan-model runs for every benchmark: the page renders every section; nothing of base/rl/armC
appears in any chart, legend, tick label, table or caption; missing data gives placeholders (KL
"not yet run"); takeaways are computed from the data; the KL card renders once a `kl` run exists;
and the main summary / `--with-dan` pages are unaffected. CPU only, synthetic stores.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pytest
from matplotlib.text import Text

from code.ndbench import cli
from code.ndbench.analysis import dan_views as dv
from code.ndbench.analysis import models_registry as mr
from code.ndbench.core.io import write_jsonl
from code.ndbench.core.types import JSON
from code.ndbench.dash import dan_only, dan_pages, summary
from code.ndbench.dash.chart_data import ChartData
from code.ndbench.figures import dan_only as F
from code.ndbench.figures import style as S
from code.ndbench.interp._store import new_interp_run, write_interp_rows
from code.ndbench.logic.exclusion import seen_sidecar_path
from code.ndbench.store import Store

from .test_wp6_helpers import make_item, make_problem, manifest, start_run
from .test_wp6_summary import build_fully_populated_store

REGISTRY = mr.load_registry()
FZ, T1 = "dan12-fz-s0", "dan12-t1-s0"
LEAN = "lean_v2"
OURS = re.compile(r"\b(base|rl|armC)\b")


def _run(
    store: Store,
    suite: str,
    model: str,
    k: int,
    cs: dict[str, int],
    version: str = "v1",
    *,
    rejudge: bool = False,
) -> str:
    run_id = start_run(store, suite=suite, version=version, model=model, k=k, judge_id=LEAN)
    store.append_items(run_id, [make_item(run_id, pid, k, c) for pid, c in cs.items()])
    if rejudge:
        nd = store.new_derived_run(derived_from=run_id, judge_id="nd_v2")
        store.append_items(nd, [make_item(nd, pid, k, c) for pid, c in cs.items()])
        store.finish_run(nd, "complete", 0.0)
    return run_id


def _sample(store: Store, run_id: str, pid: str, i: int, ok: bool, reason: str = "") -> None:
    store.append_samples(
        run_id,
        [
            {
                "pid": pid,
                "chunk": 0,
                "i": i,
                "body": "N1 P PR ; QED",
                "native": None,
                "n_tokens": 3,
                "finished": True,
                "ok": ok,
                "reason": reason,
                "kind": "ok" if ok else "bad_line",
                "written": 1,
                "pruned": 1 if ok else None,
                "first_bad_line": None,
            }
        ],
    )


def _sidecar(store: Store, suite: str, version: str, seen: dict[str, bool]) -> None:
    path = seen_sidecar_path(store.root, suite, version, "dan12")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"suite": suite, "version": version, "seen": seen}))


#: Synthetic textbook72 counts per (cap, role): seed s solves the first n of dev and train14.
TEXTBOOK = {
    ("dan12", "fz"): ((1, 1, 2), (1, 0, 1)),
    ("dan12", "t1"): ((3, 3, 4), (1, 1, 2)),
    ("dan6", "fz"): ((1, 1, 1), (0, 0, 0)),
    ("dan6", "t1"): ((2, 3, 2), (1, 1, 1)),
}


def populate_dan(store: Store) -> None:
    """Dan-model runs (judged lean_v2) for every benchmark of the Dan-only page, on top of a
    store that already holds our own base/rl/armC data."""
    # textbook_dev (dev0..dev3 from the demo store) + textbook_train14 for all twelve models
    tr = [make_problem(f"tr{i}", "textbook_train14", "v1", length=4) for i in range(2)]
    store.write_suite(tr, manifest("textbook_train14", "v1", len(tr)))
    for (fam, role), (dev_n, tr_n) in TEXTBOOK.items():
        for seed in range(3):
            m = f"{fam}-{role}-s{seed}"
            run = _run(
                store,
                "textbook_dev",
                m,
                8,
                {f"dev{i}": (4 if i < dev_n[seed] else 0) for i in range(4)},
                rejudge=True,
            )
            _run(
                store,
                "textbook_train14",
                m,
                8,
                {f"tr{i}": (2 if i < tr_n[seed] else 0) for i in range(2)},
                rejudge=True,
            )
            if m == T1:  # judge-agreement samples: one Lean-only accept with a premise error
                nd = f"{run}~nd_v2"
                _sample(store, run, "dev0", 0, True)
                _sample(store, nd, "dev0", 0, True)
                _sample(store, run, "dev1", 0, True)
                _sample(store, nd, "dev1", 0, False, "premise not restated")
                _sample(store, run, "dev2", 0, False)
                _sample(store, nd, "dev2", 0, False, "bad")
    # textbook_law (own version so ours' v1 stays as is): 4 theorems, one seen by Dan's tiers
    tl = [make_problem(f"tl{i}", "textbook_law", "dan_v1", length=5) for i in range(4)]
    store.write_suite(tl, manifest("textbook_law", "dan_v1", len(tl)))
    _sidecar(store, "textbook_law", "dan_v1", {f"tl{i}": i == 0 for i in range(4)})
    _run(store, "textbook_law", FZ, 8, {"tl0": 8, "tl1": 0, "tl2": 1, "tl3": 0}, "dan_v1")
    _run(store, "textbook_law", T1, 8, {"tl0": 8, "tl1": 6, "tl2": 8, "tl3": 2}, "dan_v1")
    # held-out
    held = [make_problem(f"h{i}", "dan_heldout", "v1", length=3) for i in range(10)]
    store.write_suite(held, manifest("dan_heldout", "v1", len(held)))
    _run(store, "dan_heldout", FZ, 1, {f"h{i}": int(i < 8) for i in range(10)})
    _run(store, "dan_heldout", T1, 1, {f"h{i}": 1 for i in range(10)})
    # families (and_tower n = 1..3, 5 items each): fz clears n = 1, t1 every n (censored)
    for m, cs in ((FZ, (16, 0, 0)), (T1, (16, 16, 16))):
        _run(
            store,
            "families",
            m,
            16,
            {f"and_tower_n{n}_{i}": cs[n - 1] for n in (1, 2, 3) for i in range(5)},
        )
    # laws (s = 0, 2, 4, 20 items each)
    for m, cs in ((FZ, (8, 2, 0)), (T1, (14, 8, 1))):
        _run(
            store,
            "laws",
            m,
            16,
            {f"law_s{s}_{i}": c for s, c in zip((0, 2, 4), cs, strict=True) for i in range(20)},
        )
    # glue: joined1..4 (demo store) + their renamed parts, all mastered
    parts = [
        make_problem(f"joined{j}__p{i}", "glue_parts_renamed", "v1", length=3)
        for j in range(1, 5)
        for i in range(2)
    ]
    store.write_suite(parts, manifest("glue_parts_renamed", "v1", len(parts)))
    for m, cs in ((FZ, (4, 8, 1, 10)), (T1, (8, 12, 4, 15))):
        _run(store, "glue", m, 16, {f"joined{j}": c for j, c in zip(range(1, 5), cs, strict=True)})
        _run(store, "glue_parts_renamed", m, 16, {p.pid: 16 for p in parts})
    # padding v2: 2 bases x 3 pad types x m = 0..2
    pads = [
        make_problem(
            f"pad-{t}-m{mm}-b{b}",
            "padding",
            "v2",
            length=3 + mm,
            tags={"pad_type": t, "m": mm, "base_pid": f"b{b}"},
        )
        for t in dv.PAD_TYPES
        for mm in range(3)
        for b in range(2)
    ]
    store.write_suite(pads, manifest("padding", "v2", len(pads)))
    p_rate = {
        T1: {
            "P2": (1.0, 0.75, 0.5),
            "P1-shared": (1.0, 0.5, 0.25),
            "P1-disjoint": (1.0, 0.25, 0.0),
        },
        FZ: {
            "P2": (0.75, 0.5, 0.25),
            "P1-shared": (0.75, 0.25, 0.0),
            "P1-disjoint": (0.75, 0.0, 0.0),
        },
    }
    for m, rates in p_rate.items():
        _run(
            store,
            "padding",
            m,
            4,
            {p.pid: int(4 * rates[str(p.tags["pad_type"])][int(str(p.tags["m"]))]) for p in pads},
            "v2",
        )
    # prefix_state: 20 items at 1 and at 3 lines left
    pre = [
        make_problem(
            f"ps_{ll}_{i}", "prefix_state", "v1", tags={"j": 5 - ll, "L": 5, "actions_left": ll}
        )
        for ll in (1, 3)
        for i in range(20)
    ]
    store.write_suite(pre, manifest("prefix_state", "v1", len(pre)))
    for m, cs in ((FZ, (8, 2)), (T1, (14, 9))):
        _run(store, "prefix_state", m, 16, {p.pid: cs[0] if "_1_" in p.pid else cs[1] for p in pre})
    # skill_pool_sub: 20 items per (group, L) at L = 7, 8
    sk = [
        make_problem(f"sk_{g}_{L}_{i}", "skill_pool_sub", "v1", length=L, tags={"skill_group": g})
        for g in ("plain", "case_split", "classical")
        for L in (7, 8)
        for i in range(20)
    ]
    store.write_suite(sk, manifest("skill_pool_sub", "v1", len(sk)))
    rate = {
        FZ: {"plain": 4, "case_split": 3, "classical": 1},
        T1: {"plain": 8, "case_split": 10, "classical": 4},
    }
    for m, r in rate.items():
        _run(store, "skill_pool_sub", m, 16, {p.pid: r[str(p.tags["skill_group"])] for p in sk})
    # steering on interp_eval_dev/sealed (d0-2, s0-2), k = 8
    steer = {
        FZ: 3,
        f"steer:{FZ}:0:0.5": 3,
        f"steer:{FZ}:1:0.5": 4,
        f"steer:{FZ}:1:2": 1,
        f"steer:{T1}:1:-1": 6,
    }
    for m, c in steer.items():
        _run(store, "interp_eval_dev", m, 8, {f"d{i}": c for i in range(3)})
        _run(store, "interp_eval_sealed", m, 8, {f"s{i}": max(c - 1, 0) for i in range(3)})
    # interpolation on the k = 128 subset (k0-2)
    for m, c in ((FZ, 20), ("dan12-interp:0.5", 60), (T1, 90), ("dan12-interp:1.25", 80)):
        _run(store, "interp_eval_k128", m, 128, {f"k{i}": c for i in range(3)})
    # probes (fz/t1 rows) + its position-only baselines in the run's printed output
    cfg = {"base_model": FZ, "rl_model": T1, "n_train": 30, "l_min": 6, "l_max": 10}
    probe_rows: list[dict[str, JSON]] = [
        {
            "model": role,
            "layer": layer,
            "n_train_examples": 200,
            "next_rule": {
                "test_acc": 0.6 + 0.04 * layer + (0.01 if role == "t1" else 0.0),
                "n_test": 40,
            },
            "actions_to_go": {"test_r2": 0.7 + 0.01 * layer, "n_test": 40},
        }
        for role in ("fz", "t1")
        for layer in range(6)
    ]
    run_id = new_interp_run(store, "probes", cfg)
    write_interp_rows(store, "probes", run_id, probe_rows)
    logs = Path(store.root) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "interp_probes_dan12.log").write_text(
        "# header line\n"
        + json.dumps(
            {
                "run_id": run_id,
                "baselines": {
                    "next_rule_acc_majority": 0.3,
                    "next_rule_acc_step_index_premises": 0.62,
                    "next_rule_acc_step_index_premises_depth": 0.7,
                    "actions_to_go_r2_step_index_premises": 0.75,
                    "actions_to_go_r2_step_index_premises_depth": 0.8,
                },
            },
            indent=2,
        )
        + "\n"
    )
    # likelihood (fz/t1 rows)
    lik: list[dict[str, JSON]] = [
        {
            "pid": f"r{i}",
            "origin": origin,
            "L": 6,
            "lines": 6,
            "fz": {"logp_marginal": -6.0 - i, "n_orderings": 1},
            "t1": {"logp_marginal": -6.0 - i + delta, "n_orderings": 1},
        }
        for origin, delta in (("reserve", -3.6), ("family", 7.2))
        for i in range(5)
    ]
    run_id = new_interp_run(
        store, "likelihood", {"base_model": FZ, "rl_model": T1, "l_min": 6, "l_max": 10}
    )
    write_interp_rows(store, "likelihood", run_id, lik)


def add_kl_run(store: Store) -> str:
    cfg = {"base_model": FZ, "rl_model": T1, "exclude_defining_names": True}
    run_id = new_interp_run(store, "kl", cfg)
    rows = [
        {
            "pid": f"k{i}",
            "suite": "interp_eval_dev",
            "direction": d,
            "policy": pol,
            "ok": i % 2 == 0,
            "n_steps": 4,
            "n_tokens": 10,
            "kl_sum": kl * 10,
            "kl_per_token": kl,
            "logratio_sum": kl * 10,
            # NEGE carries 6x the token-average KL, PR none
            "by_kind": {"NEGE": [kl * 6, 1], "PR": [0.0, 5], "AS": [kl * 4, 4]},
        }
        for d, pol, kl in (("rl||base", "t1", 0.2), ("base||rl", "fz", 0.05))
        for i in range(4)
    ]
    write_jsonl(store.store_dir / "interp" / "kl" / f"{run_id}.jsonl", rows)
    # the run's printed summary records the attempts it could not score
    logs = Path(store.root) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "interp_kl_dan12.log").write_text(
        json.dumps({"n_rows": len(rows), "n_skipped": 8, "run_id": run_id}, indent=2)
    )
    return run_id


@pytest.fixture(scope="module")
def synth(tmp_path_factory: pytest.TempPathFactory) -> Store:
    store = Store(tmp_path_factory.mktemp("dan_only") / "store")
    build_fully_populated_store(store)
    populate_dan(store)
    return store


@pytest.fixture(scope="module")
def charts(synth: Store) -> list[ChartData]:
    return dan_only.dan_only_chart_data(synth, registry=REGISTRY)


def _by(charts: list[ChartData]) -> dict[str, ChartData]:
    return {c.key: c for c in charts}


# ================================================================================ rendering


def test_every_chart_has_data_except_kl(charts: list[ChartData]) -> None:
    by = _by(charts)
    assert [c.key for c in charts][:2] == ["do_textbook", "do_table1"]
    missing = {c.key for c in charts if c.kind == "missing"}
    assert missing == {"do_kl"}, {c.key: c.missing for c in charts if c.kind == "missing"}
    assert by["do_kl"].extra["missing_label"] == "not yet run"
    for c in charts:
        if c.kind != "missing":
            assert c.takeaway and c.caption, c.key
            # every judged benchmark names its judge; likelihood/probes judge nothing
            assert "Judge: Lean" in c.caption or c.key in ("do_likelihood", "do_probes"), c.key


def test_page_renders_sections_figures_and_tables(synth: Store, tmp_path: Path) -> None:
    path = dan_only.build_dan_only(synth, tmp_path / "site", registry=REGISTRY)
    assert path.name == "index.html"
    page = path.read_text()
    for needle in (
        "Figure 1:",
        "Eliciting proofs",
        "Internals",
        "Limitations",
        "Table 1:",
        "Table 2:",
        "Figure 14:",
        "not yet run",
        "99.",
        "data:image/svg+xml",
    ):
        assert needle in page, needle
    assert "<a " not in page  # the site's only page: links nowhere
    assert len(page.encode()) < 16 * 1024 * 1024


def _visible_text(page: str) -> str:
    page = re.sub(r'src(set)?="data:[^"]*"', "", page)
    page = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", page, flags=re.S)
    return html.unescape(re.sub(r"<[^>]+>", " ", page))


def test_no_base_rl_or_armc_anywhere(synth: Store, charts: list[ChartData], tmp_path: Path) -> None:
    # page text, captions, takeaways and data tables
    text = _visible_text(dan_only.render_page(synth, registry=REGISTRY))
    text = text.replace(dv.DAN_SOURCE, "")  # Dan's summary path contains "after-rl-level"
    assert not OURS.findall(text), sorted(set(OURS.findall(text)))
    # every plotted series / bar / dot is one of Dan's models or a non-model series
    dan = set(dv.dan_models(REGISTRY))
    allowed = {"pass@1", "pass@128"}
    for c in charts:
        names = [s.name for s in c.series] + [b.name.rpartition(": ")[2] for b in c.bars]
        names += [n for r in c.rows for n, *_ in r.points]
        for p in c.extra.get("panels", {}).values():
            names += [s.name for s in p.series]
        for n in names:
            label = n.rpartition("|")[2].rpartition(": ")[2]
            label = (
                label.removeprefix("steer:").split(":")[0] if label.startswith("steer:") else label
            )
            assert label in dan or label in allowed, (c.key, n)
    # every figure's text (legends, ticks, panel titles), including the overview
    with S.paper_style():
        figs = [F.fig_overview(charts), F.fig_overview(charts, narrow=True)]
        for c in charts:
            for narrow in (False, True):
                fig = F.render(c, narrow=narrow)
                if fig is not None:
                    figs.append(fig)
        for fig in figs:
            words = " ".join(t.get_text() for t in fig.findobj(Text))
            assert not OURS.findall(words), OURS.findall(words)
            S.close(fig)


def test_empty_store_gives_placeholders_not_crashes(tmp_path: Path) -> None:
    store = Store(tmp_path / "empty")
    charts = dan_only.dan_only_chart_data(store, registry=REGISTRY)
    assert all(c.kind == "missing" for c in charts)
    page = dan_only.render_page(store, registry=REGISTRY)
    assert "no data yet" in page and "not yet run" in page and "Limitations" in page


# ============================================================================== takeaways


def test_takeaways_are_computed_from_the_data(charts: list[ChartData]) -> None:
    by = _by(charts)
    # Table 1: synthetic counts vs Dan's published ones (dan12 t1 s0: 3 + 1 = 4 vs 52)
    t72 = by["do_table1"].extra["t72"]
    assert t72["dan12-t1"]["ours"] == [4, 4, 6]
    assert t72["dan12-t1"]["dan"] == [52, 51, 52]
    worst = max(abs(o - d) for v in t72.values() for o, d in zip(v["ours"], v["dan"], strict=True))
    assert f"to within {worst} theorems" in by["do_table1"].takeaway
    assert "nd_v2 on the same samples changes no count" in by["do_table1"].caption
    # padding: t1 at m = 2, P2 50%, P1-shared 25%, P1-disjoint 0%; fz from 75%
    pad = by["do_padding"].takeaway
    assert "more distractors hurt t1 more than a longer proof" in pad
    assert "from 100.0% unpadded" in pad and "leave 50.0%" in pad and "25.0% (P1-shared)" in pad
    assert "fz goes from 75.0%" in pad
    # steering: the best fz setting is layer 1, alpha 0.5; alpha 2 the most harmful
    st = by["do_steering"].takeaway
    assert "(layer 1, α = 0.5)" in st and "(layer 1, α = 2)" in st
    # heldout: fz 8/10, t1 10/10
    assert "from 20.0% to 0.0%" in by["do_heldout"].takeaway
    # skills: case split 62.5% vs plain 50.0% for t1
    assert "Case splits do not cost t1 anything" in by["do_skills"].takeaway
    assert "62.5% of case-split items vs 50.0%" in by["do_skills"].takeaway
    # textbook_law: the seen theorem tl0 is excluded (n = 3)
    assert "n = 3; 1 seen theorems excluded" in by["do_textbook_law"].caption
    # likelihood: paired medians -0.60 and +1.20 nats/line
    assert "-0.60 nats/line" in by["do_likelihood"].takeaway
    assert "+1.20 nats/line" in by["do_likelihood"].takeaway
    # probes: baselines read from the run's printed output
    assert "(70.0%)" in by["do_probes"].takeaway
    # families: t1 longer than fz in the one family
    assert "longer than fz's in 1 of 1 families" in by["do_families"].takeaway


def test_padding_split_counts_each_base_once(synth: Store) -> None:
    data = dv.padding_by_pad_type(synth, T1)
    assert list(data) == ["P2", "P1-shared", "P1-disjoint"]
    for pts in data.values():
        m0 = pts[0]
        assert m0[0] == 0 and m0[2] == 2  # two bases, once each, in every panel
        assert [p[0] for p in pts] == [0, 1, 2]
    assert data["P2"][2][1] == pytest.approx(0.5)


def test_kl_run_replaces_the_placeholder(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    assert dan_only._c_kl(store).kind == "missing"  # noqa: SLF001
    add_kl_run(store)
    cd = dan_only._c_kl(store)  # noqa: SLF001
    assert cd.kind == "dot"
    assert [r.label for r in cd.rows] == ["all tokens", "NEGE", "AS", "PR"]
    take = cd.takeaway
    assert take.startswith("t1 and fz differ most on elimination steps")
    assert "KL(t1 ‖ fz) 0.200 nats/token (2.0 per sample)" in take
    assert "KL(fz ‖ t1) 0.050 nats/token (0.5 per sample)" in take
    assert "eliminations (NEGE) carry 0.30-0.30 (fz" not in take  # ordered fz||t1 first
    assert "0.30-0.30 (KL(fz ‖ t1)) and 1.20-1.20 (KL(t1 ‖ fz))" in take
    assert "PR stay under 0.02 in both directions" in take
    assert "defining-name tokens are excluded" in cd.caption
    assert "8 of 16 sampled attempts (50%) were skipped" in cd.caption
    page = dan_only.render_page(store, registry=REGISTRY)
    assert "covers in-grammar attempts only" in page
    with S.paper_style():
        fig = F.render(cd)
        assert fig is not None
        S.close(fig)


def test_probe_baselines_come_from_the_run_output(synth: Store) -> None:
    data = dv.dan_probes(synth)
    assert data is not None
    assert data["baselines"]["next_rule_acc_step_index_premises_depth"] == 0.7
    assert [x[0] for x in data["models"]["fz"]] == list(range(6))


# =================================================================== other pages unchanged


def test_main_summary_and_with_dan_pages_are_unaffected(synth: Store, tmp_path: Path) -> None:
    """Rendering the Dan-only page leaves no state behind (style registry, caches) that would
    change the main summary or the `--with-dan` pages, and the audited Dan builders' defaults
    still compute every family including ours."""
    before_main = summary.build_summary(synth, tmp_path / "a").read_bytes()
    before_dan = dan_pages.render_dan_section(synth, registry=REGISTRY)
    dan_only.render_page(synth, registry=REGISTRY)
    assert summary.build_summary(synth, tmp_path / "b").read_bytes() == before_main
    assert dan_pages.render_dan_section(synth, registry=REGISTRY) == before_dan
    full = dan_pages._data_dan_yue(synth, REGISTRY)  # noqa: SLF001
    assert set(full.extra["panels"]) == {"ours", "dan12", "dan6"}


def test_cli_dan_only_page_and_figures(synth: Store, tmp_path: Path) -> None:
    out = str(synth.root)
    assert (
        cli.main(
            ["dash", "--out", out, "--dash-dir", str(tmp_path / "d"), "--summary", "--dan-only"]
        )
        == 0
    )
    assert (tmp_path / "d" / "index.html").exists()
    assert not (tmp_path / "d" / "summary.html").exists()
    assert cli.main(["dash", "--out", out, "--dash-dir", str(tmp_path / "e"), "--dan-only"]) != 0
    assert (
        cli.main(
            [
                "dash",
                "--out",
                out,
                "--dash-dir",
                str(tmp_path / "e"),
                "--summary",
                "--dan-only",
                "--with-dan",
            ]
        )
        != 0
    )
    assert cli.main(["figures", "--out", out, "--fig-dir", str(tmp_path / "f"), "--dan-only"]) == 0
    names = {p.name for p in (tmp_path / "f").iterdir()}
    assert {
        "fig01_overview.pdf",
        "fig02_do_textbook.pdf",
        "tab01_do_table1.tex",
        "tab02_do_steering.tex",
        "captions.md",
    } <= names
    tex = (tmp_path / "f" / "tab01_do_table1.tex").read_text()
    assert r"\toprule" in tex and "<b>" not in tex and "|" not in tex.split("tabular}{")[1][:8]


def _steering_store(root: Path, *, with_t1: bool) -> Store:
    store = Store(root / "store")
    for suite, pre in (("interp_eval_dev", "d"), ("interp_eval_sealed", "s")):
        probs = [make_problem(f"{pre}{i}", suite, "v1", length=9) for i in range(3)]
        store.write_suite(probs, manifest(suite, "v1", len(probs)))
    steer = {FZ: 3, f"steer:{FZ}:1:0.5": 4, f"steer:{T1}:1:-1": 6}
    if with_t1:
        steer[T1] = 7
    for m, c in steer.items():
        _run(store, "interp_eval_dev", m, 8, {f"d{i}": c for i in range(3)})
        _run(store, "interp_eval_sealed", m, 8, {f"s{i}": c for i in range(3)})
    return store


def test_steering_uses_the_unsteered_t1_baseline_once_it_exists(tmp_path: Path) -> None:
    without = dan_only._c_steering(_steering_store(tmp_path / "a", with_t1=False))  # noqa: SLF001
    assert "no unsteered t1 run" in without.takeaway
    assert "t1 has no unsteered run" in without.caption
    assert not without.opts["has_t1_baseline"]
    cd = dan_only._c_steering(_steering_store(tmp_path / "b", with_t1=True))  # noqa: SLF001
    assert cd.opts["has_t1_baseline"]
    labels = [r[0] for r in cd.extra["table"]["rows"]]
    assert labels[-2:] == ["t1 (unsteered)", "t1 − δ (layer 1)"]  # side by side
    assert "against 87.5% / 87.5% unsteered (−12.5 / −12.5 points;" in cd.takeaway
    assert "t1 has no unsteered run" not in cd.caption
    assert any(b.name.endswith(f": t1|{T1}") for b in cd.bars)


def test_likelihood_prefers_the_newest_run_without_name_tokens(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")

    def rows(delta: float) -> list[dict[str, JSON]]:
        return [
            {
                "pid": f"r{i}",
                "origin": "reserve",
                "L": 6,
                "lines": 6,
                "fz": {"logp_marginal": -6.0, "n_orderings": 1},
                "t1": {"logp_marginal": -6.0 + delta, "n_orderings": 1},
            }
            for i in range(3)
        ]

    pilot = new_interp_run(store, "likelihood", {"base_model": FZ, "rl_model": T1, "n": 1})
    write_interp_rows(store, "likelihood", pilot, rows(-0.6))
    cd = dan_only._c_likelihood(store)  # noqa: SLF001
    assert "include the environment-assigned defining-name tokens" in cd.caption
    noname = new_interp_run(
        store, "likelihood", {"base_model": FZ, "rl_model": T1, "exclude_defining_names": True}
    )
    write_interp_rows(store, "likelihood", noname, rows(1.2))
    newer = new_interp_run(store, "likelihood", {"base_model": FZ, "rl_model": T1, "n": 3})
    write_interp_rows(store, "likelihood", newer, rows(-1.2))
    cd = dan_only._c_likelihood(store)  # noqa: SLF001
    assert cd.extra["run_id"] == noname
    assert "defining-name tokens are excluded from the score" in cd.caption
    assert "+0.20 nats/line" in cd.takeaway
