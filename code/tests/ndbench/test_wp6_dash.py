"""WP6 acceptance tests for `dash/build.py`: every page renders on a synthetic store covering all
eight benchmarks, the generated HTML has no external URLs, and the dashboard also renders cleanly
from a legacy-only store (plan §3 WP6 row's acceptance tests)."""

from __future__ import annotations

from pathlib import Path

from code.ndbench.dash import build as dash_build
from code.ndbench.edl import run_edl
from code.ndbench.store import Store

from .test_wp5_fixtures import ConstantNllAdapter
from .test_wp6_edl import _toy_problems  # pyright: ignore[reportPrivateUsage]
from .test_wp6_helpers import make_item, make_problem, manifest, start_run

_BENCHMARK_PAGES = (
    "yue.html",
    "families.html",
    "glue.html",
    "skills.html",
    "subst.html",
    "padding.html",
    "prefix.html",
    "edl.html",
)
# Added by the interp work (dash/interp_pages.py); the synthetic benchmark store has no interp data,
# so these render their "no data yet" panels.
_INTERP_PAGES = (
    "interp_likelihood.html",
    "interp_steering.html",
    "interp_weights.html",
    "interp_probes.html",
)


def build_full_demo_store(store: Store) -> None:
    """Populates `store` with a minimal but non-empty instance of every benchmark (plan §3's
    acceptance test: "a synthetic store ... makes every view non-empty"). Shared by the dashboard
    test below and by the standalone demo-generation script (WP6 report)."""

    dev = [make_problem(f"dev{i}", "textbook_dev", "v1", length=8) for i in range(4)]
    store.write_suite(dev, manifest("textbook_dev", "v1", len(dev)))
    base_run = start_run(store, suite="textbook_dev", version="v1", model="base", k=32)
    rl_run = start_run(store, suite="textbook_dev", version="v1", model="rl", k=32)
    store.append_items(base_run, [make_item(base_run, f"dev{i}", 32, 16) for i in range(4)])
    store.append_items(rl_run, [make_item(rl_run, f"dev{i}", 32, 20) for i in range(4)])

    fam = [
        make_problem(
            f"and_tower_n{n}", "families", "v1", length=n + 2, tags={"family": "and_tower", "n": n}
        )
        for n in range(1, 4)
    ]
    store.write_suite(fam, manifest("families", "v1", len(fam)))
    fam_run = start_run(store, suite="families", version="v1", model="base", k=32)
    store.append_items(
        fam_run,
        [make_item(fam_run, f"and_tower_n{n}", 32, 32 if n < 3 else 0) for n in range(1, 4)],
    )

    parts = [
        make_problem("part_a", "parts_pool", "v1", length=3, ref_lines=3),
        make_problem("part_b", "parts_pool", "v1", length=4, ref_lines=4),
    ]
    store.write_suite(parts, manifest("parts_pool", "v1", len(parts)))
    parts_run = start_run(store, suite="parts_pool", version="v1", model="base", k=32)
    store.append_items(
        parts_run, [make_item(parts_run, "part_a", 32, 30), make_item(parts_run, "part_b", 32, 25)]
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
    store.append_items(
        joined_run,
        [make_item(joined_run, "joined1", 32, 16, long_enough=20, fail_lines=[2, 5, 8])],
    )

    ctrl = [make_problem("ctrl1", "gen_ctrl", "v1", length=8)] + [
        make_problem(f"nat{5 + i}", "gen_ctrl", "v1", length=5 + i) for i in range(3)
    ]
    store.write_suite(ctrl, manifest("gen_ctrl", "v1", len(ctrl)))
    ctrl_run = start_run(store, suite="gen_ctrl", version="v1", model="base", k=32)
    store.append_items(
        ctrl_run,
        [make_item(ctrl_run, "ctrl1", 32, 10)]
        + [make_item(ctrl_run, f"nat{5 + i}", 32, 20 - 8 * i) for i in range(3)],
    )

    sk = [
        make_problem("i1", "skill_pool", "v1", length=8, tags={"classical": True}),
        make_problem("i2", "skill_pool", "v1", length=8, tags={"classical": False}),
    ]
    store.write_suite(sk, manifest("skill_pool", "v1", len(sk)))
    sk_run = start_run(store, suite="skill_pool", version="v1", model="base", k=32)
    store.append_items(sk_run, [make_item(sk_run, "i1", 32, 32), make_item(sk_run, "i2", 32, 0)])

    laws = [
        make_problem(f"law_s{s}", "laws", "v1", length=7, tags={"schema": "contraposition", "s": s})
        for s in (0, 2, 4)
    ]
    store.write_suite(laws, manifest("laws", "v1", len(laws)))
    laws_run = start_run(store, suite="laws", version="v1", model="base", k=32)
    store.append_items(
        laws_run,
        [
            make_item(laws_run, f"law_s{s}", 32, 32 - 8 * i, tokens_mean=50.0 + 50 * i)
            for i, s in enumerate((0, 2, 4))
        ],
    )
    tb = [make_problem("tb1", "textbook_law", "v1", length=7, tags={"schema": "contraposition"})]
    store.write_suite(tb, manifest("textbook_law", "v1", len(tb)))
    tb_run = start_run(store, suite="textbook_law", version="v1", model="base", k=32)
    store.append_items(tb_run, [make_item(tb_run, "tb1", 32, 20)])

    pad = [
        make_problem(
            f"pad{m}", "padding", "v1", length=5 + m, tags={"pad_type": "P1-shared", "m": m}
        )
        for m in range(3)
    ]
    store.write_suite(pad, manifest("padding", "v1", len(pad)))
    pad_run = start_run(store, suite="padding", version="v1", model="base", k=32)
    store.append_items(pad_run, [make_item(pad_run, f"pad{m}", 32, 32 - 8 * m) for m in range(3)])

    pfx = [
        make_problem("pfx_a", "prefix", "v1", tags={"j": 2, "L": 10}),
        make_problem("pfx_b", "prefix", "v1", tags={"j": 5, "L": 10}),
    ]
    store.write_suite(pfx, manifest("prefix", "v1", len(pfx)))
    pfx_run = start_run(store, suite="prefix", version="v1", model="base", k=10)
    store.append_items(
        pfx_run,
        [
            make_item(pfx_run, "pfx_a", 10, 6, ok_written=[10] * 6, fail_lines=[3, 3, 4, 10]),
            make_item(pfx_run, "pfx_b", 10, 5, ok_written=[10] * 5, fail_lines=[6, 7, 8, 9, 10]),
        ],
    )

    # EDL rows in the real `run_edl` schema (the null-control learner keeps this fast and exact).
    edl_rows = run_edl(
        ConstantNllAdapter(c=2.0, label="base"),
        _toy_problems(8, 4),
        lrs=(1e-4,),
        orders=1,
        n=8,
        m=4,
        ckpts=(4, 8),
    )
    store.append_edl("edl-base-8-4", edl_rows)

    start_run(store, suite="textbook_dev", version="v1", model="base", k=256, status="legacy")


def test_dash_build_all_pages_nonempty_and_no_urls(tmp_path: Path) -> None:
    store = Store(tmp_path)
    build_full_demo_store(store)
    out_dir = tmp_path / "dash_out"
    written = dash_build.build(store, out_dir)
    names = {p.name for p in written}
    assert names == {*_BENCHMARK_PAGES, *_INTERP_PAGES, "runs.html", "index.html"}

    for name in _BENCHMARK_PAGES:
        text = (out_dir / name).read_text()
        assert "no data yet" not in text, f"{name} unexpectedly rendered empty"

    for path in written:
        text = path.read_text()
        assert "http://" not in text
        assert "https://" not in text

    runs_text = (out_dir / "runs.html").read_text()
    assert "legacy" in runs_text
    index_text = (out_dir / "index.html").read_text()
    assert "legacy" in index_text  # the run-status summary table


def test_dash_build_from_legacy_only_store(tmp_path: Path) -> None:
    store = Store(tmp_path)
    start_run(store, suite="textbook_dev", version="v1", model="base", k=256, status="legacy")
    start_run(store, suite="textbook_dev", version="v1", model="rl", k=256, status="legacy")
    out_dir = tmp_path / "dash_out"
    written = dash_build.build(store, out_dir)
    assert len(written) == len(_BENCHMARK_PAGES) + len(_INTERP_PAGES) + 2

    for path in written:
        text = path.read_text()
        assert "http://" not in text
        assert "https://" not in text

    runs_text = (out_dir / "runs.html").read_text()
    assert "legacy" in runs_text
    for name in _BENCHMARK_PAGES:
        text = (out_dir / name).read_text()
        assert "no data yet" in text
