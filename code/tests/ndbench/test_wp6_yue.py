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
    """`import_legacy` writes the pre-ndbench textbook comparison under its own suite name
    (`"legacy_textbook"`, not `"textbook_dev"`) and its own model labels (`"base"`/`"champion"`,
    not `"base"`/`"rl"`) -- `yue.view` must look there, not at `(suite, model)` reused verbatim."""
    store = _build_crossover_store(tmp_path)
    legacy_base = start_run(
        store, suite="legacy_textbook", version="k256_s0", model="base", k=256, status="legacy"
    )
    # legacy agrees with the current first-chunk c/n on every item (same p=0.1, same n).
    store.append_items(
        legacy_base, [make_item(legacy_base, f"p{i}", 256, round(0.1 * 256)) for i in range(40)]
    )
    # the legacy "champion" row (-> this dashboard's "rl") disagrees on every item (p=0 vs rl's
    # actual p=0.9/0.0 mix), to check the "rl" -> "champion" direction is actually looked up, not
    # silently skipped.
    legacy_champ = start_run(
        store, suite="legacy_textbook", version="k256_s0", model="champion", k=256, status="legacy"
    )
    store.append_items(legacy_champ, [make_item(legacy_champ, f"p{i}", 256, 0) for i in range(40)])

    data = yue.view(store, {"B": 50})
    legacy = data["legacy_consistency"]["textbook_dev"]
    base_lc = legacy["base"]
    assert base_lc["n_common"] == 40
    assert base_lc["agree_rate"] == 1.0
    assert base_lc["mean_abs_diff"] == 0.0

    rl_lc = legacy["rl"]
    assert rl_lc["n_common"] == 40
    # rl's own first-chunk c is 0.9*256 on 10 items and 0 on 30; legacy champion is 0 everywhere,
    # so they "agree" (both > 0, or both == 0) only on the 30 items where rl is also 0.
    assert rl_lc["agree_rate"] == 30 / 40


def test_yue_legacy_consistency_absent_for_sets_with_no_legacy_equivalent(tmp_path: Path) -> None:
    """`textbook_law`/`gen_ctrl` have no pre-existing legacy run to compare against -- the page
    must simply omit them from `legacy_consistency`, not crash or fabricate a comparison."""
    store = Store(tmp_path)
    problems = [make_problem(f"p{i}", "gen_ctrl", "v1", length=8) for i in range(5)]
    store.write_suite(problems, manifest("gen_ctrl", "v1", len(problems)))
    run_id = start_run(store, suite="gen_ctrl", version="v1", model="base", k=8)
    store.append_items(run_id, [make_item(run_id, f"p{i}", 8, 4) for i in range(5)])

    data = yue.view(store, {"sets": {"control": "gen_ctrl"}})
    assert data["empty"] is False
    assert "control" not in data["legacy_consistency"]


def test_yue_ignores_interp_steering_and_weight_runs_sharing_a_set(tmp_path: Path) -> None:
    """WP8 integration fix list item 3: `textbook_dev` doubles as one of INTERP_SPEC.md's
    steering/interpolation eval suites, so it carries `steer:<layer>:<alpha>`/`interp:<alpha>` runs
    alongside the real base/rl/armC ones. Without filtering to `yue.DEFAULT_MODELS`,
    `models_present` would surface every one of those as its own "model", turning the Yue page
    into dozens of noise curves instead of a clean base-vs-RL(-vs-armC) replication."""
    store = Store(tmp_path)
    problems = [make_problem(f"p{i}", "textbook_dev", "v1", length=8) for i in range(5)]
    store.write_suite(problems, manifest("textbook_dev", "v1", len(problems)))
    for model in ("base", "rl", "armC", "steer:0:0.5", "interp:0.25"):
        run_id = start_run(store, suite="textbook_dev", version="v1", model=model, k=8)
        store.append_items(run_id, [make_item(run_id, f"p{i}", 8, 4) for i in range(5)])

    data = yue.view(store, {})
    assert data["empty"] is False
    assert set(data["sets"]["textbook_dev"]["models"]) == {"base", "rl", "armC"}


def test_yue_resolves_each_sets_suite_version_independently(tmp_path: Path) -> None:
    """Fix list item 1, applied per-set: `textbook_dev` and `gen_ctrl` are different suites and can
    each land on their own version with real runs -- `view` must not default every set to the same
    (or a hard-coded "v1") version."""
    store = Store(tmp_path)
    # textbook_dev: an unrun "v1" demo alongside the real "v2".
    demo = [make_problem("demo0", "textbook_dev", "v1", length=8)]
    store.write_suite(demo, manifest("textbook_dev", "v1", len(demo)))
    real = [make_problem(f"p{i}", "textbook_dev", "v2", length=8) for i in range(5)]
    store.write_suite(real, manifest("textbook_dev", "v2", len(real)))
    run_id = start_run(store, suite="textbook_dev", version="v2", model="base", k=8)
    store.append_items(run_id, [make_item(run_id, f"p{i}", 8, 4) for i in range(5)])

    # gen_ctrl only ever has "v1".
    gc = [make_problem(f"g{i}", "gen_ctrl", "v1", length=8) for i in range(5)]
    store.write_suite(gc, manifest("gen_ctrl", "v1", len(gc)))
    gc_run = start_run(store, suite="gen_ctrl", version="v1", model="base", k=8)
    store.append_items(gc_run, [make_item(gc_run, f"g{i}", 8, 2) for i in range(5)])

    data = yue.view(store, {"sets": {"textbook_dev": "textbook_dev", "control": "gen_ctrl"}})
    assert data["empty"] is False
    assert data["sets"]["textbook_dev"]["version"] == "v2"
    assert data["sets"]["control"]["version"] == "v1"


def test_yue_caps_each_sets_curve_at_its_own_k_max(tmp_path: Path) -> None:
    """Plan §2 B8: textbook_dev's own draw protocol targets k_max 4096, textbook_law 1024, and the
    gen_ctrl control set draws a fixed, non-adaptive k=512 that was never meant to be extrapolated
    past -- one shared 2^0..2^12 grid for every set would show textbook_law/control curves (and
    their crossover/per-length breakdowns) well past the k that set's own sampling supports."""
    store = Store(tmp_path)
    for suite, k in (("textbook_law", 1024), ("gen_ctrl", 512)):
        problems = [make_problem(f"{suite}{i}", suite, "v1", length=8) for i in range(5)]
        store.write_suite(problems, manifest(suite, "v1", len(problems)))
        run_id = start_run(store, suite=suite, version="v1", model="base", k=k)
        store.append_items(run_id, [make_item(run_id, f"{suite}{i}", k, k // 2) for i in range(5)])

    data = yue.view(store, {"sets": {"textbook_law": "textbook_law", "control": "gen_ctrl"}})
    assert data["empty"] is False
    law_ks = [k for k, *_ in data["sets"]["textbook_law"]["curves"]["base"]]
    assert max(law_ks) == 1024
    assert data["sets"]["textbook_law"]["k_max"] == 1024
    ctrl_ks = [k for k, *_ in data["sets"]["control"]["curves"]["base"]]
    assert max(ctrl_ks) == 512
    assert data["sets"]["control"]["k_max"] == 512


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
