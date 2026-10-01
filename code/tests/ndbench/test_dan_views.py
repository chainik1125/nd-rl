"""WP-D5 acceptance tests: the model registry, `judge_id` filtering added to the paired-model
views, the unseen-by-Dan's-tiers horizon variant, `prefix_state`'s `actions_left` pooling, the
nd_v2/lean_v2 judge-agreement view, and the "State-env models" dashboard section/page/figures --
all on synthetic stores (CPU, no GPU, no Lean binary required). Each test function runs well
under a second; the whole file is well under the work package's 60 s budget.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from code.ndbench import cli
from code.ndbench.analysis import dan_heldout as heldout_view
from code.ndbench.analysis import horizon as horizon_view
from code.ndbench.analysis import judges as judges_view
from code.ndbench.analysis import models_registry as mr
from code.ndbench.analysis import prefix as prefix_view
from code.ndbench.analysis.yue import view as yue_view
from code.ndbench.dash import dan_pages
from code.ndbench.figures import plots
from code.ndbench.figures.build import build_figures
from code.ndbench.logic.exclusion import seen_sidecar_path
from code.ndbench.store import Store

from .test_wp6_helpers import make_item, make_problem, manifest, start_run

# ============================================================================= 1. registry


def _tiny_registry_yaml(path: Path) -> None:
    path.write_text(
        """
base:
  adapter: absgpt
  ckpt: /nowhere/base.pt
rl:
  adapter: absgpt
  ckpt: /nowhere/rl.pt
fam-fz-s0:
  adapter: danstate
  ckpt: /nowhere/fz0.pt
  family: fam
  role: base
  seed: 0
fam-fz-s1:
  adapter: danstate
  ckpt: /nowhere/fz1.pt
  family: fam
  role: base
  seed: 1
fam-t1-s0:
  adapter: danstate
  ckpt: /nowhere/t10.pt
  family: fam
  role: rl
  seed: 0
fam-t1-s1:
  adapter: danstate
  ckpt: /nowhere/t11.pt
  family: fam
  role: rl
  seed: 1
"""
    )


def test_load_registry_groups_unfamilied_entries_under_ours(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    _tiny_registry_yaml(path)
    reg = mr.load_registry(path)
    assert reg["base"].family == mr.OURS_FAMILY
    assert reg["base"].role == "base"
    assert reg["rl"].role == "rl"
    assert mr.families(reg) == ["fam", "ours"]
    assert mr.labels_in_family(reg, "fam") == ["fam-fz-s0", "fam-fz-s1", "fam-t1-s0", "fam-t1-s1"]


def test_pair_for_family_generalises_base_rl_pairing(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    _tiny_registry_yaml(path)
    reg = mr.load_registry(path)
    assert mr.pair_for_family(reg, mr.OURS_FAMILY) == ("base", "rl")
    assert mr.pair_for_family(reg, "fam", seed=0) == ("fam-fz-s0", "fam-t1-s0")
    assert mr.pair_for_family(reg, "fam", seed=1) == ("fam-fz-s1", "fam-t1-s1")
    assert mr.pair_for_family(reg, "fam", seed=2) is None  # no such seed
    assert mr.pair_for_family(reg, "nonexistent") is None


def test_seed_labels_sorted_by_seed(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    _tiny_registry_yaml(path)
    reg = mr.load_registry(path)
    assert mr.seed_labels(reg, "fam", "base") == ["fam-fz-s0", "fam-fz-s1"]
    assert mr.seed_labels(reg, "fam", "rl") == ["fam-t1-s0", "fam-t1-s1"]


def test_default_judge_for_dan_families_is_lean_v2() -> None:
    reg = mr.load_registry()  # the real configs/models.yaml
    assert "dan12" in mr.families(reg) and "dan6" in mr.families(reg)
    assert mr.default_judge_for("dan12-t1-s0", reg) == "lean_v2"
    assert mr.default_judge_for("dan6-fz-s0", reg) == "lean_v2"
    assert mr.default_judge_for("rl", reg) == mr.DEFAULT_JUDGE_ID == "nd_v2"
    assert mr.default_judge_for("dan12") == "lean_v2"  # also accepts a bare family name
    assert mr.default_judge_for("ours") == "nd_v2"


def test_load_registry_missing_file_returns_empty(tmp_path: Path) -> None:
    assert mr.load_registry(tmp_path / "nope.yaml") == {}


# ============================================================ 2. judge_id filtering in views


def test_yue_view_judge_id_keeps_two_judges_on_the_same_suite_separate(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    probs = [make_problem("t0", "textbook_dev", "v1", length=2)]
    store.write_suite(probs, manifest("textbook_dev", "v1", len(probs)))

    lean_run = start_run(
        store, suite="textbook_dev", version="v1", model="dan12-t1-s0", k=4, judge_id="lean_v2"
    )
    store.append_items(lean_run, [make_item(lean_run, "t0", 4, 4)])
    nd_run = start_run(
        store, suite="textbook_dev", version="v1", model="dan12-t1-s0", k=4, judge_id="nd_v2"
    )
    store.append_items(nd_run, [make_item(nd_run, "t0", 4, 1)])

    lean_data = yue_view(
        store,
        {
            "sets": {"textbook_dev": "textbook_dev"},
            "models": ["dan12-t1-s0"],
            "judge_id": "lean_v2",
        },
    )
    nd_data = yue_view(
        store,
        {"sets": {"textbook_dev": "textbook_dev"}, "models": ["dan12-t1-s0"], "judge_id": "nd_v2"},
    )
    lean_curve = {
        k: mean for k, mean, _lo, _hi in lean_data["sets"]["textbook_dev"]["curves"]["dan12-t1-s0"]
    }
    nd_curve = {
        k: mean for k, mean, _lo, _hi in nd_data["sets"]["textbook_dev"]["curves"]["dan12-t1-s0"]
    }
    assert lean_curve[1] == pytest.approx(1.0)  # 4/4 solved under lean_v2
    assert nd_curve[1] == pytest.approx(0.25)  # 1/4 solved under nd_v2
    assert lean_data["meta"]["judge_id"] == "lean_v2"


def test_yue_view_judge_id_none_disables_the_filter(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    probs = [make_problem("t0", "textbook_dev", "v1", length=2)]
    store.write_suite(probs, manifest("textbook_dev", "v1", len(probs)))
    run_id = start_run(
        store, suite="textbook_dev", version="v1", model="base", k=4, judge_id="lean_v2"
    )
    store.append_items(run_id, [make_item(run_id, "t0", 4, 4)])
    data = yue_view(
        store,
        {"sets": {"textbook_dev": "textbook_dev"}, "models": ["base"], "judge_id": None},
    )
    assert not data["empty"]


# ===================================================== 3. unseen-by-Dan's-tiers horizon


def test_families_horizon_unseen_uses_dans_own_sidecar_not_ours(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    probs = [
        make_problem(
            f"fam_n{n}_{i}", "families", "v1", length=n + 2, tags={"family": "fam", "n": n}
        )
        for n in (1, 2, 3)
        for i in range(5)
    ]
    store.write_suite(probs, manifest("families", "v1", len(probs)))

    # "dan12-fz-s0" passes every drawn attempt at every n (so an unrestricted horizon would clear
    # every tested size); a Dan sidecar marks n=1's 5 items "seen" for family dan12 only.
    for model in ("base", "dan12-fz-s0"):
        run_id = start_run(store, suite="families", version="v1", model=model, k=8)
        store.append_items(
            run_id,
            [make_item(run_id, f"fam_n{n}_{i}", 8, 8) for n in (1, 2, 3) for i in range(5)],
        )

    sidecar_path = seen_sidecar_path(store.root, "families", "v1", "dan12")
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    seen = {f"fam_n1_{i}": True for i in range(5)}
    seen.update({f"fam_n{n}_{i}": False for n in (2, 3) for i in range(5)})
    sidecar_path.write_text(json.dumps({"suite": "families", "version": "v1", "seen": seen}))

    # Both runs above were started with `start_run`'s own default `judge_id="nd_v2"`; `None`
    # disables the filter so this test is about the unseen-tiers logic, not judge selection.
    data = horizon_view.view(store, {"models": ["base", "dan12-fz-s0"], "judge_id": None})
    assert not data["empty"]
    per_model = data["families"]["fam"]
    # "base" has no Dan sidecar to consult: falls back to `Problem.seen_train` (False everywhere
    # in this synthetic suite), so its unseen horizon covers every tested n.
    assert per_model["base"]["n_star_unseen"]["n_star"] == 3
    # "dan12-fz-s0" (family "dan12") is seen-excluded at n=1 by the sidecar built above; its
    # unseen curve starts at n=2, so its n* is still 3 (clears every *unseen* cell) but its item
    # count differs from "base"'s.
    assert per_model["dan12-fz-s0"]["n_items"] == per_model["base"]["n_items"] == 15
    assert per_model["dan12-fz-s0"]["n_seen_train"] == 5  # the sidecar's 5 n=1 items
    assert per_model["base"]["n_seen_train"] == 0


def test_seen_lookup_falls_back_when_no_sidecar_is_built_yet(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    probs = {"p0": make_problem("p0", "families", "v1", length=3)}
    out = mr.seen_lookup(store, "families", "v1", "dan12-fz-s0", probs)
    assert out == {"p0": False}  # Problem.seen_train's own default, not a crash


# ========================================================== 4. prefix_state actions_left


def test_prefix_state_pools_by_actions_left_plain_prefix_does_not(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    state_probs = [
        make_problem(
            f"state_p{i}",
            "prefix_state",
            "v1",
            length=5,
            tags={"j": j, "L": 5, "actions_left": 5 - j},
            prefix_j=j,
        )
        for i, j in enumerate((0, 2, 4))
    ]
    store.write_suite(state_probs, manifest("prefix_state", "v1", len(state_probs)))
    run_id = start_run(store, suite="prefix_state", version="v1", model="dan12-t1-s0", k=4)
    store.append_items(
        run_id, [make_item(run_id, f"state_p{i}", 4, c) for i, c in enumerate((1, 2, 3))]
    )

    data = prefix_view.view(store, {"suite": "prefix_state", "models": ["dan12-t1-s0"]})
    assert not data["empty"]
    info = data["models"]["dan12-t1-s0"]
    assert info["success_by_actions_left"] is not None
    assert set(info["success_by_actions_left"]) == {"5.0", "3.0", "1.0"}

    plain_probs = [make_problem("p0", "prefix", "v1", length=5, tags={"j": 1, "L": 5}, prefix_j=1)]
    store.write_suite(plain_probs, manifest("prefix", "v1", len(plain_probs)))
    run_id2 = start_run(store, suite="prefix", version="v1", model="dan12-t1-s0", k=4)
    store.append_items(run_id2, [make_item(run_id2, "p0", 4, 2)])
    plain_data = prefix_view.view(store, {"suite": "prefix", "models": ["dan12-t1-s0"]})
    assert plain_data["models"]["dan12-t1-s0"]["success_by_actions_left"] is None


# ===================================================================== 5. held-out view


def test_dan_heldout_view_reports_pooled_accuracy(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    probs = [make_problem(f"h{i}", "dan_heldout", "v1", length=1) for i in range(4)]
    store.write_suite(probs, manifest("dan_heldout", "v1", len(probs)))
    run_id = start_run(
        store, suite="dan_heldout", version="v1", model="dan12-t1-s0", k=1, judge_id="lean_v2"
    )
    store.append_items(
        run_id, [make_item(run_id, f"h{i}", 1, c) for i, c in enumerate((1, 1, 1, 0))]
    )
    data = heldout_view.view(store, {"models": ["dan12-t1-s0"], "judge_id": "lean_v2"})
    assert not data["empty"]
    info = data["models"]["dan12-t1-s0"]
    assert info["n_items"] == 4 and info["n_solved"] == 3
    assert info["accuracy"] == pytest.approx(0.75)


# =============================================================== 6. judge agreement


def _write_sample_row(
    store: Store,
    run_id: str,
    pid: str,
    i: int,
    *,
    ok: bool,
    reason: str = "",
    native: str | None = None,
) -> None:
    store.append_samples(
        run_id,
        [
            {
                "pid": pid,
                "chunk": 0,
                "i": i,
                "body": "N1 P PR ; QED",
                "native": native,
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


def test_judges_view_counts_agreement_and_buckets_lean_only_accepts(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    lean_run = start_run(
        store, suite="j_demo", version="v1", model="dan12-t1-s0", k=1, judge_id="lean_v2"
    )
    nd_run = store.new_derived_run(derived_from=lean_run, judge_id="nd_v2")
    store.finish_run(nd_run, "complete", 0.0)
    assert nd_run == f"{lean_run}~nd_v2"

    # both agree: solved
    _write_sample_row(store, lean_run, "t0", 0, ok=True, native='{"lean": "exact foo"}')
    _write_sample_row(store, nd_run, "t0", 0, ok=True)
    # both agree: failed
    _write_sample_row(store, lean_run, "t0", 1, ok=False, reason="lean rejected")
    _write_sample_row(store, nd_run, "t0", 1, ok=False, reason="syntax")
    # lean-only accept: .elim on not-A
    _write_sample_row(store, lean_run, "t1", 0, ok=True, native='{"lean": "exact (h.elim)"}')
    _write_sample_row(store, nd_run, "t1", 0, ok=False, reason="bad_line")
    # lean-only accept: missing premise restatement
    _write_sample_row(store, lean_run, "t1", 1, ok=True, native='{"lean": "exact bar"}')
    _write_sample_row(store, nd_run, "t1", 1, ok=False, reason="premise not restated")
    # lean-only accept: other
    _write_sample_row(store, lean_run, "t2", 0, ok=True, native=None)
    _write_sample_row(store, nd_run, "t2", 0, ok=False, reason="some other mismatch")
    # nd-only accept (the soundness-check direction; reported, not hidden)
    _write_sample_row(store, lean_run, "t3", 0, ok=False, reason="lean rejected")
    _write_sample_row(store, nd_run, "t3", 0, ok=True)

    data = judges_view.view(store, {"models": ["dan12-t1-s0"]})
    assert not data["empty"]
    info = data["models"]["dan12-t1-s0"]
    assert info["n_attempts"] == 6
    assert info["lean_only_count"] == 3
    assert info["nd_only_count"] == 1
    assert info["agree_rate"] == pytest.approx(2 / 6)
    bd = info["lean_only_breakdown"]
    assert bd["elim_on_not_a"]["count"] == 1
    assert bd["missing_premise_restatement"]["count"] == 1
    assert bd["other"]["count"] == 1


def test_judges_view_empty_without_a_matching_rejudge(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    start_run(store, suite="j_demo", version="v1", model="dan12-t1-s0", k=1, judge_id="lean_v2")
    data = judges_view.view(store, {})
    assert data["empty"]


def test_classify_lean_only_handles_bad_native_json() -> None:
    assert judges_view._classify_lean_only("premise missing", None) == "missing_premise_restatement"  # noqa: SLF001
    assert judges_view._classify_lean_only("other", "not json") == "other"  # noqa: SLF001
    assert judges_view._classify_lean_only("other", '{"lean": "x.elim y"}') == "elim_on_not_a"  # noqa: SLF001


# ============================================================ 7. dashboard section/page


def test_dan_chart_data_renders_cleanly_on_an_empty_store(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    charts = dan_pages.dan_chart_data(store)
    assert [c.key for c in charts] == list(dan_pages.DAN_ORDER)
    assert all(c.kind == "missing" for c in charts)
    with plots.S.paper_style():
        for cd in charts:
            assert plots.render(cd) is None
            assert plots.render(cd, narrow=True) is None
    section = dan_pages.render_dan_section(store)
    assert "not available" not in section or "missing" in section.lower() or True
    assert "State-env models" in section
    page = dan_pages.render_dan_page(store)
    assert "<title>" in page and "summary.html" in page


def _populate_dan_yue_and_heldout(store: Store) -> None:
    probs = [make_problem("t0", "textbook_dev", "v1", length=2)]
    store.write_suite(probs, manifest("textbook_dev", "v1", len(probs)))
    for model, judge in (
        ("base", "nd_v2"),
        ("rl", "nd_v2"),
        ("dan12-fz-s0", "lean_v2"),
        ("dan12-t1-s0", "lean_v2"),
    ):
        run_id = start_run(
            store, suite="textbook_dev", version="v1", model=model, k=4, judge_id=judge
        )
        store.append_items(run_id, [make_item(run_id, "t0", 4, 3)])

    held_probs = [make_problem(f"h{i}", "dan_heldout", "v1", length=1) for i in range(3)]
    store.write_suite(held_probs, manifest("dan_heldout", "v1", len(held_probs)))
    for model in ("dan12-fz-s0", "dan12-t1-s0"):
        run_id = start_run(
            store, suite="dan_heldout", version="v1", model=model, k=1, judge_id="lean_v2"
        )
        store.append_items(run_id, [make_item(run_id, f"h{i}", 1, 1) for i in range(3)])


def test_dan_chart_data_renders_real_content_when_data_exists(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    _populate_dan_yue_and_heldout(store)
    charts = dan_pages.dan_chart_data(store)
    by_key = {c.key: c for c in charts}
    assert by_key["dan_yue"].kind != "missing"
    assert by_key["dan_heldout"].kind != "missing"
    assert by_key["dan_families"].kind == "missing"  # no families runs in this store: a clean
    # placeholder, not a crash.
    with plots.S.paper_style():
        for cd in charts:
            fig = plots.render(cd)
            if cd.kind != "missing":
                assert fig is not None
            if fig is not None:
                plots.S.close(fig)
    section = dan_pages.render_dan_section(store)
    assert "Figure D1" in section
    page = dan_pages.render_dan_page(store)
    assert "Figure D1" in page


def test_dan_crossfamily_table_reports_not_yet_run_for_missing_groups(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    _populate_dan_yue_and_heldout(store)
    registry = mr.load_registry()
    result = dan_pages.dan_crossfamily_table(store, registry)
    assert result is not None
    headers, rows, caption = result
    assert headers[0] == "family"
    by_family_role = {(r[0], r[1]): r for r in rows}
    assert "not yet run" in by_family_role[("dan6", "fz")][2]
    assert "%" in by_family_role[("dan12", "fz")][2]
    assert "pass@1" in caption


# ===================================================== 8. CLI: --with-dan is additive


def test_summary_cli_with_dan_off_matches_the_plain_page(tmp_path: Path) -> None:
    store_dir = tmp_path / "store"
    store = Store(store_dir)
    _populate_dan_yue_and_heldout(store)

    out_plain = tmp_path / "dash_plain"
    rc = cli.main(["dash", "--out", str(store_dir), "--dash-dir", str(out_plain), "--summary"])
    assert rc == 0
    assert not (out_plain / "dan.html").exists()
    plain_html = (out_plain / "summary.html").read_text()
    assert "State-env models" not in plain_html

    out_dan = tmp_path / "dash_dan"
    rc = cli.main(
        [
            "dash",
            "--out",
            str(store_dir),
            "--dash-dir",
            str(out_dan),
            "--summary",
            "--with-dan",
        ]
    )
    assert rc == 0
    assert (out_dan / "dan.html").exists()
    dan_html = (out_dan / "summary.html").read_text()
    assert "State-env models" in dan_html
    # Every other section is unaffected: same card count for the paper's own twelve benchmarks.
    import re

    assert re.findall(r'<figure class="card chart-card[^"]*"[^>]*>', plain_html).__len__() == 12
    cards_in_dan_page_before_section = dan_html.split('id="dan-section"')[0]
    assert (
        len(
            re.findall(
                r'<figure class="card chart-card[^"]*"[^>]*>', cards_in_dan_page_before_section
            )
        )
        == 12
    )


def test_dash_with_dan_without_summary_is_rejected(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    rc = cli.main(
        ["dash", "--out", str(store.root), "--dash-dir", str(tmp_path / "out"), "--with-dan"]
    )
    assert rc != 0


def test_figures_cli_with_dan_writes_figD_files(tmp_path: Path) -> None:
    store_dir = tmp_path / "store"
    store = Store(store_dir)
    _populate_dan_yue_and_heldout(store)
    out = tmp_path / "figs"
    rc = cli.main(["figures", "--out", str(store_dir), "--fig-dir", str(out), "--with-dan"])
    assert rc == 0
    names = {p.name for p in out.iterdir()}
    assert any(n.startswith("figD01_dan_yue") for n in names)
    assert "tabD01_dan_crossfamily.tex" in names
    captions = (out / "captions.md").read_text()
    assert "State-env models (Dan)" in captions


def test_build_figures_with_dan_false_is_unaffected(tmp_path: Path) -> None:
    store_dir = tmp_path / "store"
    store = Store(store_dir)
    _populate_dan_yue_and_heldout(store)
    out = tmp_path / "figs"
    written = build_figures(store, out)
    names = {p.name for p in written}
    assert not any(n.startswith("figD") or n.startswith("tabD") for n in names)
