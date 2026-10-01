"""`analysis/interp.py`'s four views and `dash/interp_pages.py`'s four renderers: "no data yet" on
an empty `Store`, real (synthetic) data renders without external URLs, and the whole dashboard
(`dash/build.py`) still builds with the four new pages wired in (`test_wp6_dash.py` covers the
pre-existing eight; this file is this work package's own acceptance test for its four additions,
per the spec's "the dashboard builds with the new pages, whether data is present or empty")."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from code.ndbench.analysis import interp as interp_view
from code.ndbench.dash import build as dash_build
from code.ndbench.dash import interp_pages
from code.ndbench.interp import eval_suites
from code.ndbench.interp._store import new_interp_run, write_interp_rows
from code.ndbench.store import Store

from .test_wp6_dash import build_full_demo_store
from .test_wp6_helpers import make_item, make_problem, manifest, start_run

# --------------------------------------------------------------------------------------- empty


def test_views_are_empty_on_a_fresh_store(tmp_path: Path) -> None:
    store = Store(tmp_path)
    for view_fn in (
        interp_view.view_likelihood,
        interp_view.view_steering,
        interp_view.view_interpolation,
        interp_view.view_probes,
    ):
        data = view_fn(store, {})
        assert data.get("empty") is True
        assert "reason" in data


def test_renderers_show_no_data_yet_panel_on_empty_views() -> None:
    """Each renderer prepends a short intro card (what the benchmark measures, fix list item 3)
    even on an empty view, so a reader lands on an explained "no data yet" panel rather than a
    bare one; the "no data yet" panel itself is always the renderer's own last figure."""
    for render_fn in (
        interp_pages.render_interp_likelihood,
        interp_pages.render_interp_steering,
        interp_pages.render_interp_weights,
        interp_pages.render_interp_probes,
    ):
        figs = render_fn({"empty": True, "reason": "no data"})
        assert len(figs) == 2
        assert "no data yet" in figs[-1]


# ------------------------------------------------------------------------- likelihood, non-empty


def _likelihood_row(
    pid: str, origin: str, base_marg: float, rl_marg: float, p_hat: float | None
) -> dict[str, Any]:
    return {
        "pid": pid,
        "origin": origin,
        "L": 8,
        "base": {
            "logp_single": base_marg - 0.1,
            "logp_marginal": base_marg,
            "n_orderings": 4,
            "method": "exact",
            "ess": None,
        },
        "rl": {
            "logp_single": rl_marg - 0.1,
            "logp_marginal": rl_marg,
            "n_orderings": 4,
            "method": "exact",
            "ess": None,
        },
        "p_hat_base": p_hat,
        "disagreement": {
            "n_positions": 10,
            "n_disagree": 2,
            "disagree_frac": 0.2,
            "mean_base_rank": 1.5,
            "mean_base_entropy": 0.8,
        }
        if origin == "rl_verified"
        else None,
    }


def test_view_and_render_likelihood_with_data(tmp_path: Path) -> None:
    store = Store(tmp_path)
    rows = [
        _likelihood_row("r1", "reserve", -5.0, -4.0, 0.9),
        _likelihood_row("r2", "reserve", -8.0, -3.0, 0.1),  # crux candidate
        _likelihood_row("r3", "rl_verified", -6.0, -2.0, 0.05),
    ]
    run_id = new_interp_run(store, "likelihood", {})
    write_interp_rows(store, "likelihood", run_id, rows)

    data = interp_view.view_likelihood(store, {})
    assert data["empty"] is False
    assert data["meta"]["n_rows"] == 3
    assert len(data["crux_set"]) == 2  # r2 and r3 both have p_hat_base < 0.5
    assert data["mean_disagree_frac"] == 0.2

    figs = interp_pages.render_interp_likelihood(data)
    assert figs
    joined = "\n".join(figs)
    assert "http://" not in joined and "https://" not in joined


# --------------------------------------------------------------------------- steering, non-empty


def test_view_and_render_steering_with_data(tmp_path: Path) -> None:
    store = Store(tmp_path)
    problems = [make_problem(f"d{i}", eval_suites.DEV_GEN_SUITE, "v1", length=8) for i in range(4)]
    store.write_suite(problems, manifest(eval_suites.DEV_GEN_SUITE, "v1", len(problems)))
    for model, c in (("base", 8), ("steer:0:0.5", 16), ("steer:0:1", 20), ("steer:0:-1", 4)):
        run_id = start_run(store, suite=eval_suites.DEV_GEN_SUITE, version="v1", model=model, k=32)
        store.append_items(run_id, [make_item(run_id, f"d{i}", 32, c) for i in range(4)])

    data = interp_view.view_steering(store, {"suite": eval_suites.DEV_GEN_SUITE, "B": 50})
    assert data["empty"] is False
    assert "0" in data["curves_by_layer"]
    alphas = {a for a, *_r in data["curves_by_layer"]["0"]}
    assert alphas == {0.5, 1.0}
    assert "0" in data["rl_control"]
    assert data["base_pass1"] is not None

    figs = interp_pages.render_interp_steering(data)
    assert figs
    joined = "\n".join(figs)
    assert "http://" not in joined and "https://" not in joined


def test_view_steering_pools_every_default_eval_suite_not_just_dev(tmp_path: Path) -> None:
    """INTERP_SPEC.md's steering sweep evaluates `interp_eval_dev`, `interp_eval_sealed` AND
    `textbook_dev` -- `view_steering` with no `cfg["suite"]`/`cfg["suites"]` override must pool
    every one of them, not silently read only `interp_eval_dev` (the pre-fix default). `base`'s
    pass@1 is 0 on dev, 1.0 on sealed, 0.5 on textbook_dev (2 items each): a dev-only reader gets
    0.0, a pooled reader gets the item-weighted mean (0+0+1+1+0.5+0.5)/6 = 0.5.
    """
    store = Store(tmp_path)
    suites_and_c = (
        (eval_suites.DEV_GEN_SUITE, 0),
        (eval_suites.SEALED_GEN_SUITE, 4),
        ("textbook_dev", 2),
    )
    for suite, base_c in suites_and_c:
        problems = [make_problem(f"{suite}_{i}", suite, "v1", length=8) for i in range(2)]
        store.write_suite(problems, manifest(suite, "v1", len(problems)))
        run_id = start_run(store, suite=suite, version="v1", model="base", k=4)
        store.append_items(run_id, [make_item(run_id, f"{suite}_{i}", 4, base_c) for i in range(2)])

    data = interp_view.view_steering(store, {"B": 50})
    assert data["empty"] is False
    assert set(data["meta"]["suites"]) == {s for s, _c in suites_and_c}
    assert data["base_pass1"][0] == 0.5


# ---------------------------------------------------------------------- interpolation, non-empty


def test_view_and_render_interpolation_with_data(tmp_path: Path) -> None:
    store = Store(tmp_path)
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

    data = interp_view.view_interpolation(store, {"B": 50})
    assert data["empty"] is False
    alphas = {a for a, *_r in data["pass1_vs_alpha"]}
    assert alphas == {0.0, 0.5, 1.0}
    assert set(data["pass_k_by_alpha"]) == {"0.0", "0.5", "1.0"}

    figs = interp_pages.render_interp_weights(data)
    assert figs
    joined = "\n".join(figs)
    assert "http://" not in joined and "https://" not in joined


# ---------------------------------------------------------------------------- probes, non-empty


def test_view_and_render_probes_with_data(tmp_path: Path) -> None:
    store = Store(tmp_path)
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

    data = interp_view.view_probes(store, {})
    assert data["empty"] is False
    assert data["meta"]["layers"] == list(interp_view.LAYER_KEYS)
    assert len(data["next_rule_acc"]["base"]) == 5
    assert len(data["next_rule_acc"]["rl"]) == 5

    figs = interp_pages.render_interp_probes(data)
    assert figs
    joined = "\n".join(figs)
    assert "http://" not in joined and "https://" not in joined


# ------------------------------------------------------------------------------ full dash.build


def test_dash_build_includes_interp_pages_when_empty(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    out_dir = tmp_path / "dash_out"
    written = dash_build.build(store, out_dir)
    names = {p.name for p in written}
    for page in (
        "interp_likelihood.html",
        "interp_steering.html",
        "interp_weights.html",
        "interp_probes.html",
    ):
        assert page in names
        content = (out_dir / page).read_text()
        assert "no data yet" in content
        assert "http://" not in content and "https://" not in content


def test_dash_build_includes_interp_pages_alongside_full_demo_store(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    build_full_demo_store(store)  # WP6's own full 8-benchmark synthetic store
    # add a tiny bit of interp data too, so these pages render real content in the same build
    run_id = new_interp_run(store, "likelihood", {})
    write_interp_rows(
        store, "likelihood", run_id, [_likelihood_row("r1", "reserve", -5.0, -4.0, 0.9)]
    )

    out_dir = tmp_path / "dash_out"
    written = dash_build.build(store, out_dir)
    names = {p.name for p in written}
    assert len(names) == len(written)  # no accidental duplicate filenames
    assert {
        "interp_likelihood.html",
        "interp_steering.html",
        "interp_weights.html",
        "interp_probes.html",
    } <= names
    likelihood_html = (out_dir / "interp_likelihood.html").read_text()
    assert "no data yet" not in likelihood_html  # this one has real rows
