"""`analysis/edl.py` acceptance tests (WP8 integration fix list item 5).

WP6 originally wrote `analysis/edl.py` against a guessed row schema (WP5's `edl.py` had not landed
yet); this file now builds rows the way the real pipeline does -- `code.ndbench.edl.run_edl` on the
CPU test doubles from `test_wp5_fixtures.py` -- and feeds them through `Store.append_edl` exactly
as `cli.py`'s `edl` subcommand does, so `analysis.edl.view` is exercised against the actual schema
rather than a hand-written stand-in for it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from code.ndbench.analysis import edl
from code.ndbench.core.types import JSON, LengthLabel, Problem, ReferenceProof, Sequent
from code.ndbench.edl import run_edl
from code.ndbench.store import Store

from ._fixtures import load_valid_proofs
from .test_wp5_fixtures import ConstantNllAdapter, TinyGptAdapter


def _toy_problems(n_train: int, n_test: int, *, tag_class: str = "toy") -> list[Problem]:
    """The same small, real-vocabulary EDL class `test_wp5_edl.py` builds from WP0's shared
    60-proof fixture (not a real family/law suite; just enough certified, reference-bearing
    theorems to exercise `run_edl` end to end)."""
    proofs = load_valid_proofs()
    assert n_train + n_test <= len(proofs)
    problems: list[Problem] = []
    for i, proof in enumerate(proofs[: n_train + n_test]):
        split = "train" if i < n_train else "test"
        seq = Sequent.from_prompt(proof["prompt"])
        ref = ReferenceProof(
            body=proof["proof"], lines=proof["n_lines"], pruned=proof["n_lines"], source="minlen"
        )
        problems.append(
            Problem(
                pid=f"toy:{proof['name']}",
                suite="toy",
                version="v1",
                sequent=seq,
                length=LengthLabel(
                    L=proof["n_lines"], L_ub=None, L_lb=None, method="minlen_exact", bound=None
                ),
                ref=ref,
                tags={"class": tag_class, "split": split},
            )
        )
    return problems


def test_edl_view_on_real_run_edl_rows_from_a_constant_nll_learner(tmp_path: Path) -> None:
    """Plan §B7's null control (`ConstantNllAdapter`, plan Proposition 5.1) gives `EDL == 0`
    exactly and deterministically, so this is a precise check that `view()` parses the real
    `run_edl` row schema correctly (not just "doesn't crash")."""
    store = Store(tmp_path)
    problems = _toy_problems(n_train=8, n_test=4)
    adapter = ConstantNllAdapter(c=3.7, label="const_model")
    rows = run_edl(adapter, problems, lrs=(1e-4, 1e-3), orders=2, n=8, m=4, ckpts=(4, 8))
    store.append_edl("edl-const-8-4", rows)

    data = edl.view(store, {"checkpoints": [4, 8]})
    assert data["empty"] is False
    d = data["classes"]["toy"]["const_model"]
    assert d["n_rows"] == len(rows)
    assert d["n_lrs"] == 2
    # exactly 0 mathematically; floating-point summation of repeated 3.7s only gets us to within a
    # few ULPs of that (same rationale as `test_wp5_edl.py`'s equivalent `run_edl`-level check).
    assert d["edl_by_lr"]["0.0001"] == pytest.approx(0.0, abs=1e-9)
    assert d["edl_by_lr"]["0.001"] == pytest.approx(0.0, abs=1e-9)
    assert d["edl_star"] == pytest.approx(math.log(2), abs=1e-9)  # min over lr (~0) + ln(#lrs)
    assert d["l_test_theta0"] == 3.7
    assert d["l_test_thetan"] == 3.7
    curve = dict(d["edl_curve_by_lr"][d["best_lr"]])
    assert curve[4] == pytest.approx(0.0, abs=1e-9)
    assert curve[8] == pytest.approx(0.0, abs=1e-9)
    # the constant learner's nll never moves, so the CI (no per-example field in real rows) is nan.
    lo, hi = d["edl_star_ci"]
    assert math.isnan(lo) and math.isnan(hi)


def test_edl_view_on_real_run_edl_rows_from_a_trained_tiny_gpt(tmp_path: Path) -> None:
    """A freshly-initialised tiny GPT trained online (plan §3 WP5's own acceptance test, replayed
    through the dashboard's view): `EDL* > 0`, and `l_test_thetan < l_test_theta0` (the model
    actually got better at the held-out theorems, not just cheaper online)."""
    store = Store(tmp_path)
    n_train, n_test = 12, 6
    problems = _toy_problems(n_train=n_train, n_test=n_test)
    adapter = TinyGptAdapter(seed=0)
    rows = run_edl(
        adapter,
        problems,
        lrs=(3e-3,),
        orders=1,
        n=n_train,
        m=n_test,
        ckpts=(4, 8, 12),
        shifts_per_update=8,
        seed=0,
    )
    store.append_edl("edl-tinygpt-12-6", rows)

    data = edl.view(store, {"checkpoints": [4, 8, 12]})
    assert data["empty"] is False
    d = data["classes"]["toy"]["tiny_gpt"]  # TinyGptAdapter's default identity["label"]
    assert d["n_rows"] == 1
    assert d["n_lrs"] == 1
    assert d["edl_by_lr"]["0.003"] > 0.0
    assert d["edl_star"] == d["edl_by_lr"]["0.003"]  # ln(1) == 0, one lr swept
    assert d["l_test_thetan"] < d["l_test_theta0"]


def test_edl_view_deterministic_under_seed(tmp_path: Path) -> None:
    store = Store(tmp_path)
    problems = _toy_problems(n_train=8, n_test=4)
    rows = run_edl(ConstantNllAdapter(c=1.0, label="m"), problems, lrs=(1e-4,), orders=1, n=8, m=4)
    store.append_edl("edl-det", rows)
    a = edl.view(store, {"seed": 5})
    b = edl.view(store, {"seed": 5})
    # `json.dumps` (rather than `a == b` directly) sidesteps `nan != nan`: this class's CI is
    # `(nan, nan)` (no per-example nats in a real `run_edl` row, see `analysis/edl.py`'s module
    # docstring), and two structurally-identical `nan`s are still unequal under plain `==`.
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_edl_labels_a_production_style_run_by_its_run_id_not_the_raw_ckpt_path(
    tmp_path: Path,
) -> None:
    """WP8 integration fix list item: the real `AbsGptAdapter.identity()` has no `"label"` key, only
    `"ckpt"` (a full checkpoint path) -- before this fix, the EDL page displayed
    `/scratch/.../pretrain.pt` instead of `"base"`/`"rl"`. `cli.py`'s `cmd_edl` always names the run
    `f"edl-{store_label}-{n}-{m}"` (`store_label` is exactly the resolved `--model` label), so
    `view` must recover `"base"` from the run id instead of guessing from the checkpoint path."""
    store = Store(tmp_path)
    problems = _toy_problems(n_train=8, n_test=4)
    # No "label" key, mirroring the production AbsGptAdapter identity shape exactly.
    adapter = ConstantNllAdapter(c=1.0, label="")
    adapter_identity = adapter.identity()
    adapter_identity.pop("label", None)
    adapter_identity["ckpt"] = "/scratch/cowifornia/nd-rl/ab2/frozen/pretrain.pt"
    rows = run_edl(adapter, problems, lrs=(1e-4,), orders=1, n=8, m=4)
    for row in rows:
        row["model_identity"] = adapter_identity
    store.append_edl("edl-base-8-4", rows)

    data = edl.view(store, {})
    assert data["empty"] is False
    assert "base" in data["classes"]["toy"]
    assert "/scratch/cowifornia/nd-rl/ab2/frozen/pretrain.pt" not in data["classes"]["toy"]


def test_edl_rerun_ids_are_labelled_and_supersede_the_original_run(tmp_path: Path) -> None:
    """A rerun written with an explicit `--run-id edl2-base-8-4` must be labelled `"base"` (not
    the raw checkpoint path), and the default view must read it *instead of* the superseded
    `edl-base-8-4`, not pool both. Regression: the real store's EDL page showed four models per
    class (`base`/`rl` from the superseded 2-class run plus two raw `.../pretrain.pt`-style labels
    from the current `edl2-*` run)."""
    store = Store(tmp_path)
    problems = _toy_problems(n_train=8, n_test=4)
    adapter = ConstantNllAdapter(c=1.0, label="")
    identity = adapter.identity()
    identity.pop("label", None)
    identity["ckpt"] = "/scratch/cowifornia/nd-rl/ab2/frozen/pretrain.pt"
    old_rows = run_edl(adapter, problems, lrs=(1e-4,), orders=1, n=8, m=4)
    new_rows = run_edl(adapter, problems, lrs=(1e-4,), orders=2, n=8, m=4)
    for row in old_rows:
        row["model_identity"] = identity
        row["EDL"] = 100.0  # recognisably stale value
    for row in new_rows:
        row["model_identity"] = identity
    store.append_edl("edl-base-8-4", old_rows)
    store.append_edl("edl2-base-8-4", new_rows)
    store.append_edl("edl-det", new_rows[:1])  # outside the naming convention: always kept

    assert edl._label_from_run_id("edl2-base-64-64") == "base"
    assert edl._label_from_run_id("edl-arm-C-64-64") == "arm-C"
    assert edl._label_from_run_id("edl-det") is None

    data = edl.view(store, {})
    assert data["meta"]["run_ids"] == ["edl-det", "edl2-base-8-4"]
    toy = data["classes"]["toy"]
    assert "/scratch/cowifornia/nd-rl/ab2/frozen/pretrain.pt" in toy  # the unlabelled edl-det row
    assert toy["base"]["n_rows"] == len(new_rows)
    assert toy["base"]["edl_by_lr"]["0.0001"] == pytest.approx(0.0, abs=1e-9)
    assert set(toy["base"]["edl_by_order_at_best_lr"]) == {"0", "1"}

    # An explicit run_ids list is still honoured verbatim, superseded runs included.
    pinned = edl.view(store, {"run_ids": ["edl-base-8-4"]})
    assert pinned["classes"]["toy"]["base"]["edl_by_lr"]["0.0001"] == 100.0


def test_edl_empty_store(tmp_path: Path) -> None:
    store = Store(tmp_path)
    data = edl.view(store, {})
    assert data["empty"] is True


def test_render_edl_drops_nan_points_from_the_edl_curve() -> None:
    """`edl.py`'s module docstring: `edl_curve_by_lr` emits one point per `cfg["checkpoints"]`
    entry even when a row never reached that checkpoint (`nan` there) -- real data hits this
    whenever a checkpoint exceeds the run's own `n` (e.g. the default grid's 64 with n=32).
    `render_edl` must drop those points rather than hand `line_chart` a `nan` y-value, which would
    render as a literal "nan" coordinate in the SVG path (a broken/truncated line segment)."""
    from code.ndbench.dash import pages

    data: dict[str, JSON] = {
        "empty": False,
        "classes": {
            "toy": {
                "base": {
                    "edl_star": 1.0,
                    "edl_star_ci": [float("nan"), float("nan")],
                    "edl_curve_by_lr": {
                        "0.001": [[8, 0.5], [16, 0.8], [32, 1.0], [64, float("nan")]]
                    },
                    "best_lr": "0.001",
                    "l_test_theta0": 2.0,
                    "l_test_thetan": 1.5,
                    "pass8_theta0": 0.1,
                    "pass8_thetan": 0.3,
                }
            }
        },
    }
    html = "".join(pages.render_edl(data))
    assert "nan" not in html.lower()
