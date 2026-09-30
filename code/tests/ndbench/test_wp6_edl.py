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
from code.ndbench.core.types import LengthLabel, Problem, ReferenceProof, Sequent
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


def test_edl_empty_store(tmp_path: Path) -> None:
    store = Store(tmp_path)
    data = edl.view(store, {})
    assert data["empty"] is True
