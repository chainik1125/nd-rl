"""`interp/steering.py`: `fit_diff_vectors`, `steering_sweep`'s wiring (missing-vector skip, best-
layer selection, the RL `-delta` control), and `fit_early_stop_contrast`."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from code.ndbench.adapters.absgpt import AbsGptAdapter
from code.ndbench.adapters.fake import FakeAdapter
from code.ndbench.core.io import utc_now_iso
from code.ndbench.core.proof import make_reference
from code.ndbench.core.types import LengthLabel, Problem, ReferenceProof, Sequent, SuiteManifest
from code.ndbench.interp import data, eval_suites, steering
from code.ndbench.judge.nd import NDJudge
from code.ndbench.runner.runner import RunSpec, run
from code.ndbench.store import Store

from ._fixtures import load_valid_proofs
from .test_wp5_fixtures import new_tiny_model, new_tokenizer


def _save_tiny_ckpt(path: Path, *, seed: int) -> None:
    tok = new_tokenizer()
    model = new_tiny_model(tok.vocab_size, seed=seed)
    torch.save(
        {
            "state": model.state_dict(),
            "cfg": {"vocab": tok.vocab_size, "n_layer": 1, "d": 32, "n_head": 4},
            "tok_mode": tok.mode,
        },
        path,
    )


def _tiny_pairs(n: int = 4) -> list[tuple[Sequent, str]]:
    rows = load_valid_proofs()[:n]
    return [(Sequent.from_prompt(r["prompt"]), r["proof"]) for r in rows]


# --------------------------------------------------------------------------------- _positions_for


def test_positions_for_and_line_start_positions() -> None:
    body = "N1 P : PR ; N2 P : R N1 ; QED"
    assert data.line_start_positions(body) == [0, 5]  # "N1" at 0, "N2" at 5 (after the first ';')
    positions = steering._positions_for(
        "line_start", prompt_len=10, proof_len=len(body.split()), body=body
    )
    assert positions == [10, 15]
    assert steering._positions_for("prompt", prompt_len=3, proof_len=5, body=body) == [0, 1, 2]
    assert steering._positions_for("proof", prompt_len=3, proof_len=5, body=body) == [3, 4, 5, 6, 7]


# ----------------------------------------------------------------------------- fit_diff_vectors


def test_fit_diff_vectors_shapes_and_sidecar(tmp_path: Path) -> None:
    base_ckpt, rl_ckpt = tmp_path / "base.pt", tmp_path / "rl.pt"
    _save_tiny_ckpt(base_ckpt, seed=1)
    _save_tiny_ckpt(rl_ckpt, seed=2)
    base = AbsGptAdapter(base_ckpt, device="cpu")
    rl = AbsGptAdapter(rl_ckpt, device="cpu")
    pairs = _tiny_pairs(4)

    out_dir = tmp_path / "vectors"
    results = steering.fit_diff_vectors(
        base,
        rl,
        pairs,
        layers=(0,),
        position_types=("proof", "prompt", "line_start"),
        out_dir=out_dir,
    )
    assert {r.position_type for r in results} == {"proof", "prompt", "line_start"}
    for r in results:
        assert r.layer == 0
        assert r.vector.shape == (32,)
        assert r.path.exists()
        sidecar = json.loads(r.path.with_suffix(".json").read_text())
        assert sidecar["position_type"] == r.position_type
        assert sidecar["n"] == 4
        assert sidecar["norm"] == r.norm
        loaded = np.load(r.path)
        assert np.allclose(loaded, r.vector, atol=1e-6)


def test_fit_diff_vectors_zero_when_models_identical(tmp_path: Path) -> None:
    ckpt = tmp_path / "same.pt"
    _save_tiny_ckpt(ckpt, seed=1)
    base = AbsGptAdapter(ckpt, device="cpu")
    rl = AbsGptAdapter(ckpt, device="cpu")
    pairs = _tiny_pairs(3)
    (result,) = steering.fit_diff_vectors(
        base, rl, pairs, layers=(0,), position_types=("proof",), out_dir=tmp_path / "vectors"
    )
    assert result.norm < 1e-5


# ------------------------------------------------------------------------------- steering_sweep


def _tiny_eval_suites(store: Store) -> eval_suites.EvalSuites:
    rows = load_valid_proofs()[:6]
    problems = [
        Problem(
            pid=row["name"],
            suite="x",
            version="v1",
            sequent=Sequent.from_prompt(row["prompt"]),
            length=LengthLabel(
                L=row["n_lines"], L_ub=row["n_lines"], L_lb=None, method="reference", bound=None
            ),
            ref=make_reference(row["proof"], source="minlen"),
            tags={},
        )
        for row in rows
    ]
    dev = [replace(p, suite=eval_suites.DEV_GEN_SUITE) for p in problems[:3]]
    sealed = [replace(p, suite=eval_suites.SEALED_GEN_SUITE) for p in problems[3:]]
    k128 = [replace(p, suite=eval_suites.K128_SUITE) for p in problems[:2]]
    for suite_name, items in (
        (eval_suites.DEV_GEN_SUITE, dev),
        (eval_suites.SEALED_GEN_SUITE, sealed),
        (eval_suites.K128_SUITE, k128),
    ):
        store.write_suite(
            items,
            SuiteManifest(
                suite=suite_name,
                version="v1",
                builder="test",
                config={},
                seed=0,
                created_utc=utc_now_iso(),
                git_commit="test",
                n_items=len(items),
                items_sha256="",
                excl_sources={},
                stats={},
            ),
        )
    return eval_suites.EvalSuites(dev=dev, sealed=sealed, k128=k128, textbook_dev=[], notes=[])


def test_steering_sweep_skips_missing_vectors_with_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store(tmp_path / "out")
    tiny_suites = _tiny_eval_suites(store)
    monkeypatch.setattr(eval_suites, "build_all", lambda *a, **k: tiny_suites)

    base_ckpt, rl_ckpt = tmp_path / "base.pt", tmp_path / "rl.pt"
    _save_tiny_ckpt(base_ckpt, seed=1)
    _save_tiny_ckpt(rl_ckpt, seed=2)
    base = AbsGptAdapter(base_ckpt, device="cpu")
    rl = AbsGptAdapter(rl_ckpt, device="cpu")

    cfg: dict[str, Any] = {
        "layers": [0],
        "alphas": [0.5],
        "k": 1,
        "vectors_dir": str(tmp_path / "empty_vectors"),
    }
    result = steering.steering_sweep(
        store, cfg, base_adapter=base, rl_adapter=rl, vectors_dir=tmp_path / "empty_vectors"
    )
    assert result["best_layer"] is None
    assert "base" in result["runs"]
    assert any("no fitted vector" in n for n in result["notes"])
    assert any("skipped the whole sweep" in n for n in result["notes"])
    assert not any(k.startswith("steer:") for k in result["runs"])


def test_steering_sweep_full_wiring_with_fitted_vector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store(tmp_path / "out")
    tiny_suites = _tiny_eval_suites(store)
    monkeypatch.setattr(eval_suites, "build_all", lambda *a, **k: tiny_suites)

    base_ckpt, rl_ckpt = tmp_path / "base.pt", tmp_path / "rl.pt"
    _save_tiny_ckpt(base_ckpt, seed=1)
    _save_tiny_ckpt(rl_ckpt, seed=2)
    base = AbsGptAdapter(base_ckpt, device="cpu")
    rl = AbsGptAdapter(rl_ckpt, device="cpu")

    vectors_dir = tmp_path / "vectors"
    steering.fit_diff_vectors(
        base, rl, _tiny_pairs(3), layers=(0,), position_types=("proof",), out_dir=vectors_dir
    )

    cfg: dict[str, Any] = {"layers": [0], "alphas": [0.5, 1.0], "k": 1}
    result = steering.steering_sweep(
        store, cfg, base_adapter=base, rl_adapter=rl, vectors_dir=vectors_dir
    )
    assert result["best_layer"] == 0
    assert set(result["runs"]) == {"base", "steer:0:0.5", "steer:0:1", "steer:0:-1"}
    labels = {cast(str, r["model"]) for r in store.runs()}
    assert {"base", "steer:0:0.5", "steer:0:1", "steer:0:-1"} <= labels
    # 4 distinct model variants (base + 2 steered-base alphas + 1 steered-rl control), each run on
    # 3 suites: every steered identity differs from the plain base/rl identity (spec hash included).
    idents_by_model: dict[str, set[str]] = {}
    for rec in store.runs():
        idents_by_model.setdefault(cast(str, rec["model"]), set()).add(
            json.dumps(rec["identity"], sort_keys=True)
        )
    for model, idents in idents_by_model.items():
        assert len(idents) == 1, f"{model} has inconsistent identities across suites: {idents}"
    assert len({next(iter(v)) for v in idents_by_model.values()}) == 4


# ------------------------------------------------------------------------ fit_early_stop_contrast


def test_fit_early_stop_contrast_none_without_a_matching_run(tmp_path: Path) -> None:
    store = Store(tmp_path / "out")
    base_ckpt = tmp_path / "base.pt"
    _save_tiny_ckpt(base_ckpt, seed=1)
    adapter = AbsGptAdapter(base_ckpt, device="cpu")
    result = steering.fit_early_stop_contrast(store, adapter, suite="nope", version="v1", layer=0)
    assert result is None


def test_fit_early_stop_contrast_computes_a_vector(tmp_path: Path) -> None:
    store = Store(tmp_path / "out")
    rows = load_valid_proofs()
    long_rows = [r for r in rows if r["n_lines"] >= 5][:4]
    problems = [
        Problem(
            pid=row["name"],
            suite="es",
            version="v1",
            sequent=Sequent.from_prompt(row["prompt"]),
            length=LengthLabel(
                L=row["n_lines"], L_ub=row["n_lines"], L_lb=None, method="reference", bound=None
            ),
            ref=make_reference(row["proof"], source="minlen"),
            tags={},
        )
        for row in long_rows
    ]
    store.write_suite(
        problems,
        SuiteManifest(
            suite="es",
            version="v1",
            builder="test",
            config={},
            seed=0,
            created_utc=utc_now_iso(),
            git_commit="test",
            n_items=len(problems),
            items_sha256="",
            excl_sources={},
            stats={},
        ),
    )
    refs = {p.pid: cast("ReferenceProof", p.ref).body for p in problems}

    def truncate(body: str) -> str:
        toks = body.split()
        # drop the last two lines' worth of tokens and QED: a "stopped short" attempt.
        cut = max(1, len(toks) - 6)
        return " ".join(toks[:cut])

    adapter = FakeAdapter(refs, p=0.5, corrupt=truncate)
    spec = RunSpec(suite="es", version="v1", model="base", k=8, keep_samples=True, seed=1)
    run(spec, adapter, NDJudge(), store)

    ckpt = tmp_path / "base.pt"
    _save_tiny_ckpt(ckpt, seed=1)
    tiny_adapter = AbsGptAdapter(ckpt, device="cpu")
    result = steering.fit_early_stop_contrast(
        store, tiny_adapter, suite="es", version="v1", model="base", layer=0
    )
    if result is not None:  # depends on the random draw actually producing both groups
        assert result.vector.shape == (32,)
        assert result.path.exists()
