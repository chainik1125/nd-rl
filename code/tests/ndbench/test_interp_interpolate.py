"""`interp/interpolate.py` (weight interpolation), `interp/eval_suites.py` (the steering/
interpolation eval suites), and `interp/runner_support.py` (the shared ordinary-`Store`-run
plumbing both `interpolate.py` and `steering.py` use)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from code.ndbench.core.types import Problem
from code.ndbench.interp import eval_suites, interpolate
from code.ndbench.interp.runner_support import run_eval_suites
from code.ndbench.store import Store
from code.ndbench.vendor import load_absgpt_module

from .test_wp5_fixtures import TinyGptAdapter, new_tiny_model, new_tokenizer

_model_mod = load_absgpt_module("model")


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


# ------------------------------------------------------------------------------- interpolate()


def test_interpolate_endpoints_and_midpoint(tmp_path: Path) -> None:
    base_ckpt = tmp_path / "base.pt"
    rl_ckpt = tmp_path / "rl.pt"
    _save_tiny_ckpt(base_ckpt, seed=1)
    _save_tiny_ckpt(rl_ckpt, seed=2)
    base_state = torch.load(base_ckpt, weights_only=True)["state"]
    rl_state = torch.load(rl_ckpt, weights_only=True)["state"]
    assert any(not torch.equal(base_state[k], rl_state[k]) for k in base_state)  # sanity: differ

    out_dir = tmp_path / "ckpts"
    p0 = interpolate.interpolate(base_ckpt, rl_ckpt, 0.0, out_dir)
    p1 = interpolate.interpolate(base_ckpt, rl_ckpt, 1.0, out_dir)
    p_half = interpolate.interpolate(base_ckpt, rl_ckpt, 0.5, out_dir)

    s0 = torch.load(p0, weights_only=True)["state"]
    s1 = torch.load(p1, weights_only=True)["state"]
    s_half = torch.load(p_half, weights_only=True)["state"]
    for k in base_state:
        assert torch.allclose(s0[k], base_state[k])
        assert torch.allclose(s1[k], rl_state[k])
        assert torch.allclose(s_half[k], (base_state[k].float() + rl_state[k].float()) / 2)

    sidecar = json.loads(p_half.with_suffix(".pt.sha256.json").read_text())
    assert sidecar["alpha"] == 0.5
    assert sidecar["base_ckpt"] == str(base_ckpt)
    assert sidecar["rl_ckpt"] == str(rl_ckpt)
    assert sidecar["out_sha256"]


def test_interpolate_rejects_shape_mismatch(tmp_path: Path) -> None:
    tok = new_tokenizer()
    base_model = new_tiny_model(tok.vocab_size, seed=1, d=32)
    rl_model = new_tiny_model(tok.vocab_size, seed=2, d=32)
    # corrupt one tensor's shape in the "rl" checkpoint to trigger the mismatch check.
    rl_state = dict(rl_model.state_dict())
    key = next(iter(rl_state))
    rl_state[key] = rl_state[key][:1]
    base_ckpt = tmp_path / "base.pt"
    rl_ckpt = tmp_path / "rl.pt"
    torch.save({"state": base_model.state_dict(), "cfg": {}, "tok_mode": "abs"}, base_ckpt)
    torch.save({"state": rl_state, "cfg": {}, "tok_mode": "abs"}, rl_ckpt)
    try:
        interpolate.interpolate(base_ckpt, rl_ckpt, 0.5, tmp_path / "out")
    except ValueError as e:
        assert "shape mismatch" in str(e)
    else:
        raise AssertionError("expected a ValueError")


def test_build_checkpoints_materialises_only_nontrivial_alphas_and_is_idempotent(
    tmp_path: Path,
) -> None:
    base_ckpt = tmp_path / "base.pt"
    rl_ckpt = tmp_path / "rl.pt"
    _save_tiny_ckpt(base_ckpt, seed=1)
    _save_tiny_ckpt(rl_ckpt, seed=2)
    out_dir = tmp_path / "ckpts"
    result = interpolate.build_checkpoints(out_dir, base_ckpt=base_ckpt, rl_ckpt=rl_ckpt)
    assert set(result) == set(interpolate.CKPT_ALPHAS) == {0.25, 0.5, 0.75, 1.25, 1.5}
    for p in result.values():
        assert p.exists()
        assert p.with_suffix(p.suffix + ".sha256.json").exists()

    mtimes = {a: p.stat().st_mtime_ns for a, p in result.items()}
    result2 = interpolate.build_checkpoints(out_dir, base_ckpt=base_ckpt, rl_ckpt=rl_ckpt)
    assert {a: p.stat().st_mtime_ns for a, p in result2.items()} == mtimes  # untouched, not rebuilt


def test_absgpt_adapter_loads_interpolated_checkpoint_and_identity_differs(tmp_path: Path) -> None:
    """The real production adapter (not a test double), on CPU, with fabricated tiny checkpoints --
    confirms `AbsGptAdapter` really does load an interpolated checkpoint unchanged."""
    from code.ndbench.adapters.absgpt import AbsGptAdapter

    base_ckpt = tmp_path / "base.pt"
    rl_ckpt = tmp_path / "rl.pt"
    _save_tiny_ckpt(base_ckpt, seed=1)
    _save_tiny_ckpt(rl_ckpt, seed=2)
    out_dir = tmp_path / "ckpts"
    half_path = interpolate.interpolate(base_ckpt, rl_ckpt, 0.5, out_dir)

    base_adapter = AbsGptAdapter(base_ckpt, device="cpu")
    half_adapter = AbsGptAdapter(half_path, device="cpu")
    assert base_adapter.identity()["ckpt_sha256"] != half_adapter.identity()["ckpt_sha256"]

    from code.ndbench.core.types import Sequent

    seq = Sequent(premises=("P",), conclusion="P")
    lp = half_adapter.logprob([(seq, "N1 P : PR ; N2 P : R N1 ; QED")])[0]
    assert isinstance(lp.nats, float)


# ------------------------------------------------------------------------------- eval_suites


def test_build_dev_and_sealed_gen_suites_match_known_counts(tmp_path: Path) -> None:
    store = Store(tmp_path / "out")
    dev = eval_suites.build_dev_gen(store)
    sealed = eval_suites.build_sealed_gen(store)
    assert len(dev) == 720
    assert len(sealed) == 1240
    assert all(p.suite == eval_suites.DEV_GEN_SUITE for p in dev)
    assert all(p.suite == eval_suites.SEALED_GEN_SUITE for p in sealed)
    # round-trips through the Store
    assert len(store.load_suite(eval_suites.DEV_GEN_SUITE, "v1")) == 720


def test_k128_subset_is_stratified_and_fixed_size(tmp_path: Path) -> None:
    store = Store(tmp_path / "out")
    dev = eval_suites.build_dev_gen(store)
    sealed = eval_suites.build_sealed_gen(store)
    k128 = eval_suites.build_k128_subset(dev, sealed, store, seed=0)
    assert len(k128) == eval_suites.K128_N_ITEMS == 150
    ls = {p.length.L for p in k128}
    source_ls = {p.length.L for p in (*dev, *sealed)}
    assert ls <= source_ls
    assert len(ls) > 1  # actually stratified across more than one L value
    # deterministic given the seed
    k128_again = eval_suites.build_k128_subset(dev, sealed, store, seed=0)
    assert [p.pid for p in k128] == [p.pid for p in k128_again]


def test_load_textbook_dev_skips_with_note_when_absent(tmp_path: Path) -> None:
    store = Store(tmp_path / "out")
    assert eval_suites.load_textbook_dev(store) == []
    suites = eval_suites.EvalSuites(dev=[], sealed=[], k128=[], textbook_dev=[], notes=[])
    assert suites.textbook_dev == []


# --------------------------------------------------------------------------- run_eval_suites


def test_run_eval_suites_writes_ordinary_store_runs(tmp_path: Path) -> None:
    store = Store(tmp_path / "out")
    dev = eval_suites.build_dev_gen(store)[:3]
    sealed = eval_suites.build_sealed_gen(store)[:2]
    k128 = eval_suites.build_k128_subset(dev, sealed, store, n_items=2, seed=0)
    # rewrite the (tiny) suites under the store so `run` can `load_suite` exactly these pids
    manifest_cfg = {}
    from code.ndbench.core.io import utc_now_iso
    from code.ndbench.core.types import SuiteManifest

    def _rewrite(suite: str, problems: list[Problem]) -> None:
        store.write_suite(
            problems,
            SuiteManifest(
                suite=suite,
                version="v1",
                builder="test",
                config=manifest_cfg,
                seed=0,
                created_utc=utc_now_iso(),
                git_commit="test",
                n_items=len(problems),
                items_sha256="",
                excl_sources={},
                stats={},
            ),
        )

    _rewrite(eval_suites.DEV_GEN_SUITE, dev)
    _rewrite(eval_suites.SEALED_GEN_SUITE, sealed)
    _rewrite(eval_suites.K128_SUITE, k128)
    suites = eval_suites.EvalSuites(dev=dev, sealed=sealed, k128=k128, textbook_dev=[], notes=[])

    adapter = TinyGptAdapter(seed=7)
    result = run_eval_suites(
        store, adapter, model_label="interp:0.5", suites=suites, k_pass1=1, k_pass_k=1
    )
    assert set(result) == {
        eval_suites.DEV_GEN_SUITE,
        eval_suites.SEALED_GEN_SUITE,
        eval_suites.K128_SUITE,
    }
    for suite_name in result:
        run_id = cast(str, cast("dict[str, Any]", result[suite_name])["run_id"])
        assert store.items(run_id)  # rows were actually written
        rec = store.runs(run_id=run_id)[0]
        assert rec["model"] == "interp:0.5"


# ---------------------------------------------------------------------------- run_interpolation


def test_run_interpolation_orchestrates_alpha_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store(tmp_path / "out")
    dev = eval_suites.build_dev_gen(store)[:2]
    sealed = eval_suites.build_sealed_gen(store)[:2]
    k128 = eval_suites.build_k128_subset(dev, sealed, store, n_items=2, seed=0)
    tiny_suites = eval_suites.EvalSuites(
        dev=dev, sealed=sealed, k128=k128, textbook_dev=[], notes=["stub"]
    )

    from code.ndbench.core.io import utc_now_iso
    from code.ndbench.core.types import SuiteManifest

    for suite_name, problems in (
        (eval_suites.DEV_GEN_SUITE, dev),
        (eval_suites.SEALED_GEN_SUITE, sealed),
        (eval_suites.K128_SUITE, k128),
    ):
        store.write_suite(
            problems,
            SuiteManifest(
                suite=suite_name,
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

    monkeypatch.setattr(eval_suites, "build_all", lambda *a, **k: tiny_suites)

    base_ckpt = tmp_path / "base.pt"
    rl_ckpt = tmp_path / "rl.pt"
    _save_tiny_ckpt(base_ckpt, seed=1)
    _save_tiny_ckpt(rl_ckpt, seed=2)

    def adapter_for(alpha: float, ckpt_path: Path | None) -> TinyGptAdapter:
        del ckpt_path
        return TinyGptAdapter(
            seed=int(alpha * 100) + 1, label=interpolate.interp_model_label(alpha)
        )

    cfg: dict[str, Any] = {
        "alphas": [0.0, 0.5, 1.0],
        "ckpt_dir": str(tmp_path / "ckpts"),
        "k_pass1": 1,
        "k_pass_k": 1,
    }
    # patch the module-level checkpoint paths so build_checkpoints (still called for the full
    # CKPT_ALPHAS set) uses our tiny fabricated checkpoints instead of the real 13MB ones.
    monkeypatch.setattr(interpolate.data, "BASE_CKPT", base_ckpt)
    monkeypatch.setattr(interpolate.data, "RL_CKPT", rl_ckpt)

    summary = interpolate.run_interpolation(store, cfg, adapter_for=adapter_for)
    assert summary["alphas"] == ["0", "0.5", "1"]
    assert summary["notes"] == ["stub"]
    assert set(summary["runs"]) == {"0", "0.5", "1"}
    for tag in ("0", "0.5", "1"):
        assert set(summary["runs"][tag]) == {
            eval_suites.DEV_GEN_SUITE,
            eval_suites.SEALED_GEN_SUITE,
            eval_suites.K128_SUITE,
        }

    recs = {cast(str, r["model"]) for r in store.runs()}
    assert recs == {"base", "interp:0.5", "rl"}
