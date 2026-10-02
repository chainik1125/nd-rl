"""`interp/danstate_support.py` + the CLI/config generalisation (`DANSTATE_PLAN.md` §7 WP-D6):
likelihood (I1, ordering dedup by canonical action sequence + NaN filtering), steering (I2,
Dan-specific vector fitting + the generalised `steering.steering_sweep`), interpolation (I3, the
*unmodified* `interpolate.interpolate` on Dan-shaped checkpoints), probes (I4, the action-step
dataset + classifier), and a model-label-resolution sanity check -- all CPU, tiny random
`ALiBiGPT` checkpoints, on temp `Store`s. Also re-runs a slice of the pre-existing absgpt interp
tests' own assertions directly (not by import -- see `test_absgpt_path_label_defaults_unchanged`)
to pin the "old path untouched" guarantee this work package's instructions require.
"""

from __future__ import annotations

import math
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
import yaml

from code.ndbench.adapters.base import SteeringTerm
from code.ndbench.adapters.danstate import DanStateAdapter
from code.ndbench.core.io import utc_now_iso
from code.ndbench.core.proof import make_reference
from code.ndbench.core.types import LengthLabel, Problem, Sequent, SuiteManifest
from code.ndbench.interp import danstate_support as dsup
from code.ndbench.interp import eval_suites, interpolate, steering
from code.ndbench.interp._store import load_interp_rows
from code.ndbench.judge.nd import NDJudge
from code.ndbench.store import Store
from code.ndbench.vendor import load_danstate

# ----------------------------------------------------------------------------------- fixtures

_SIMPLE = ("THM ( P & Q ) SEQ P PRF", "N1 ( P & Q ) : PR ; N2 P : ANDE1 N1 ; QED")
_IMP = (
    "THM ( P & Q ) SEQ ( P > Q ) PRF",
    "N1 ( P & Q ) : PR ; N2 | P : AS ; N3 | Q : ANDE2 N1 ; N4 ( P > Q ) : IMPI N2 N3 ; QED",
)
_OR_ELIM = (
    "THM ( P v Q ) SEQ ( Q v P ) PRF",
    "N1 ( P v Q ) : PR ; N2 | P : AS ; N3 | ( Q v P ) : ORI2 N2 ; N4 | Q : AS ; "
    "N5 | ( Q v P ) : ORI1 N4 ; N6 ( Q v P ) : ORE N1 N2 N3 N4 N5 ; QED",
)
#: Two independent, *structurally identical* derived lines (neither cited by anything else): the
#: box/PR ordering model finds 2 orderings (swap them), but both render to the literal same body
#: (same rule, same citation, canonical names assigned by position) -- `canonical_actions` dedup
#: must collapse them to 1 (confirmed directly against `orderings.py`/`state_env.canonicalise`
#: before writing this test).
_DUP = (
    "THM ( P & Q ) SEQ ( P & Q ) PRF",
    "N1 ( P & Q ) : PR ; N2 P : ANDE1 N1 ; N3 P : ANDE1 N1 ; N4 ( P & Q ) : R N1 ; QED",
)
#: N2/N3 open and close a box no IMPI/NEGI/ORE ever cites -- `DanStateAdapter.logprob`'s
#: `allow_reencode=False` default NaNs this out (module docstring's "Likelihood" NaN case).
_UNCITED_BOX = (
    "THM ( P & Q ) SEQ P PRF",
    "N1 ( P & Q ) : PR ; N2 | P : AS ; N3 P : ANDE1 N1 ; QED",
)


def _tiny_cfg() -> dict[str, Any]:
    return dict(vocab=111, n_layer=2, d=32, n_head=4, max_len=64, d_ff=64, arch="best")


def _save_tiny_dan_ckpt(path: Path, *, seed: int) -> None:
    ds = load_danstate()
    torch.manual_seed(seed)
    cfg = _tiny_cfg()
    model = ds.best_model.ALiBiGPT(**cfg)
    torch.save(
        {"cfg": model.cfg, "state": model.state_dict(), "tok_mode": "lean_staten", "extra": {}},
        path,
    )


@pytest.fixture
def fz_t1(tmp_path: Path) -> Generator[tuple[DanStateAdapter, DanStateAdapter], None, None]:
    fz_ckpt, t1_ckpt = tmp_path / "fz.pt", tmp_path / "t1.pt"
    _save_tiny_dan_ckpt(fz_ckpt, seed=1)
    _save_tiny_dan_ckpt(t1_ckpt, seed=2)
    fz = DanStateAdapter(
        ckpt=fz_ckpt, device="cpu", batch=8, max_action=16, max_steps=8, name_base_max=4
    )
    t1 = DanStateAdapter(
        ckpt=t1_ckpt, device="cpu", batch=8, max_action=16, max_steps=8, name_base_max=4
    )
    try:
        yield fz, t1
    finally:
        fz.close()
        t1.close()


def _reserve_row(pid: str, prompt: str, body: str, l_true: int) -> dict[str, Any]:
    return {
        "source": "gen",
        "minlen_proof": body,
        "L_true": l_true,
        "name": pid,
        "thm": Sequent.from_prompt(prompt).thm(),
    }


def _write_reserve_fixture(path: Path, rows: list[dict[str, Any]]) -> None:
    import json

    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _sub(row: dict[str, Any], key: str) -> dict[str, Any]:
    """Narrow one `likelihood`-row field (typed `JSON` -- a recursive, un-subscriptable union by
    design) to `dict[str, Any]` for these tests' own follow-up field access."""
    return cast("dict[str, Any]", row[key])


# --------------------------------------------------------------------------------- I1: likelihood


def test_likelihood_dedupes_orderings_and_filters_nan(
    tmp_path: Path, fz_t1: tuple[DanStateAdapter, DanStateAdapter]
) -> None:
    fz, t1 = fz_t1
    store = Store(tmp_path / "out")
    reserve_path = tmp_path / "reserve.jsonl"
    _write_reserve_fixture(
        reserve_path,
        [
            _reserve_row("dup", *_DUP, l_true=4),
            _reserve_row("uncited", *_UNCITED_BOX, l_true=2),
        ],
    )
    cfg = {
        "n_neutral": 2,
        "l_min": 1,
        "l_max": 10,
        "reserve_path": str(reserve_path),
        "family_limit": 0,
        "cap": 1000,
        "m": 32,
        "seed": 0,
    }
    result = dsup.run_likelihood_dan(store, cfg, base_adapter=fz, rl_adapter=t1)
    assert result["n_items"] == 2
    assert result["n_nan_fz"] >= 1
    assert result["n_nan_t1"] >= 1

    rows = {cast(str, r["pid"]): r for r in load_interp_rows(store, "likelihood", result["run_id"])}
    assert set(rows) == {"dup", "uncited"}

    dup_row = rows["dup"]
    assert dup_row["origin"] == "reserve"  # `data.load_neutral_proofs`'s own origin label
    dup_fz, dup_t1 = _sub(dup_row, "fz"), _sub(dup_row, "t1")
    # `n_orderings` is purely structural (box/PR model), so both columns agree exactly; dedup by
    # canonical action sequence collapses the 2 orderings to 1 (module docstring).
    assert dup_fz["n_orderings"] == 2
    assert dup_t1["n_orderings"] == 2
    assert dup_fz["n_unique_actions"] == 1
    assert dup_t1["n_unique_actions"] == 1
    assert dup_fz["n_duplicate_orderings"] == 1
    assert dup_fz["method"] == "single"
    assert not math.isnan(dup_fz["logp_marginal"])
    assert dup_fz["logp_marginal"] == pytest.approx(dup_fz["logp_single"])

    uncited_fz = _sub(rows["uncited"], "fz")
    uncited_t1 = _sub(rows["uncited"], "t1")
    assert math.isnan(uncited_fz["logp_marginal"])
    assert math.isnan(uncited_t1["logp_marginal"])


def test_dedupe_orderings_helper_directly(fz_t1: tuple[DanStateAdapter, DanStateAdapter]) -> None:
    fz, _t1 = fz_t1
    seq = Sequent.from_prompt(_DUP[0])
    bodies, logqs, method, n, n_dup = dsup._dedupe_orderings_by_canonical_actions(
        fz, seq, _DUP[1], cap=1000, m=32, rng=__import__("random").Random(0)
    )
    assert n == 2
    assert len(bodies) == 1
    assert logqs is None
    assert method == "single"
    assert n_dup == 1


def test_marginal_logprob_dan_matches_direct_logprob_when_single_ordering(
    fz_t1: tuple[DanStateAdapter, DanStateAdapter],
) -> None:
    fz, _t1 = fz_t1
    seq = Sequent.from_prompt(_SIMPLE[0])
    result = dsup.marginal_logprob_dan(fz, seq, _SIMPLE[1])
    assert result["method"] == "single"
    (lp,) = fz.logprob([(seq, _SIMPLE[1])])
    assert result["logp_single"] == pytest.approx(lp.nats)
    assert result["logp_marginal"] == pytest.approx(lp.nats)


def test_marginal_logprob_dan_forwards_exclude_defining_names(
    fz_t1: tuple[DanStateAdapter, DanStateAdapter],
) -> None:
    fz, _t1 = fz_t1
    seq = Sequent.from_prompt(_SIMPLE[0])
    result = dsup.marginal_logprob_dan(fz, seq, _SIMPLE[1], exclude_defining_names=True)
    (excl,) = fz.logprob([(seq, _SIMPLE[1])], exclude_defining_names=True)
    (full,) = fz.logprob([(seq, _SIMPLE[1])])
    assert result["logp_marginal"] == pytest.approx(excl.nats)
    assert excl.nats != pytest.approx(full.nats)


# ------------------------------------------------------------------------------------ I2: steering


def test_fit_diff_vectors_dan_zero_delta_reproduces_unsteered_activations(
    tmp_path: Path, fz_t1: tuple[DanStateAdapter, DanStateAdapter]
) -> None:
    fz, _t1 = fz_t1
    pairs = [(Sequent.from_prompt(p), b) for p, b in (_SIMPLE, _IMP)]
    out_dir = tmp_path / "vectors"
    # fz vs fz: delta is exactly zero at every layer.
    results = dsup.fit_diff_vectors_dan(fz, fz, pairs, layers=(0, 1), out_dir=out_dir)
    assert {r.layer for r in results} == {0, 1}
    for r in results:
        assert np.allclose(r.vector, 0.0)
        assert (out_dir / f"diff_l{r.layer}_action.npy").exists()
        assert (out_dir / f"diff_l{r.layer}_action.json").exists()

    vec_path = out_dir / "diff_l0_action.npy"
    term = SteeringTerm(layer=0, vector=str(vec_path), alpha=7.0, where="proof")
    steered = fz.with_steering((term,))
    try:
        seq = Sequent.from_prompt(_SIMPLE[0])
        # `positions="action"` (not the default "act"): the vector was fit on action positions
        # and `where="proof"` only ever touches those -- the `<act>` token itself always
        # *precedes* its own action tokens within a row, so causal attention guarantees
        # `where="proof"` can never move it (not a useful place to look for a steering effect).
        (a1,) = fz.activations([(seq, _SIMPLE[1])], [0], positions="action")
        (a2,) = steered.activations([(seq, _SIMPLE[1])], [0], positions="action")
        assert np.allclose(a1, a2, atol=1e-6)
    finally:
        steered.close()


def test_fit_diff_vectors_dan_nonzero_delta_changes_activations(
    tmp_path: Path, fz_t1: tuple[DanStateAdapter, DanStateAdapter]
) -> None:
    fz, t1 = fz_t1
    pairs = [(Sequent.from_prompt(p), b) for p, b in (_SIMPLE, _IMP, _OR_ELIM)]
    out_dir = tmp_path / "vectors"
    results = dsup.fit_diff_vectors_dan(fz, t1, pairs, layers=(0,), out_dir=out_dir)
    (r0,) = results
    assert r0.n_positions > 0
    assert r0.norm >= 0.0

    vec_path = out_dir / "diff_l0_action.npy"
    term = SteeringTerm(layer=0, vector=str(vec_path), alpha=1.0, where="proof")
    steered = fz.with_steering((term,))
    try:
        seq = Sequent.from_prompt(_SIMPLE[0])
        (a1,) = fz.activations([(seq, _SIMPLE[1])], [0], positions="action")
        (a2,) = steered.activations([(seq, _SIMPLE[1])], [0], positions="action")
        assert not np.allclose(a1, a2, atol=1e-6)
    finally:
        steered.close()


def _tiny_dan_eval_suites(store: Store) -> eval_suites.EvalSuites:
    rows = [(_SIMPLE, "s0"), (_IMP, "s1"), (_OR_ELIM, "s2")]
    problems = [
        Problem(
            pid=pid,
            suite="x",
            version="v1",
            sequent=Sequent.from_prompt(prompt),
            length=LengthLabel(L=None, L_ub=None, L_lb=None, method="reference", bound=None),
            ref=make_reference(body, source="minlen"),
            tags={},
        )
        for (prompt, body), pid in rows
    ]
    dev = [replace(p, suite=eval_suites.DEV_GEN_SUITE) for p in problems[:2]]
    sealed = [replace(p, suite=eval_suites.SEALED_GEN_SUITE) for p in problems[2:]]
    for suite_name, items in (
        (eval_suites.DEV_GEN_SUITE, dev),
        (eval_suites.SEALED_GEN_SUITE, sealed),
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
    return eval_suites.EvalSuites(dev=dev, sealed=sealed, k128=[], textbook_dev=[], notes=[])


def test_steering_sweep_generalised_labels_and_judge_for_dan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fz_t1: tuple[DanStateAdapter, DanStateAdapter]
) -> None:
    """`steering.steering_sweep` end to end on Dan's pair: generalised `base_label`/`rl_label`
    produce `steer:<base_label>:...` (not the bare old `steer:<layer>:<alpha>`, since
    `base_label != "base"`), and `judge=NDJudge()` (a real judge Dan's ND-body `Attempt`s can be
    checked by, no `lean` binary needed) is actually used."""
    fz, t1 = fz_t1
    store = Store(tmp_path / "out")
    tiny_suites = _tiny_dan_eval_suites(store)
    monkeypatch.setattr(eval_suites, "build_all", lambda *a, **k: tiny_suites)

    vectors_dir = tmp_path / "vectors"
    pairs = [(Sequent.from_prompt(p), b) for p, b in (_SIMPLE, _IMP, _OR_ELIM)]
    dsup.fit_diff_vectors_dan(fz, t1, pairs, layers=(0,), out_dir=vectors_dir)

    cfg: dict[str, Any] = {"layers": [0], "alphas": [0.5], "k": 1, "position_type": "action"}
    result = steering.steering_sweep(
        store,
        cfg,
        base_adapter=fz,
        rl_adapter=t1,
        vectors_dir=vectors_dir,
        judge=NDJudge(),
        base_label="dan12-fz-s0",
        rl_label="dan12-t1-s0",
    )
    assert set(result["runs"]) == {
        "dan12-fz-s0",
        "steer:dan12-fz-s0:0:0.5",
        "steer:dan12-t1-s0:0:-1",
    }
    labels = {cast(str, r["model"]) for r in store.runs()}
    assert {"dan12-fz-s0", "steer:dan12-fz-s0:0:0.5", "steer:dan12-t1-s0:0:-1"} <= labels
    for rec in store.runs():
        assert rec["judge_id"] == NDJudge.judge_id


# -------------------------------------------------------------------------------- I3: interpolate


def test_interpolate_at_endpoints_reproduces_fz_and_t1_logits(
    tmp_path: Path, fz_t1: tuple[DanStateAdapter, DanStateAdapter]
) -> None:
    fz, t1 = fz_t1
    fz_ckpt, t1_ckpt = fz._ckpt_path, t1._ckpt_path
    out_dir = tmp_path / "interp_ckpts"

    a0_path = interpolate.interpolate(fz_ckpt, t1_ckpt, 0.0, out_dir)
    a1_path = interpolate.interpolate(fz_ckpt, t1_ckpt, 1.0, out_dir)

    a0 = DanStateAdapter(
        ckpt=a0_path, device="cpu", batch=8, max_action=16, max_steps=8, name_base_max=4
    )
    a1 = DanStateAdapter(
        ckpt=a1_path, device="cpu", batch=8, max_action=16, max_steps=8, name_base_max=4
    )
    try:
        seq = Sequent.from_prompt(_SIMPLE[0])
        (lp_fz,) = fz.logprob([(seq, _SIMPLE[1])])
        (lp_a0,) = a0.logprob([(seq, _SIMPLE[1])])
        assert lp_a0.nats == pytest.approx(lp_fz.nats, abs=1e-5)

        (lp_t1,) = t1.logprob([(seq, _SIMPLE[1])])
        (lp_a1,) = a1.logprob([(seq, _SIMPLE[1])])
        assert lp_a1.nats == pytest.approx(lp_t1.nats, abs=1e-5)

        # the interpolated checkpoint itself still carries Dan's own cfg/tok_mode/extra keys.
        raw = torch.load(str(a0_path), map_location="cpu", weights_only=True)
        assert set(raw) == {"cfg", "state", "tok_mode", "extra"}
        assert raw["tok_mode"] == "lean_staten"
    finally:
        a0.close()
        a1.close()


def test_interp_model_label_dan_family_prefix() -> None:
    assert interpolate.interp_model_label(0.0) == "base"  # old defaults, unchanged
    assert interpolate.interp_model_label(1.0) == "rl"
    assert interpolate.interp_model_label(0.5) == "interp:0.5"
    assert (
        interpolate.interp_model_label(
            0.5, base_label="dan12-fz-s0", rl_label="dan12-t1-s0", prefix="dan12-interp"
        )
        == "dan12-interp:0.5"
    )
    assert (
        interpolate.interp_model_label(
            0.0, base_label="dan12-fz-s0", rl_label="dan12-t1-s0", prefix="dan12-interp"
        )
        == "dan12-fz-s0"
    )


# ------------------------------------------------------------------------------------- I4: probes


def test_classify_steps_matches_known_rule_sequences() -> None:
    ds = load_danstate()
    expected = {
        _SIMPLE[1]: ["PR", "ANDE1", "FINAL"],
        _IMP[1]: ["PR", "AS", "ANDE2", "IMPI", "FINAL"],
        _OR_ELIM[1]: ["PR", "AS", "ORI2", "AS", "ORI1", "ORE", "FINAL"],
    }
    for (prompt, body), rules in (
        (_SIMPLE, expected[_SIMPLE[1]]),
        (_IMP, expected[_IMP[1]]),
        (_OR_ELIM, expected[_OR_ELIM[1]]),
    ):
        meta = dsup.classify_steps(ds, prompt, body)
        assert [m["rule"] for m in meta] == rules
        assert meta[-1]["creates_line"] is False
        assert all(m["creates_line"] for m in meta[:-1])


def test_build_probe_dataset_shapes_and_labels(
    fz_t1: tuple[DanStateAdapter, DanStateAdapter],
) -> None:
    fz, _t1 = fz_t1
    items = [
        ("simple", Sequent.from_prompt(_SIMPLE[0]), _SIMPLE[1]),
        ("imp", Sequent.from_prompt(_IMP[0]), _IMP[1]),
    ]
    layers = (0, 1)
    examples = dsup.build_probe_dataset(fz, items, layers=layers)
    # SIMPLE has 2 line-creating steps (PR, ANDE1), IMP has 3 (PR, AS, ANDE2, IMPI) = 3 not 2;
    # recount: PR, AS, ANDE2, IMPI are all `creates_line=True` -> 4 for IMP, 2 for SIMPLE.
    assert len(examples) == 2 + 4
    by_pid: dict[str, list[dsup.DanProbeExample]] = {}
    for e in examples:
        by_pid.setdefault(e.pid, []).append(e)
    assert [e.next_rule for e in by_pid["simple"]] == ["PR", "ANDE1"]
    assert [e.actions_to_go for e in by_pid["simple"]] == [1, 0]
    assert [e.next_rule for e in by_pid["imp"]] == ["PR", "AS", "ANDE2", "IMPI"]
    for e in examples:
        assert set(e.layer_features) == {"0", "1"}
        for v in e.layer_features.values():
            assert v.shape == (fz._cfg["d"],)


def test_run_probes_dan_end_to_end(
    tmp_path: Path, fz_t1: tuple[DanStateAdapter, DanStateAdapter]
) -> None:
    fz, t1 = fz_t1
    store = Store(tmp_path / "out")
    reserve_path = tmp_path / "reserve.jsonl"
    # 7 distinct theorems (reusing the 3 proof shapes under different pids): `probes.split_by_pid`
    # rounds `val_frac`/`test_frac` (0.15 each) to the nearest item count, which needs n >= ~7 to
    # give both a non-empty val *and* test split (a 2-3 item split degenerates to all-train,
    # which crashes `_feature_matrix`'s `np.stack([])` on the empty val/test side -- a pre-
    # existing `probes.py` property this module's own `run_layer_probes_dan` inherits, not
    # something to work around here).
    _write_reserve_fixture(
        reserve_path,
        [_reserve_row(f"simple{i}", *_SIMPLE, l_true=2) for i in range(3)]
        + [_reserve_row(f"imp{i}", *_IMP, l_true=3) for i in range(2)]
        + [_reserve_row(f"or_elim{i}", *_OR_ELIM, l_true=5) for i in range(2)],
    )
    cfg: dict[str, Any] = {
        "n_train": 7,
        "l_min": 1,
        "l_max": 10,
        "reserve_path": str(reserve_path),
        "family_limit": 0,
        "layers": [0, 1],
        "seed": 0,
    }
    result = dsup.run_probes_dan(store, cfg, base_adapter=fz, rl_adapter=t1)
    assert result["n_train"] == 7
    assert result["n_rows"] == 2 * 2  # 2 models x 2 layers
    rows = load_interp_rows(store, "probes", result["run_id"])
    assert {cast(str, r["model"]) for r in rows} == {"fz", "t1"}
    assert {cast(int, r["layer"]) for r in rows} == {0, 1}
    for r in rows:
        assert "next_rule" in r and "actions_to_go" in r
    assert result["baselines"]
    assert result["baselines"]["n_train"] >= 1


# ---------------------------------------------------------------- model-label / CLI generalisation


def test_resolve_labeled_pair_builds_danstate_from_models_yaml(
    tmp_path: Path, fz_t1: tuple[DanStateAdapter, DanStateAdapter]
) -> None:
    from code.ndbench.interp.__main__ import _is_danstate, _resolve_labeled_pair

    fz, t1 = fz_t1
    registry_path = tmp_path / "models.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "dan-fz": {"adapter": "danstate", "ckpt": str(fz._ckpt_path), "family": "dan12"},
                "dan-t1": {"adapter": "danstate", "ckpt": str(t1._ckpt_path), "family": "dan12"},
            }
        )
    )
    cfg = {
        "base_model": "dan-fz",
        "rl_model": "dan-t1",
        "models_config": str(registry_path),
        "device": "cpu",
        "model_kwargs": {"batch": "8", "max_action": "16", "max_steps": "8"},
    }
    base, rl, base_label, rl_label, family, base_ckpt, rl_ckpt = _resolve_labeled_pair(cfg)
    try:
        assert base_label == "dan-fz"
        assert rl_label == "dan-t1"
        assert family == "dan12"
        assert base_ckpt == str(fz._ckpt_path)
        assert rl_ckpt == str(t1._ckpt_path)
        assert _is_danstate(base)
        assert _is_danstate(rl)
    finally:
        if base is not None:
            base.close()
        if rl is not None:
            rl.close()


def test_eval_suites_from_cfg_filters_and_validates() -> None:
    from code.ndbench.interp.runner_support import eval_suites_from_cfg

    assert eval_suites_from_cfg({}) is None
    assert eval_suites_from_cfg({"eval_suites": ["interp_eval_k128"]}) == ["interp_eval_k128"]
    with pytest.raises(ValueError):
        eval_suites_from_cfg({"eval_suites": "interp_eval_k128"})
