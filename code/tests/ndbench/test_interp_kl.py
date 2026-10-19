"""`adapters.danstate.DanStateAdapter.masked_next_logprobs` / `adapters.absgpt.AbsGptAdapter.
masked_next_logprobs` (the exact, full-distribution teacher-forcing primitive each adapter adds)
and `interp/kl.py`'s `run_kl` (the token-level KL measurement built on top of them): tiny random
CPU checkpoints for both model families, on temp `Store`s -- no GPU, no real checkpoints.
"""

from __future__ import annotations

import json
import math
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from code.ndbench.adapters.absgpt import GPT, AbsGptAdapter, Tokenizer
from code.ndbench.adapters.danstate import DanStateAdapter
from code.ndbench.core.io import append_jsonl, utc_now_iso
from code.ndbench.core.types import LengthLabel, Problem, Sequent, SuiteManifest
from code.ndbench.interp import kl
from code.ndbench.interp._store import interp_runs, load_interp_rows
from code.ndbench.store import Store
from code.ndbench.vendor import load_danstate

# ----------------------------------------------------------------------------------- fixtures

_SIMPLE = ("THM ( P & Q ) SEQ P PRF", "N1 ( P & Q ) : PR ; N2 P : ANDE1 N1 ; QED")
_IMP = (
    "THM ( P & Q ) SEQ ( P > Q ) PRF",
    "N1 ( P & Q ) : PR ; N2 | P : AS ; N3 | Q : ANDE2 N1 ; N4 ( P > Q ) : IMPI N2 N3 ; QED",
)
#: N2/N3 open and close a box no IMPI/NEGI/ORE ever cites -- outside the grammar by default
#: (`DanStateAdapter.logprob`'s `allow_reencode=False` convention; `masked_next_logprobs` mirrors
#: it exactly).
_UNCITED_BOX = (
    "THM ( P & Q ) SEQ P PRF",
    "N1 ( P & Q ) : PR ; N2 | P : AS ; N3 P : ANDE1 N1 ; QED",
)


def _dan_tiny_cfg() -> dict[str, Any]:
    return dict(vocab=111, n_layer=2, d=32, n_head=4, max_len=64, d_ff=64, arch="best")


def _save_tiny_dan_ckpt(path: Path, *, seed: int) -> None:
    ds = load_danstate()
    torch.manual_seed(seed)
    cfg = _dan_tiny_cfg()
    model = ds.best_model.ALiBiGPT(**cfg)
    torch.save(
        {"cfg": model.cfg, "state": model.state_dict(), "tok_mode": "lean_staten", "extra": {}},
        path,
    )


@pytest.fixture
def fz_t1(tmp_path: Path) -> Generator[tuple[DanStateAdapter, DanStateAdapter], None, None]:
    """`name_base_max=0`: exactly one valid canonical-name base, so `logprob`'s own marginal
    collapses to that single base's score (no `logsumexp`/`- log(name_base_max + 1)` offset to
    account for) -- lets `test_masked_next_logprobs_dan_logratio_matches_logprob` compare
    `masked_next_logprobs`' `logratio_sum` directly against `logprob(...).nats` differences."""
    fz_ckpt, t1_ckpt = tmp_path / "fz.pt", tmp_path / "t1.pt"
    _save_tiny_dan_ckpt(fz_ckpt, seed=1)
    _save_tiny_dan_ckpt(t1_ckpt, seed=2)
    fz = DanStateAdapter(
        ckpt=fz_ckpt, device="cpu", batch=8, max_action=16, max_steps=8, name_base_max=0
    )
    t1 = DanStateAdapter(
        ckpt=t1_ckpt, device="cpu", batch=8, max_action=16, max_steps=8, name_base_max=0
    )
    try:
        yield fz, t1
    finally:
        fz.close()
        t1.close()


def _make_tiny_absgpt_ckpt(path: Path, *, seed: int) -> None:
    torch.manual_seed(seed)
    tok = Tokenizer("abs")
    model = GPT(vocab=tok.vocab_size, n_layer=1, d=32, n_head=4, max_len=128)
    torch.save({"state": model.state_dict(), "tok_mode": "abs", "cfg": dict(model.cfg)}, path)


@pytest.fixture
def abs_base_rl(tmp_path: Path) -> Generator[tuple[AbsGptAdapter, AbsGptAdapter], None, None]:
    base_ckpt, rl_ckpt = tmp_path / "abs_base.pt", tmp_path / "abs_rl.pt"
    _make_tiny_absgpt_ckpt(base_ckpt, seed=3)
    _make_tiny_absgpt_ckpt(rl_ckpt, seed=4)
    base = AbsGptAdapter(ckpt=base_ckpt, device="cpu", batch=4)
    rl = AbsGptAdapter(ckpt=rl_ckpt, device="cpu", batch=4)
    try:
        yield base, rl
    finally:
        base.close()
        rl.close()


def _manual_full_logp(model: Any, tok: Any, ids: list[int], mask_start: int) -> np.ndarray:
    """Manual, unbatched reference: `log_softmax` at every position `>= mask_start` of a single
    teacher-forced row (the same computation `test_dan_adapter.py`'s own `_manual_step_logprob`
    uses, generalised to return the full vocab row rather than gathering one target)."""
    import torch.nn.functional as F

    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(x[:, :-1]).float()
    logp = F.log_softmax(logits, dim=-1)
    return logp[0, mask_start:].numpy()


# --------------------------------------------------------- DanStateAdapter.masked_next_logprobs


def test_masked_next_logprobs_dan_self_vs_self_zero_kl(
    fz_t1: tuple[DanStateAdapter, DanStateAdapter],
) -> None:
    fz, _t1 = fz_t1
    seq = Sequent.from_prompt(_IMP[0])
    items = [(seq, _IMP[1], 0)]
    (r1,) = fz.masked_next_logprobs(items)
    (r2,) = fz.masked_next_logprobs(items)
    assert r1 is not None and r2 is not None
    assert r1.n_tokens == r2.n_tokens > 0
    assert np.array_equal(r1.target_ids, r2.target_ids)
    p = np.exp(r1.log_probs.astype(np.float64))
    kl_self = float(np.sum(p * (r1.log_probs.astype(np.float64) - r2.log_probs.astype(np.float64))))
    assert kl_self == pytest.approx(0.0, abs=1e-5)


def test_masked_next_logprobs_dan_kl_nonnegative_for_different_weights(
    fz_t1: tuple[DanStateAdapter, DanStateAdapter],
) -> None:
    fz, t1 = fz_t1
    seq = Sequent.from_prompt(_IMP[0])
    items = [(seq, _IMP[1], 0)]
    (r_fz,) = fz.masked_next_logprobs(items)
    (r_t1,) = t1.masked_next_logprobs(items)
    assert r_fz is not None and r_t1 is not None
    assert np.array_equal(r_fz.target_ids, r_t1.target_ids)
    p = np.exp(r_fz.log_probs.astype(np.float64))
    kl_val = float(
        np.sum(p * (r_fz.log_probs.astype(np.float64) - r_t1.log_probs.astype(np.float64)))
    )
    assert kl_val >= -1e-6


def test_masked_next_logprobs_dan_matches_manual_full_distribution(
    fz_t1: tuple[DanStateAdapter, DanStateAdapter],
) -> None:
    """One step's full `log_softmax` row matches an unbatched manual reference exactly (module
    docstring's `_manual_full_logp`), for both the state-action row's own `mask_start` and its
    full vocab width."""
    fz, _t1 = fz_t1
    ds = load_danstate()
    prompt, body = _SIMPLE
    seq = Sequent.from_prompt(prompt)
    (r,) = fz.masked_next_logprobs([(seq, body, 0)])
    assert r is not None

    steps, _toks, _env = ds.state_env.decompose(prompt, body, canon=True)
    (state_toks, action_toks, _hist) = steps[0]
    pre = fz._tok.encode_toks(state_toks)
    post = fz._tok.encode_toks(action_toks) + [fz._tok.eos]
    manual = _manual_full_logp(fz._model, fz._tok, pre + post, len(pre) - 1)
    n0 = len(post)
    np.testing.assert_allclose(r.log_probs[:n0], manual, atol=1e-5)
    assert list(r.target_ids[:n0]) == post


def test_masked_next_logprobs_dan_logratio_matches_logprob(
    fz_t1: tuple[DanStateAdapter, DanStateAdapter],
) -> None:
    """With `name_base_max=0` (the `fz_t1` fixture's own setup), `logprob`'s marginal collapses to
    a single encoding, so `masked_next_logprobs`' `logratio_sum` (gathered at each model's own
    target, summed over scored positions) reproduces `logprob(...).nats`'s difference exactly --
    `interp/kl.py`'s "logratio_sum matches logprob at the sample's base" contract."""
    fz, t1 = fz_t1
    seq = Sequent.from_prompt(_IMP[0])
    (lp_fz,) = fz.logprob([(seq, _IMP[1])])
    (lp_t1,) = t1.logprob([(seq, _IMP[1])])
    assert len(lp_fz.per_encoding) == 1 and len(lp_t1.per_encoding) == 1

    items = [(seq, _IMP[1], 0)]
    (r_fz,) = fz.masked_next_logprobs(items)
    (r_t1,) = t1.masked_next_logprobs(items)
    assert r_fz is not None and r_t1 is not None
    assert np.array_equal(r_fz.target_ids, r_t1.target_ids)
    logratio_sum = sum(
        float(r_t1.log_probs[k, r_t1.target_ids[k]] - r_fz.log_probs[k, r_fz.target_ids[k]])
        for k in range(r_t1.n_tokens)
    )
    assert math.isclose(logratio_sum, lp_t1.nats - lp_fz.nats, abs_tol=1e-4)


def test_masked_next_logprobs_dan_exclude_defining_names_default_is_off(
    fz_t1: tuple[DanStateAdapter, DanStateAdapter],
) -> None:
    fz, _t1 = fz_t1
    seq = Sequent.from_prompt(_IMP[0])
    (r_default,) = fz.masked_next_logprobs([(seq, _IMP[1], 0)])
    (r_excl,) = fz.masked_next_logprobs([(seq, _IMP[1], 0)], exclude_defining_names=True)
    (r_incl,) = fz.masked_next_logprobs([(seq, _IMP[1], 0)], exclude_defining_names=False)
    assert r_default is not None and r_excl is not None and r_incl is not None
    # `masked_next_logprobs`' own `exclude_defining_names` default is False (the Dan adapter's
    # existing `logprob` convention); `interp/kl.py` is the one that defaults it to True --
    # checked separately via `_run_kl_dan`'s own cfg.
    assert r_default.n_tokens == r_incl.n_tokens
    assert r_excl.n_tokens < r_incl.n_tokens


def test_masked_next_logprobs_dan_none_on_uncited_box_unless_allow_reencode(
    fz_t1: tuple[DanStateAdapter, DanStateAdapter],
) -> None:
    fz, _t1 = fz_t1
    seq = Sequent.from_prompt(_UNCITED_BOX[0])
    (r,) = fz.masked_next_logprobs([(seq, _UNCITED_BOX[1], 0)])
    assert r is None
    (r_reencode,) = fz.masked_next_logprobs([(seq, _UNCITED_BOX[1], 0)], allow_reencode=True)
    assert r_reencode is not None


def test_masked_next_logprobs_dan_none_on_malformed_body(
    fz_t1: tuple[DanStateAdapter, DanStateAdapter],
) -> None:
    fz, _t1 = fz_t1
    seq = Sequent.from_prompt("THM ( P & Q ) SEQ P PRF")
    (r,) = fz.masked_next_logprobs([(seq, "N1 ( P & Q ) : NOTARULE ; QED", 0)])
    assert r is None


def test_masked_next_logprobs_dan_rows_schema(
    fz_t1: tuple[DanStateAdapter, DanStateAdapter],
) -> None:
    fz, _t1 = fz_t1
    seq = Sequent.from_prompt(_IMP[0])
    (r,) = fz.masked_next_logprobs([(seq, _IMP[1], 0)])
    assert r is not None
    assert r.n_steps == 5  # PR, AS, ANDE2, IMPI, FINAL
    assert r.log_probs.shape == (r.n_tokens, fz._tok.vocab_size)
    assert r.target_ids.shape == (r.n_tokens,)
    assert r.step_of.shape == (r.n_tokens,)
    assert r.step_of.max() == r.n_steps - 1
    assert r.n_tokens == int((r.step_of >= 0).sum())


# ------------------------------------------------------------ AbsGptAdapter.masked_next_logprobs


def test_masked_next_logprobs_absgpt_self_vs_self_zero_kl(
    abs_base_rl: tuple[AbsGptAdapter, AbsGptAdapter],
) -> None:
    base, _rl = abs_base_rl
    seq = Sequent.from_prompt(_IMP[0])
    items = [(seq, _IMP[1])]
    (r1,) = base.masked_next_logprobs(items)
    (r2,) = base.masked_next_logprobs(items)
    assert r1 is not None and r2 is not None
    p = np.exp(r1.log_probs.astype(np.float64))
    kl_self = float(np.sum(p * (r1.log_probs.astype(np.float64) - r2.log_probs.astype(np.float64))))
    assert kl_self == pytest.approx(0.0, abs=1e-5)


def test_masked_next_logprobs_absgpt_kl_nonnegative(
    abs_base_rl: tuple[AbsGptAdapter, AbsGptAdapter],
) -> None:
    base, rl = abs_base_rl
    seq = Sequent.from_prompt(_IMP[0])
    items = [(seq, _IMP[1])]
    (r_base,) = base.masked_next_logprobs(items)
    (r_rl,) = rl.masked_next_logprobs(items)
    assert r_base is not None and r_rl is not None
    assert np.array_equal(r_base.target_ids, r_rl.target_ids)
    p = np.exp(r_rl.log_probs.astype(np.float64))
    diff = r_rl.log_probs.astype(np.float64) - r_base.log_probs.astype(np.float64)
    kl_val = float(np.sum(p * diff))
    assert kl_val >= -1e-6


def test_masked_next_logprobs_absgpt_matches_manual_full_distribution(
    abs_base_rl: tuple[AbsGptAdapter, AbsGptAdapter],
) -> None:
    base, _rl = abs_base_rl
    seq = Sequent.from_prompt(_IMP[0])
    (r,) = base.masked_next_logprobs([(seq, _IMP[1])])
    assert r is not None
    pid = base._tok.encode_prompt(seq.prompt())
    qid = base._tok.encode_proof(_IMP[1])
    manual = _manual_full_logp(base._model, base._tok, pid + qid, len(pid) - 1)
    np.testing.assert_allclose(r.log_probs, manual, atol=1e-5)
    assert list(r.target_ids) == qid


def test_masked_next_logprobs_absgpt_rows_schema_and_step_of(
    abs_base_rl: tuple[AbsGptAdapter, AbsGptAdapter],
) -> None:
    base, _rl = abs_base_rl
    seq = Sequent.from_prompt(_IMP[0])
    (r,) = base.masked_next_logprobs([(seq, _IMP[1])])
    assert r is not None
    # 4 lines (N1 PR, N2 AS, N3 ANDE2, N4 IMPI) + 1 pseudo "FINAL" step for QED.
    assert r.n_steps == 5
    assert r.step_of[-1] == 4  # the trailing QED token's own pseudo-step
    assert r.log_probs.shape == (r.n_tokens, base._tok.vocab_size)


def test_masked_next_logprobs_absgpt_none_on_truncated_native(
    abs_base_rl: tuple[AbsGptAdapter, AbsGptAdapter],
) -> None:
    base, _rl = abs_base_rl
    seq = Sequent.from_prompt(_IMP[0])
    truncated = "N1 ( P & Q ) : PR ; N2 | P : AS"  # no closing ';', no QED
    (r,) = base.masked_next_logprobs([(seq, truncated)])
    assert r is None


# ------------------------------------------------------------------- _stratified_subsample


def test_stratified_subsample_round_robins_by_pid() -> None:
    samples = [
        kl._RawSample(pid="a", i=i, chunk=0, body="", native=None, ok=True) for i in range(5)
    ] + [kl._RawSample(pid="b", i=i, chunk=0, body="", native=None, ok=True) for i in range(2)]
    out = kl._stratified_subsample(samples, 4)
    assert len(out) == 4
    # round-robin: a0, b0, a1, b1 -- every pid contributes before any pid contributes twice.
    assert [(s.pid, s.i) for s in out] == [("a", 0), ("b", 0), ("a", 1), ("b", 1)]
    assert kl._stratified_subsample(samples, 100) == samples
    assert kl._stratified_subsample(samples, None) == samples


# ----------------------------------------------------------------------------------- run_kl: Dan


def _dan_native(base: int, steps: int, end: str, lean: str | None = None) -> str:
    return json.dumps({"lean": lean, "base": base, "steps": steps, "end": end})


def _write_problem_suite(
    store: Store, suite: str, version: str, items: list[tuple[str, str, str]]
) -> None:
    problems = [
        Problem(
            pid=pid,
            suite=suite,
            version=version,
            sequent=Sequent.from_prompt(prompt),
            length=LengthLabel(L=None, L_ub=None, L_lb=None, method="reference", bound=None),
            ref=None,
            tags={},
        )
        for pid, prompt, _body in items
    ]
    store.write_suite(
        problems,
        SuiteManifest(
            suite=suite,
            version=version,
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


def _register_run(
    store: Store,
    run_id: str,
    *,
    suite: str,
    version: str,
    model: str,
    judge_id: str,
    k: int,
) -> None:
    append_jsonl(
        store.runs_path,
        [
            {
                "run_id": run_id,
                "suite": suite,
                "version": version,
                "model": model,
                "status": "complete",
                "judge_id": judge_id,
                "k": k,
                "finished_utc": utc_now_iso(),
            }
        ],
    )


def test_run_kl_dan_end_to_end(
    tmp_path: Path, fz_t1: tuple[DanStateAdapter, DanStateAdapter]
) -> None:
    fz, t1 = fz_t1
    store = Store(tmp_path / "out")
    items = [("simple", _SIMPLE[0], _SIMPLE[1]), ("imp", _IMP[0], _IMP[1])]
    _write_problem_suite(store, "s", "v1", items)

    rl_run_id, base_run_id = "s-v1-t1-r0", "s-v1-fz-r0"
    _register_run(store, rl_run_id, suite="s", version="v1", model="t1", judge_id="lean_v2", k=4)
    _register_run(store, base_run_id, suite="s", version="v1", model="fz", judge_id="lean_v2", k=1)

    done0 = _dan_native(0, 2, "done")
    store.append_samples(
        rl_run_id,
        [
            {"pid": "simple", "chunk": 0, "i": 0, "body": _SIMPLE[1], "native": done0, "ok": True},
            {
                "pid": "imp",
                "chunk": 0,
                "i": 0,
                "body": _IMP[1],
                "native": _dan_native(0, 4, "done"),
                "ok": True,
            },
            {
                "pid": "imp",
                "chunk": 0,
                "i": 1,
                "body": "UNPARSEABLE env: bad",
                "native": _dan_native(0, 3, "syntax:bad"),
                "ok": False,
            },
            {
                "pid": "no_such_pid",
                "chunk": 0,
                "i": 0,
                "body": _SIMPLE[1],
                "native": done0,
                "ok": True,
            },
        ],
    )
    store.append_samples(
        base_run_id,
        [{"pid": "simple", "chunk": 0, "i": 0, "body": _SIMPLE[1], "native": done0, "ok": False}],
    )

    cfg: dict[str, Any] = {
        "max_samples_per_suite": 100,
        "suites": [{"suite": "s", "version": "v1"}],
        "exclude_defining_names": True,
    }
    result = kl.run_kl(store, cfg, base_adapter=fz, rl_adapter=t1, base_label="fz", rl_label="t1")

    assert result["n_rows"] == 3  # 2 rl rows (unparseable + unknown-pid skipped) + 1 base row
    assert result["n_skipped"] == 2
    assert result["suites"] == [{"suite": "s", "version": "v1"}]
    assert result["run_ids"] == {"s:v1:t1": rl_run_id, "s:v1:fz": base_run_id}

    rows = [cast("dict[str, Any]", r) for r in load_interp_rows(store, "kl", result["run_id"])]
    assert len(rows) == 3
    by_direction: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_direction.setdefault(row["direction"], []).append(row)
        assert row["suite"] == "s"
        assert set(row) == {
            "pid",
            "suite",
            "direction",
            "policy",
            "ok",
            "n_steps",
            "n_tokens",
            "kl_sum",
            "kl_per_token",
            "logratio_sum",
            "by_kind",
        }
        assert row["n_tokens"] > 0
        assert row["kl_sum"] >= -1e-6
        assert row["kl_per_token"] == pytest.approx(row["kl_sum"] / row["n_tokens"])
        assert sum(v[1] for v in row["by_kind"].values()) == row["n_tokens"]

    assert {r["pid"] for r in by_direction["rl||base"]} == {"simple", "imp"}
    assert all(r["policy"] == "t1" for r in by_direction["rl||base"])
    assert {r["pid"] for r in by_direction["base||rl"]} == {"simple"}
    assert by_direction["base||rl"][0]["policy"] == "fz"
    assert by_direction["base||rl"][0]["ok"] is False

    # Run cfg records base_model/rl_model/exclude_defining_names/max_samples_per_suite/suites/
    # run ids.
    (run_record,) = [r for r in interp_runs(store, "kl") if r["run_id"] == result["run_id"]]
    recorded_cfg = cast("dict[str, Any]", run_record["cfg"])
    assert recorded_cfg["base_model"] == "fz"
    assert recorded_cfg["rl_model"] == "t1"
    assert recorded_cfg["exclude_defining_names"] is True
    assert recorded_cfg["max_samples_per_suite"] == 100
    assert recorded_cfg["suites"] == [{"suite": "s", "version": "v1"}]
    assert recorded_cfg["run_ids"] == {"s:v1:t1": rl_run_id, "s:v1:fz": base_run_id}

    pooled = result["pooled"]
    assert pooled["rl||base"]["n"] == 2
    assert pooled["base||rl"]["n"] == 1
    assert pooled["rl||base"]["by_ok"]["ok"]["n"] + pooled["rl||base"]["by_ok"]["not_ok"]["n"] == 2
    assert pooled["base||rl"]["by_ok"]["not_ok"]["n"] == 1
    assert "mean_kl_per_token_pooled" in pooled["rl||base"]
    assert "by_kind" in pooled["rl||base"]


def test_run_kl_dan_defaults_exclude_defining_names_true(
    tmp_path: Path, fz_t1: tuple[DanStateAdapter, DanStateAdapter]
) -> None:
    """`interp/kl.py`'s own default differs from `masked_next_logprobs`'s (False, matching
    `logprob`): `_run_kl_dan` defaults `exclude_defining_names` to `True` (module docstring's
    "make this the default and record it in the output")."""
    fz, t1 = fz_t1
    store = Store(tmp_path / "out")
    _write_problem_suite(store, "s", "v1", [("imp", _IMP[0], _IMP[1])])
    rl_run_id = "s-v1-t1-r0"
    _register_run(store, rl_run_id, suite="s", version="v1", model="t1", judge_id="lean_v2", k=1)
    store.append_samples(
        rl_run_id,
        [
            {
                "pid": "imp",
                "chunk": 0,
                "i": 0,
                "body": _IMP[1],
                "native": _dan_native(0, 4, "done"),
                "ok": True,
            }
        ],
    )
    cfg: dict[str, Any] = {"max_samples_per_suite": 10, "suites": [{"suite": "s", "version": "v1"}]}
    result = kl.run_kl(store, cfg, base_adapter=fz, rl_adapter=t1, base_label="fz", rl_label="t1")
    (run_record,) = [r for r in interp_runs(store, "kl") if r["run_id"] == result["run_id"]]
    recorded_cfg = cast("dict[str, Any]", run_record["cfg"])
    assert recorded_cfg["exclude_defining_names"] is True


# -------------------------------------------------------------------------------- run_kl: absgpt


def test_run_kl_absgpt_end_to_end(
    tmp_path: Path, abs_base_rl: tuple[AbsGptAdapter, AbsGptAdapter]
) -> None:
    base, rl = abs_base_rl
    store = Store(tmp_path / "out")
    items = [("imp", _IMP[0], _IMP[1])]
    _write_problem_suite(store, "s", "v1", items)  # family-agnostic: just writes a Problem suite

    rl_run_id, base_run_id = "s-v1-rl-r0", "s-v1-base-r0"
    _register_run(store, rl_run_id, suite="s", version="v1", model="rl", judge_id="nd_v2", k=2)
    _register_run(store, base_run_id, suite="s", version="v1", model="base", judge_id="nd_v2", k=1)

    store.append_samples(
        rl_run_id,
        [
            {"pid": "imp", "chunk": 0, "i": 0, "body": _IMP[1], "native": _IMP[1], "ok": True},
            {
                "pid": "imp",
                "chunk": 0,
                "i": 1,
                "body": "N1 ( P & Q ) : PR",
                "native": "N1 ( P & Q ) : PR",  # truncated, no QED -- skipped
                "ok": False,
            },
        ],
    )
    store.append_samples(
        base_run_id,
        [{"pid": "imp", "chunk": 0, "i": 0, "body": _IMP[1], "native": _IMP[1], "ok": False}],
    )

    cfg: dict[str, Any] = {"max_samples_per_suite": 10, "suites": [{"suite": "s", "version": "v1"}]}
    result = kl.run_kl(
        store, cfg, base_adapter=base, rl_adapter=rl, base_label="base", rl_label="rl"
    )

    assert result["n_rows"] == 2
    assert result["n_skipped"] == 1
    rows = [cast("dict[str, Any]", r) for r in load_interp_rows(store, "kl", result["run_id"])]
    assert len(rows) == 2
    assert {(r["direction"], r["pid"]) for r in rows} == {("rl||base", "imp"), ("base||rl", "imp")}
    for row in rows:
        assert row["n_steps"] == 5  # 4 lines + the trailing "FINAL" QED step
        assert "FINAL" in row["by_kind"] or row["n_tokens"] == 0


# ------------------------------------------------------------------------------- dispatch errors


def test_run_kl_rejects_mismatched_adapter_pair(
    tmp_path: Path,
    fz_t1: tuple[DanStateAdapter, DanStateAdapter],
    abs_base_rl: tuple[AbsGptAdapter, AbsGptAdapter],
) -> None:
    fz, _t1 = fz_t1
    base, _rl = abs_base_rl
    store = Store(tmp_path / "out")
    with pytest.raises(ValueError, match="unsupported adapter pair"):
        kl.run_kl(store, {}, base_adapter=fz, rl_adapter=base, base_label="fz", rl_label="rl")


def test_run_kl_rejects_missing_adapter(tmp_path: Path) -> None:
    store = Store(tmp_path / "out")
    with pytest.raises(ValueError, match="needs both"):
        kl.run_kl(store, {}, base_adapter=None, rl_adapter=None)
