"""Training: the copied best-recipe loop against the fork's, think weights, cold start, offline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from code.thinking import ckpt as C
from code.thinking.data import build, to_fork
from code.thinking.tok import D, ThinkTokenizer, remap_ids
from code.thinking.train import finetune, make_batch_w, think_lens_tensor, train_best

from ._pipeline_helpers import PILOT, STAGE1, isolate, tiny_model

OLD = D.lean_tok.LeanTokenizer("lean_staten")


def _args(**kw: Any) -> argparse.Namespace:
    a = dict(
        seed=0,
        n_layer=2,
        d=32,
        n_head=4,
        d_ff=64,
        lr=1e-3,
        wd=0.1,
        best_steps=6,
        budget_secs=1e9,
        tok_budget=2048,
        mtp=0.3,
        curve_every=1e9,
        no_compile=True,
        think_weight=1.0,
        lr_scale=1.0,
    )
    a.update(kw)
    return argparse.Namespace(**a)


def _pilot_records() -> list[Any]:
    return build(traces=[PILOT], lean=False)["records"]


def test_train_copy_matches_fork() -> None:
    """With no init model and weight 1 the copy reproduces state_train_best.train bit for bit."""
    t = ThinkTokenizer()
    recs = _pilot_records()
    m1, e1 = train_best(_args(), t, recs, None, "cpu", D.record)  # delegates to the fork
    m2, e2 = train_best(_args(), t, recs, None, "cpu", D.record, force_copy=True)
    s1, s2 = m1.state_dict(), m2.state_dict()
    assert s1.keys() == s2.keys()
    for k in s1:
        assert torch.equal(s1[k], s2[k]), k
    assert e1["steps"] == e2["steps"] == 6 and e1["pairs_seen"] == e2["pairs_seen"]


def test_think_weight_zero_is_the_action_only_loss() -> None:
    t = ThinkTokenizer()
    recs = _pilot_records()
    ids, plen, _, lens = D.state_train_best.to_device(to_fork(recs), "cpu")
    tlen = think_lens_tensor(recs)
    idxs = torch.arange(ids.shape[0])
    model = tiny_model(t.vocab_size)
    x, m0 = make_batch_w(ids, plen, tlen, lens, idxs, ids.shape[1], t, None, 0.0, shift=False)
    _, m1 = make_batch_w(ids, plen, tlen, lens, idxs, ids.shape[1], t, None, 1.0, shift=False)
    with torch.no_grad():
        logits = model(x[:, :-1])
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1), reduction="none"
        ).view(x.shape[0], -1)
        ar = torch.arange(x.shape[1])[None]
        act = (ar >= (plen + tlen)[:, None]) & (ar < lens[:, None])
        want = (ce * act[:, 1:]).sum() / act[:, 1:].sum()
        got = D.best_model.mtp_loss(model, None, x, m0, weight=0)[1]
        full = D.best_model.mtp_loss(model, None, x, m1, weight=0)[1]
    assert torch.allclose(got, want, atol=1e-6) and not torch.allclose(got, full)


def test_think_weight_trains_through_the_copy() -> None:
    t = ThinkTokenizer()
    m, e = train_best(
        _args(think_weight=0.5, best_steps=3), t, _pilot_records(), None, "cpu", D.record
    )
    assert e["steps"] == 3 and m.cfg["vocab"] == 114


def _old_inputs(n: int = 4, T: int = 30) -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randint(0, OLD.vocab_size, (n, T))


def test_cold_start_preserves_old_logits_tiny() -> None:
    old = tiny_model(OLD.vocab_size, seed=3)
    new, nt = C.expand_vocab(old, OLD, seed=1)
    r = torch.tensor(remap_ids(OLD, nt))
    x = _old_inputs()
    with torch.no_grad():
        a, b = old(x), new(r[x])
    assert torch.equal(b[..., r], a)
    added = [nt.think, nt.end_think, nt.no]
    hw, ew = new.head.weight.detach(), new.emb.weight.detach()
    assert torch.allclose(hw[added], old.head.weight.detach().mean(0).expand(3, -1))
    assert len({tuple(ew[i].tolist()) for i in added}) == 3
    assert new.cfg["vocab"] == 114 and old.cfg["vocab"] == 111


@pytest.mark.skipif(not STAGE1.exists(), reason="Stage-1 checkpoint not available")
def test_cold_start_preserves_old_logits_stage1() -> None:
    old, tok, _ = C.load_ckpt(STAGE1)
    new, nt, info = C.cold_start(STAGE1)
    assert info["init_mode"] == "lean_staten"
    r = torch.tensor(remap_ids(tok, nt))
    x = _old_inputs(3, 40)
    with torch.no_grad():
        assert torch.equal(new(r[x])[..., r], old(x))


def test_vocab_tensors_checks_the_layout() -> None:
    m = tiny_model(OLD.vocab_size)
    assert set(C.vocab_tensors(m)) == {"emb.weight", "head.weight"}
    m.head.weight = m.emb.weight  # tied
    with pytest.raises(AssertionError):
        C.vocab_tensors(m)


def test_save_never_uploads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ND_OFFLINE", raising=False)
    monkeypatch.delenv("ND_REGISTRY", raising=False)
    monkeypatch.setenv("ND_RUN_ID", "must-not-upload")
    for k in ("ND_REGISTRY_SYNC", "ND_REGISTRY_DIR", "LEAN_GATE_LOG"):
        monkeypatch.setenv(k, "unset-by-test")
    calls: list[Any] = []

    def boom(*a: Any, **k: Any) -> Any:
        calls.append(a)
        raise AssertionError("upload path taken")

    monkeypatch.setattr(D.record, "bucket_cp", boom)
    monkeypatch.setattr(D.record, "hf_bin", boom)
    monkeypatch.setattr(D.record, "remote_size", boom)
    out = tmp_path / "run"
    env = C.run_env(out)
    assert os.environ["ND_OFFLINE"] == "1" and env["ND_REGISTRY_DIR"] == str(
        (out / "registry").resolve()
    )
    os.environ.pop("ND_OFFLINE")  # even if something unsets it, save_ckpt forces it again
    info = C.save_ckpt(out / "m.pt", tiny_model(114), "lean_thinkn", extra={"x": 1})
    assert calls == [] and info["offline"] is True and info["uri"] is None
    assert json.loads((out / "m.pt.upload.json").read_text())["offline"] is True
    D.record.sync(raise_on_fail=True)
    assert calls == []
    rows = list((out / "registry").glob("*.jsonl"))
    assert len(rows) == 1 and "ckpt_saved" in rows[0].read_text()
    m, tok, extra = C.load_ckpt(out / "m.pt")
    assert tok.mode == "lean_thinkn" and extra == {"x": 1} and m.cfg["vocab"] == 114
    D.record._state["file"] = None


def test_finetune_runs_with_think_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    isolate(tmp_path, monkeypatch)
    t = ThinkTokenizer()
    recs = _pilot_records()
    m = tiny_model(t.vocab_size)
    m, last = finetune(m, t, recs, steps=4, recs=4, warmup=2, think_weight=0.0, log_every=100)
    assert last > 0 and not m.training
    m, last = finetune(m, t, recs, steps=2, recs=4, warmup=1, log_every=100)
    assert last > 0
