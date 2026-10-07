"""`train.run` on lean_thinke pickles: tokenizer from the mode, <e0> held-out states, the
checkpoint, unsupported combinations, and lean_thinkn unchanged."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import pytest

from code.thinking import ckpt as C
from code.thinking import progtrace, train
from code.thinking.data import build, build_graded, heldout_records, save
from code.thinking.effort import effort_id
from code.thinking.tok import EffortTokenizer, ThinkTokenizer

from ._pipeline_helpers import PILOT, isolate, needs_k12
from .test_data_graded import _jsonl, _specs, _traces

E = EffortTokenizer()


def _args(data: Path, out: Path, held: Path | None, **kw: Any) -> argparse.Namespace:
    a: dict[str, Any] = dict(
        data=str(data),
        heldout=None if held is None else str(held),
        out=str(out),
        seed=0,
        best_steps=4,
        budget_secs=1e9,
        init=None,
        init_seed=0,
        think_weight=1.0,
        lr_scale=1.0,
        lr=1e-3,
        wd=0.1,
        tok_budget=2048,
        mtp=0.3,
        curve_every=1e9,
        no_compile=True,
        best_dims="2,32,4,64",
        max_len=0,
        device="cpu",
    )
    a.update(kw)
    return argparse.Namespace(**a)


def _held(tmp_path: Path) -> Path:
    from ._pipeline_helpers import k12_head

    return _jsonl(tmp_path / "held.jsonl", k12_head(6))


@needs_k12
def test_effort_pickle_trains_and_reloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    isolate(tmp_path, monkeypatch)
    top = _traces(progtrace.Cfg.v1(), 12)
    data = build_graded(_specs(tmp_path, top))
    pkl = tmp_path / "g.pkl"
    save(data, pkl)
    held = _held(tmp_path)
    extra = train.run(_args(pkl, tmp_path / "run", held))
    assert extra["steps"] == 4 and extra["final_val"] is not None
    assert extra["max_len"] == 1024  # the fork's STB.train path (default flags)
    m, tok, ex = C.load_ckpt(tmp_path / "run" / "model.pt")
    assert tok.mode == "lean_thinke" and isinstance(tok, EffortTokenizer)
    assert m.cfg["vocab"] == 118 and tok.ref0 == 54 and ex["data"] == str(pkl)


@needs_k12
def test_heldout_states_end_in_e0(tmp_path: Path) -> None:
    recs = heldout_records(_held(tmp_path), E, "empty")
    out = train.with_effort(recs, E, 0)
    assert out and all(p[-1] == effort_id(E, 0) for r in out for p, _, _ in r)
    assert all(
        p[:-1] == p0
        for r, r0 in zip(out, recs, strict=True)
        for (p, _, _), (p0, _, _) in zip(r, r0, strict=True)
    )
    assert all(n == 2 for r in out for _, _, n in r)


@needs_k12
def test_unsupported_combinations_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    isolate(tmp_path, monkeypatch)
    top = _traces(progtrace.Cfg.v1(), 4)
    pkl = tmp_path / "g.pkl"
    save(build_graded(_specs(tmp_path, top)), pkl)
    with pytest.raises(SystemExit, match="unsupported"):
        train.run(_args(pkl, tmp_path / "r1", None, init="some_lean_staten.pt"))
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    d["think_format"] = "memo"
    bad = tmp_path / "memo.pkl"
    with open(bad, "wb") as f:
        pickle.dump(d, f)
    with pytest.raises(SystemExit, match="unsupported"):
        train.run(_args(bad, tmp_path / "r2", None))
    d["think_format"], d["mode"] = None, "lean_staten"
    with open(bad, "wb") as f:
        pickle.dump(d, f)
    with pytest.raises(SystemExit, match="unsupported pickle mode"):
        train.run(_args(bad, tmp_path / "r3", None))


@needs_k12
def test_lean_thinkn_pickle_still_trains(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    isolate(tmp_path, monkeypatch)
    pkl = tmp_path / "n.pkl"
    save(build(traces=[PILOT], lean=False), pkl)
    train.run(_args(pkl, tmp_path / "run", _held(tmp_path)))
    m, tok, _ = C.load_ckpt(tmp_path / "run" / "model.pt")
    assert tok.mode == "lean_thinkn" and type(tok) is ThinkTokenizer and m.cfg["vocab"] == 114
