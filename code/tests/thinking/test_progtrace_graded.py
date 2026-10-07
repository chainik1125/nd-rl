"""Graded effort levels of the programmatic traces: budgets, strict validity, e256 == v1,
compression (not truncation) and the CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from code.thinking import progtrace as pt

from ._pipeline_helpers import k12_head, needs_k12

N = 150
STORED = Path("/scratch/cowifornia/nd-rl/think/traces/prog_traces.jsonl")

Out = list[tuple[dict[str, Any], list[dict[str, Any]]]]


def _run(level: int, n: int = N, ds: dict[str, Any] | None = None) -> Out:
    cfg = pt.Cfg.v1(level=level)
    return [
        pt.trace_for(f"k12_{i}", r["prompt"], r["proof"], cfg, seed=0, source="K12")
        for i, r in enumerate(k12_head(n))
    ]


def _body(step: dict[str, str]) -> int:
    return len(pt.td().tokens(step["think"])) if step["think"] else 0


@pytest.fixture(scope="module")
def runs() -> dict[int, Out]:
    return {lv: _run(lv) for lv in (0, 16, 64, 256)}


def test_bad_level_rejected() -> None:
    with pytest.raises(ValueError):
        pt.Cfg(level=32)
    assert pt.Cfg.v1(level=16).level == 16


@needs_k12
def test_levels_respect_budget(runs: dict[int, Out]) -> None:
    for lv in (16, 64):
        for rec, infos in runs[lv]:
            for st, inf in zip(rec["steps"], infos, strict=True):
                assert _body(st) <= lv, (rec["id"], lv, st["think"])
                assert inf["think_tokens"] == _body(st)
            assert rec["gen"]["level"] == lv


@needs_k12
def test_levels_validate_strictly_and_keep_actions(runs: dict[int, Out]) -> None:
    T = pt.td()
    for lv in (0, 16, 64):
        for (rec, _), (rec0, _) in zip(runs[lv], runs[256], strict=True):
            v = T.validate(rec, lean=False, strict=True)
            assert v["ok"], (rec["id"], lv, v["error"])
            assert [s["action"] for s in rec["steps"]] == [s["action"] for s in rec0["steps"]]


@needs_k12
def test_e256_equals_v1_and_stored_traces(runs: dict[int, Out]) -> None:
    plain = [
        pt.trace_for(f"k12_{i}", r["prompt"], r["proof"], pt.Cfg.v1(), seed=0, source="K12")[0]
        for i, r in enumerate(k12_head(N))
    ]
    assert [r for r, _ in runs[256]] == plain
    assert all("level" not in r["gen"] for r in plain)
    if not STORED.exists():
        pytest.skip("stored v1 traces missing")
    ds = pt.load_distractors()
    rows = k12_head(100)
    with open(STORED) as f:
        stored = [json.loads(next(f)) for _ in range(100)]
    for i, (r, s) in enumerate(zip(rows, stored, strict=True)):
        pid = f"k12_{i}"
        assert s["id"] == pid
        rec, _ = pt.trace_for(
            pid, r["prompt"], r["proof"], pt.Cfg.v1(), 0, 0, ds.get(pid), source="K12"
        )
        assert rec["steps"] == s["steps"], pid


@needs_k12
def test_e0_is_empty(runs: dict[int, Out]) -> None:
    for rec, infos in runs[0]:
        assert all(st["think"] == "" for st in rec["steps"])
        assert all(inf["think_tokens"] == 0 for inf in infos)
        assert rec["gen"]["level"] == 0


@needs_k12
def test_compression_not_truncation(runs: dict[int, Out]) -> None:
    T = pt.td()
    for lv in (16, 64):
        for (rec, infos), (rec0, _), r in zip(runs[lv], runs[256], k12_head(N), strict=True):
            envs = pt.build_tree(r["prompt"], [pt.td().tokens(s["action"]) for s in rec["steps"]])[
                0
            ]
            for st, st0, inf, env in zip(rec["steps"], rec0["steps"], infos, envs, strict=True):
                assert inf.get("other_segments", 0) == 0
                if not st["think"]:
                    continue
                full = set(st0["think"].split("\n"))
                full |= {ln + " <no>" for ln in full}
                full.add("<no>")
                sc = T.scope(env)
                looks = {f"{n} : {pt.ascii_line(pt.ftoks(f))}" for n, f in sc.items()}
                for ln in st["think"].split("\n"):
                    assert ln in full or ln in looks or ln == st["action"], (
                        rec["id"],
                        lv,
                        ln,
                    )


@needs_k12
def test_ladder_grades(runs: dict[int, Out]) -> None:
    mean: dict[int, float] = {}
    for lv in (16, 64, 256):
        xs = [
            inf["think_tokens"]
            for _, infos in runs[lv]
            for inf in infos
            if inf["kind"] == "decision" and inf["think_tokens_e256"]
        ]
        mean[lv] = sum(xs) / len(xs)
    assert mean[16] < mean[64] < mean[256]
    for (rec64, infos), (rec, _) in zip(runs[64], runs[256], strict=True):
        for s64, s, inf in zip(rec64["steps"], rec["steps"], infos, strict=True):
            if inf["think_tokens_e256"] <= 64:
                assert s64["think"] == s["think"]


@needs_k12
def test_cli_level(tmp_path: Path) -> None:
    out = tmp_path / "t.jsonl"
    argv = ["--level", "16", "--limit", "30", "--workers", "2", "--out", str(out)]
    assert pt.main([*argv, "--distractors", ""]) == 0
    recs = [json.loads(x) for x in out.read_text().splitlines()]
    stats = json.loads((tmp_path / "t.stats.json").read_text())
    assert len(recs) > 25 and all(r["gen"]["level"] == 16 for r in recs)
    assert stats["config"]["level"] == 16 and stats["max_think_tokens"] <= 16
    assert stats["other_segments"] == 0 and stats["rungs"]
