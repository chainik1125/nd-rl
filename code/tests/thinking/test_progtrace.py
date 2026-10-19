"""Programmatic traces: strict validity, fixed reference actions, determinism, the v2 profile's
audit fixes (traces/audit/AUDIT.md) and the batch CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from code.thinking import progtrace as pt

from ._pipeline_helpers import K12, k12_head, needs_k12

N = 120


def _traces(
    cfg: pt.Cfg, n: int = N, seed: int = 0
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    return [
        pt.trace_for(f"k12_{i}", r["prompt"], r["proof"], cfg, seed=seed)
        for i, r in enumerate(k12_head(n))
    ]


@needs_k12
def test_v1_and_v2_traces_validate_strictly_and_keep_reference_actions() -> None:
    T = pt.td()
    rows = k12_head(N)
    for cfg in (pt.Cfg.v1(), pt.Cfg()):
        for (rec, infos), r in zip(_traces(cfg), rows, strict=True):
            v = T.validate(rec, lean=False, strict=True)
            assert v["ok"], (rec["id"], v["error"])
            steps, _, _ = T.decompose(r["prompt"], r["proof"], canon=True)
            ref = [pt.ascii_line(a) for _, a, _ in steps]
            assert [s["action"] for s in rec["steps"]] == ref
            assert len(infos) == len(ref)


@needs_k12
def test_generation_is_deterministic_per_seed() -> None:
    a = [rec for rec, _ in _traces(pt.Cfg(), 40)]
    b = [rec for rec, _ in _traces(pt.Cfg(), 40)]
    c = [rec for rec, _ in _traces(pt.Cfg(), 40, seed=1)]
    assert a == b
    assert a != c


@needs_k12
def test_v2_has_no_premise_thinks_and_ex_falso_no_longer_fills_every_block() -> None:
    n_premise_think = n_efq_no = n_blocks = n_leaf = 0
    for rec, infos in _traces(pt.Cfg(), 200):
        for st, inf in zip(rec["steps"], infos, strict=True):
            if inf["kind"] == "premise":
                n_premise_think += bool(st["think"])
            if inf["kind"] == "decision" and st["think"]:
                n_blocks += 1
                n_efq_no += "|- F <no>" in st["think"].split("\n")
                n_leaf += bool(inf.get("leaf_cands"))
    assert n_premise_think == 0
    assert n_blocks > 100
    assert n_efq_no / n_blocks < 0.12  # v1: about 0.2 of decision blocks
    assert n_leaf / n_blocks > 0.4


@needs_k12
def test_v2_leaf_blocks_never_name_the_action_before_its_candidate() -> None:
    """At a leaf decided by candidates, no reduction line has the action's term: the wrong
    candidates are checked before the answer is written."""
    n = 0
    for rec, infos in _traces(pt.Cfg(), 200):
        for st, inf in zip(rec["steps"], infos, strict=True):
            if not inf.get("leaf_cands") or not st["think"]:
                continue
            act = st["action"]
            term = act.split(" := ", 1)[1].rstrip(" ;")
            lines = st["think"].split("\n")
            first = next((i for i, ln in enumerate(lines) if ln == act), len(lines))
            assert not any(
                ln.startswith("|- ") and ln.split(" := ", 1)[-1] == term for ln in lines[:first]
            ), (rec["id"], st["think"])
            n += 1
    assert n > 50


@needs_k12
def test_cli_writes_traces_and_stats(tmp_path: Path) -> None:
    out = tmp_path / "t.jsonl"
    assert pt.main(["--limit", "30", "--workers", "2", "--out", str(out), "--distractors", ""]) == 0
    recs = [json.loads(x) for x in out.read_text().splitlines()]
    stats = json.loads((tmp_path / "t.stats.json").read_text())
    assert len(recs) == stats["counts"]["traces"] and len(recs) > 25
    assert stats["config"]["format"] == "progtrace/2" and stats["config"]["source"] == str(K12)
    assert stats["max_pair_tokens"] < 1024
