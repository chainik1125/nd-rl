"""Graded effort pairs (`data.build_graded`): level draws, effort tokens, budgets, CLI specs.

Fixtures are programmatic traces (`progtrace.trace_for`); the stand-in level files for 16 / 64
hold the same records with every think emptied (they satisfy any budget)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from code.thinking import progtrace
from code.thinking.data import (
    build_graded,
    draw_level,
    main,
    parse_graded,
)
from code.thinking.effort import effort_id
from code.thinking.tok import EffortTokenizer

from ._pipeline_helpers import k12_head, needs_k12

E = EffortTokenizer()
PROBS = {0: 0.2, 16: 0.2, 64: 0.3, 256: 0.3}


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _emptied(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**r, "steps": [{**s, "think": ""} for s in r["steps"]]} for r in rows]


def _traces(cfg: progtrace.Cfg, n: int = 40) -> list[dict[str, Any]]:
    return [
        progtrace.trace_for(f"k12_{i}", r["prompt"], r["proof"], cfg)[0]
        for i, r in enumerate(k12_head(n))
    ]


def _specs(tmp: Path, top: list[dict[str, Any]], l16: Any = None, l64: Any = None) -> list[Any]:
    l16 = _emptied(top) if l16 is None else l16
    l64 = _emptied(top) if l64 is None else l64
    return [
        (0, None, 0.2),
        (16, _jsonl(tmp / "e16.jsonl", l16), 0.2),
        (64, _jsonl(tmp / "e64.jsonl", l64), 0.3),
        (256, _jsonl(tmp / "e256.jsonl", top), 0.3),
    ]


@pytest.fixture(scope="module")
def top() -> list[dict[str, Any]]:
    return _traces(progtrace.Cfg.v1())


def test_draw_is_deterministic_and_matches_probabilities() -> None:
    specs = [(lv, None if lv == 0 else "p", p) for lv, p in PROBS.items()]
    ids = [f"k12_{i}" for i in range(5000)]
    draws = [draw_level(i, specs) for i in ids]
    assert draws == [draw_level(i, list(reversed(specs))) for i in ids]  # order-free input
    for lv, p in PROBS.items():
        assert abs(draws.count(lv) / len(ids) - p) < 0.02, lv


@needs_k12
def test_pairs_end_in_the_effort_token_and_levels_are_consistent(
    tmp_path: Path, top: list[dict[str, Any]]
) -> None:
    data = build_graded(_specs(tmp_path, top))
    assert data["mode"] == "lean_thinke" and data["itos"] == list(E.itos)
    assert data["no_trace_mode"] == "empty" and data["has_traces"]
    assert len(data["records"]) == len(data["meta"]) == len(top)
    rep = data["report"]
    assert sum(rep["effort_levels"].values()) == rep["traces_valid"] == len(top)
    assert rep["missing_at_level"] == {} and rep["invalid_trace_reasons"] == {}
    specs = [(lv, None, p) for lv, p in PROBS.items()]
    for rec, pairs, m in zip(top, data["records"], data["meta"], strict=True):
        lv = draw_level(rec["id"], specs)
        assert m["level"] == lv and m["id"] == rec["id"] and m["n_lines"] is None
        assert m["source"] == ("trace_emptied" if lv == 0 else "trace")
        assert len(pairs) == len(rec["steps"])
        assert all(p[-1] == effort_id(E, lv) for p, _, _ in pairs)
        assert all(not set(p[:-1]) & set(E.effort_ids) for p, _, _ in pairs)  # one token only
        assert all(E.itos[p[0]] == "<st>" and E.itos[p[-2]] == "<act>" for p, _, _ in pairs)
        if lv == 0:
            assert all(n == 2 for _, _, n in pairs)
        assert all(q[0] == E.think and q[-1] == E.eos for _, q, _ in pairs)
    assert data["records"][0] != []
    assert rep["lengths"]["pairs"] == sum(len(r) for r in data["records"])


@needs_k12
def test_states_decode_to_the_env_state(tmp_path: Path, top: list[dict[str, Any]]) -> None:
    from code.thinking.dan import load

    D = load()
    data = build_graded(_specs(tmp_path, top))
    for rec, pairs in list(zip(top, data["records"], strict=True))[:5]:
        env = D.state_env.Env(rec["prompt"], canon=True, base=0, assign=False)
        for (p, _, _), st in zip(pairs, rec["steps"], strict=True):
            assert [E.itos[i] for i in p[:-1]] == env.state_tokens()
            ok, why = env.apply(_toks(st["action"]))
            assert ok, why


def _toks(action: str) -> list[str]:
    from code.thinking.dan import trace_dsl

    return trace_dsl().tokens(action)


@needs_k12
def test_a_block_over_the_level_budget_is_rejected(
    tmp_path: Path, top: list[dict[str, Any]]
) -> None:
    """Over-budget level files are a broken input: `ValueError`, not a silent drop."""
    assert any(s["think"] for r in top for s in r["steps"])
    # the level-16 stand-in is the full level-256 traces: far over 16 tokens
    specs = _specs(tmp_path, top, l16=top, l64=top)
    with pytest.raises(ValueError, match="exceeds level"):
        build_graded(specs)


@needs_k12
def test_missing_ids_are_counted(tmp_path: Path, top: list[dict[str, Any]]) -> None:
    full = build_graded(_specs(tmp_path, top))
    specs = [(lv, None, p) for lv, p in PROBS.items()]
    drop16 = [r for r in top if draw_level(r["id"], specs) == 16][:3]
    keep = [r for r in _emptied(top) if r["id"] not in {d["id"] for d in drop16}]
    data = build_graded(_specs(tmp_path, top, l16=keep))
    assert data["report"]["missing_at_level"] == {"16": len(drop16)} and len(drop16) > 0
    assert len(data["records"]) == len(full["records"]) - len(drop16)
    assert data["report"]["traces_in"] == len(top)


@needs_k12
def test_a_changed_action_is_rejected(tmp_path: Path, top: list[dict[str, Any]]) -> None:
    bad = _emptied(top)
    for r in bad:
        r["steps"][0] = {**r["steps"][0], "action": "exact n1"}
    with pytest.raises(ValueError, match="differs"):
        build_graded(_specs(tmp_path, top, l16=bad, l64=bad))


def test_bad_specs_raise(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    p.write_text("")
    for bad in (
        [(0, None, 0.2), (16, p, 0.2), (64, p, 0.3), (256, p, 0.2)],  # sum 0.9
        [(0, None, 0.5), (16, p, 0.5)],  # no 256
        [(0, p, 0.5), (256, p, 0.5)],  # level 0 with a path
        [(0, None, 0.5), (256, None, 0.5)],  # level 256 without a path
        [(0, None, 0.5), (0, None, 0.2), (256, p, 0.3)],  # duplicate level
        [(32, p, 0.5), (256, p, 0.5)],  # unknown level
    ):
        with pytest.raises(ValueError):
            build_graded(bad)


def test_parse_graded() -> None:
    assert parse_graded(["0:0.2", "16:/a/e16.jsonl:0.2"]) == [
        (0, None, 0.2),
        (16, "/a/e16.jsonl", 0.2),
    ]
    with pytest.raises(ValueError):
        parse_graded(["16"])


@needs_k12
def test_cli_builds_and_rejects_mixing(
    tmp_path: Path, top: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    sp = _specs(tmp_path, top)
    args = [f"{lv}:{p}:{pr}" if p else f"{lv}:{pr}" for lv, p, pr in sp]
    out = tmp_path / "g.pkl"
    assert main(["--graded", *args, "--no-lean", "--exclude", "--out", str(out)]) == 0
    assert out.exists() and "levels" in capsys.readouterr().out
    rep = json.loads(Path(str(out) + ".report.json").read_text())
    assert rep["config"]["draw"] == "unit_hash(id/effort)" and rep["traces_in"] == len(top)
    with pytest.raises(SystemExit):
        main(["--graded", *args, "--traces", str(sp[3][1]), "--out", str(out)])


@needs_k12
def test_real_level_traces_respect_their_budgets(tmp_path: Path, top: list[dict[str, Any]]) -> None:
    if "level" not in progtrace.Cfg.__dataclass_fields__:
        pytest.skip("progtrace.Cfg has no level")
    l16 = _traces(progtrace.Cfg.v1(level=16))
    l64 = _traces(progtrace.Cfg.v1(level=64))
    data = build_graded(_specs(tmp_path, top, l16=l16, l64=l64))
    assert data["report"]["effort_levels"][16] > 0 and data["report"]["effort_levels"][64] > 0
    for lv in (16, 64):
        assert data["report"]["think_tokens_by_level"][str(lv)]["max"] <= lv
