"""Search traces (`searchtrace.py`): strict validity, determinism, independence from the reference
proof, certified dead ends (the countermodel DSL extension), the per-level budgets, the min_level
labels, and the graded-pickle hook for levels whose proofs differ."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from code.thinking import searchtrace as st
from code.thinking.data import build_graded

from ._pipeline_helpers import k12_head, needs_k12

N = 60
T = st.td()


def _gen(level: Any, n: int = N, **kw: Any) -> list[dict[str, Any]]:
    out = []
    for i, r in enumerate(k12_head(n)):
        rec, _, _ = st.trace_for(f"k12_{i}", r["prompt"], level, **kw)
        out.append(rec)
    return out


@pytest.fixture(scope="module")
def e256() -> list[dict[str, Any]]:
    return _gen(256)


def _body(think: str) -> int:
    toks = T.tokens(think)
    return len(toks)


# ------------------------------------------------------------------------------------ validity
@needs_k12
@pytest.mark.parametrize("level", [0, 16, 64, 256, 1024, st.UNCAPPED, "auto"])
def test_traces_validate_strictly_and_replay_in_the_environment(level: Any) -> None:
    for rec in _gen(level, 40):
        v = T.validate(rec, lean=False, strict=True)  # replays every action in state_env
        assert v["ok"], (rec["id"], level, v["error"])
        assert v["soft"] == []
        assert rec["gen"]["strategy"] in st.STRATEGIES
        assert not any(s in json.dumps(rec["steps"]) for s in st.STRATEGIES)  # never in the text


@needs_k12
def test_record_format_matches_progtrace(e256: list[dict[str, Any]]) -> None:
    for rec in e256:
        assert {"id", "source", "mode", "prompt", "steps", "gen"} <= set(rec)
        for s in rec["steps"]:
            assert isinstance(s["think"], str) and isinstance(s["action"], str)
            assert "min_level" in s


# ------------------------------------------------------------------------------- determinism
@needs_k12
def test_generation_is_deterministic(e256: list[dict[str, Any]]) -> None:
    assert _gen(256) == e256
    assert _gen(64, 20) == _gen(64, 20)


@needs_k12
def test_strategy_is_sampled_from_the_id_alone() -> None:
    ids = [f"k12_{i}" for i in range(400)]
    first = [st.strategy_order(i)[0] for i in ids]
    assert first == [st.strategy_order(i)[0] for i in ids]
    assert set(first) == set(st.STRATEGIES)
    for i in ids[:20]:
        assert sorted(st.strategy_order(i)) == sorted(st.STRATEGIES)


# ------------------------------------------------------------------- the reference is never read
def test_search_entry_points_take_no_reference() -> None:
    for fn in (st.prove, st.Search.__init__, st.gen):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"proof", "reference", "ref", "actions"}, (fn, params)


@needs_k12
def test_traces_are_identical_without_or_with_a_corrupted_reference() -> None:
    rows = k12_head(N)
    junk = "N1 P : PR ; QED"
    for i, r in enumerate(rows):
        other = rows[(i + 7) % len(rows)]["proof"]
        base, _, p_ref = st.trace_for(f"k12_{i}", r["prompt"], 256, reference=r["proof"])
        for ref in (None, other, junk):
            rec, _, _ = st.trace_for(f"k12_{i}", r["prompt"], 256, reference=ref)
            assert rec == base, (i, ref)
        assert "overhead" in p_ref  # the reference is used afterwards, for statistics


# --------------------------------------------------------------------------- certified dead ends
def _goal_lines(rec: dict[str, Any]) -> list[tuple[Any, list[str]]]:
    env = T.Env(rec["prompt"], canon=True, base=0, assign=False)
    out = []
    for s in rec["steps"]:
        for ln in s["think"].split("\n") if s["think"] else []:
            out.append((st.Ctx(env), T.tokens(ln)))
        env.apply(T.tokens(s["action"]))
    return out


@needs_k12
def test_every_dead_end_carries_a_countermodel_that_refutes_it(e256: list[dict[str, Any]]) -> None:
    n_dead = 0
    for rec in e256:
        for ctx, toks in _goal_lines(rec):
            if "<no>" not in toks:
                continue
            n_dead += 1
            q = toks.index("<no>")
            val = T.parse_countermodel(toks[q + 1 :])
            assert val, toks  # e256 writes evidence on every dead end
            k = toks.index("⊢")
            local, i = [], 0
            while i < k:
                a, j = T.pform(toks, i + 2)
                local.append(a)
                i = j
            g, j = T.pform(toks, k + 1)
            if j == q:  # a dead goal: a classical countermodel exists and this one works
                assert not ctx.valid(local, g)
                assert T.refutes(val, ctx.hyps + local, g)
    assert n_dead > 10


def _check(env: Any, lines: list[str], action: str) -> list[str]:
    think = ["<think>", *T.tokens("\n".join(lines)), "</think>"]
    return T.check_step(env, think, T.tokens(action), {})


def test_validator_checks_countermodels() -> None:
    env = T.Env("THM P SEQ ( P v Q ) PRF", canon=True, base=0, assign=False)
    env.apply(T.tokens("have n1 : P := h1 ;"))
    act = "have n2 : ( P v Q ) := Or.inl n1 ;"
    ok = ["|- ( P v Q ) := Or.inr |- Q <no> P ( ~ Q )", "|- ( P v Q ) := Or.inl n1"]
    assert _check(env, ok, act) == []
    assert _check(env, ["|- ( P v Q ) := Or.inr |- Q", "|- Q <no> P ( ~ Q )"], act) == []
    assert _check(env, ["|- ( P v Q ) := Or.inr |- Q <no>"], act) == []  # evidence is optional
    bad = [
        ["|- ( P v Q ) := Or.inr |- Q <no> P Q"],  # Q true: refutes nothing
        ["|- ( P v Q ) := Or.inr |- Q <no> ( ~ P ) ( ~ Q )"],  # premise P false
        ["|- ( P v Q ) := Or.inr |- Q <no> ( ~ Q ) P"],  # atoms out of order
        ["|- ( P v Q ) := Or.inr |- Q <no> P P ( ~ Q )"],  # an atom twice
        ["|- ( P v Q ) := Or.inr |- Q <no> P ( ~ X )"],  # not an atom
        ["|- ( P v Q ) := Or.inl n1 <no> P ( ~ Q )"],  # a sound reduction
        ["|- ( P v Q ) := Or.inr |- Q", "|- Q <no> Q"],
        ["|- ( P v Q ) := Or.inr |- Q", "|- Q <no> ( ~ Q )"],  # P unknown: premise not true
    ]
    for lines in bad:
        with pytest.raises(T.TraceError):
            _check(env, lines, act)


def test_countermodels_are_minimal_kleene_certificates() -> None:
    P, Q, R = ("atom", "P"), ("atom", "Q"), ("atom", "R")
    hyps = [("imp", P, Q), R]
    cm = st.countermodel(hyps, Q)
    assert cm is not None and T.refutes(cm, hyps, Q)
    assert cm == {"P": False, "Q": False, "R": True}
    assert st.countermodel([P], P) is None  # entailed: no certificate
    assert st.fmt_countermodel({"P": True, "S": False}) == ["P", "(", "¬", "S", ")"]


# -------------------------------------------------------------------------------------- budgets
@needs_k12
@pytest.mark.parametrize("level,cap", [(0, 0), (16, 16), (64, 64), (256, 240), (1024, 1024)])
def test_each_level_respects_its_body_cap(level: int, cap: int) -> None:
    assert st.budget_for(level).cap == cap
    for rec in _gen(level, 40):
        for s in rec["steps"]:
            assert _body(s["think"]) <= cap, (rec["id"], level, s["think"])
            if level == 0:
                assert s["think"] == ""


def test_budgets_grow_with_the_level() -> None:
    b = [st.budget_for(x) for x in (16, 64, 256, 1024)]
    assert [x.cap for x in b] == [16, 64, 240, 1024]
    assert [x.scan for x in b] == sorted(x.scan for x in b)
    assert not b[0].evidence and all(x.evidence for x in b[1:])
    assert st.budget_for(256, cap=4096).cap == 4096
    assert st.budget_for(st.UNCAPPED).cap == st.SAFETY_CAP


# ----------------------------------------------------------------------------------- min_level
@needs_k12
def test_min_level_is_the_smallest_viable_level() -> None:
    for i, r in enumerate(k12_head(30)):
        steps, infos = st.prove(r["prompt"], st.make_strategy("dfs", f"k12_{i}"), 256)
        for s, inf in zip(steps, infos, strict=True):
            if inf["kind"] != "decision":
                assert s["min_level"] == 0
                continue
            vl = s["viable_levels"]
            assert vl == sorted(vl) and set(vl) <= set(st.DEFAULT_LEVELS)
            assert s["min_level"] == (vl[0] if vl else None)


@needs_k12
def test_auto_traces_use_each_steps_min_level() -> None:
    for i, r in enumerate(k12_head(30)):
        steps, infos = st.prove(r["prompt"], st.make_strategy("hyp", f"k12_{i}"), "auto")
        for s, inf in zip(steps, infos, strict=True):
            if inf["kind"] == "decision":
                lv = inf["level"]
                assert _body(s["think"]) <= st.budget_for(lv).cap if lv else s["think"] == ""


# ---------------------------------------------------------------------------- graded pickle hook
@needs_k12
def test_graded_build_accepts_levels_with_different_proofs(tmp_path: Path) -> None:
    files = {}
    for lv in (0, 16, 64, 256):
        p = tmp_path / f"e{lv}.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in _gen(lv, 30)))
        files[lv] = p
    specs = [(0, files[0], 0.2), (16, files[16], 0.2), (64, files[64], 0.3), (256, files[256], 0.3)]
    data = build_graded(specs, any_proof=True)
    assert data["report"]["traces_valid"] == 30
    for lv in (16, 64):
        assert data["report"]["think_tokens_by_level"][str(lv)].get("max", 0) <= lv
    with pytest.raises(ValueError):  # the default keeps the same-proof contract
        build_graded(specs)


# ----------------------------------------------------------------- one canonical proof, control arm
@needs_k12
def test_every_level_writes_the_same_proof_by_default(e256: list[dict[str, Any]]) -> None:
    acts = [[s["action"] for s in r["steps"]] for r in e256]
    for lv in (0, 16, 64, st.UNCAPPED):
        assert [[s["action"] for s in r["steps"]] for r in _gen(lv)] == acts, lv


@needs_k12
def test_plain_records_give_the_same_actions_for_the_no_think_control(
    tmp_path: Path, e256: list[dict[str, Any]]
) -> None:
    from code.thinking.data import build

    tr = tmp_path / "e256.jsonl"
    tr.write_text("".join(json.dumps(r) + "\n" for r in e256))
    n = st.emit_plain(tr, tmp_path / "plain.jsonl")
    rows = [json.loads(x) for x in (tmp_path / "plain.jsonl").read_text().splitlines()]
    assert n == len(e256) == len(rows)
    for rec, row in zip(e256, rows, strict=True):
        steps, _, _ = T.decompose(row["prompt"], row["proof"], canon=True)
        assert [st.pt.ascii_line(a) for _, a, _ in steps] == [s["action"] for s in rec["steps"]]
    think = build(traces=[str(tr)], exclude=[], lean=False)
    plain = build(plain=[str(tmp_path / "plain.jsonl")], no_trace_mode="none", exclude=[])
    assert len(think["records"]) == len(plain["records"]) == len(e256)
    for a, b in zip(think["records"], plain["records"], strict=True):
        assert [p for p, _, _ in a] == [p for p, _, _ in b]  # the same states, step by step
