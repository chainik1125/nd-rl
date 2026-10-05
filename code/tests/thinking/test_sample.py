"""Think-aware samplers: budgets, think-block masks, what reaches Env.apply, token accounting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from code.thinking.data import trace_pairs
from code.thinking.sample import (
    attempt_budget,
    decode_wave,
    env_generate_think,
    judge,
    think_generate,
)
from code.thinking.tok import D, ThinkTokenizer

from ._pipeline_helpers import (
    ScriptedModel,
    bias_logits,
    isolate,
    needs_lean,
    pilot,
    pilot_rec,
    script_from_pairs,
    tiny_model,
)

T = ThinkTokenizer()
PROMPTS = [pilot_rec("k12_37120")["prompt"], pilot_rec("k12_22530")["prompt"]]


def _biased(token: str, strength: float = 50.0) -> Any:
    b = torch.zeros(T.vocab_size)
    b[T.stoi[token]] = strength
    return bias_logits(tiny_model(T.vocab_size), b)


def _wave(model: Any, budgets: list[int], min_think: int = 0, **kw: Any) -> Any:
    states = [T.encode_toks(D.state_env.Env(p, canon=True).state_tokens()) for p in PROMPTS]
    pids = [
        s + [T.think] + ([T.end_think] if b == 0 else [])
        for s, b in zip(states, budgets, strict=True)
    ]
    return decode_wave(
        model, T, pids, budgets, True, min_think=min_think, max_action=16, seed=1, **kw
    )


def test_min_budget_holds_back_the_close() -> None:
    w = _wave(_biased("</think>"), [6, 6], min_think=3)
    for ids in w.ids:
        assert ids.index(T.end_think) == 3 and T.eos not in ids[:3]
    w = _wave(_biased("</think>"), [6, 6], min_think=0)
    assert all(ids[0] == T.end_think for ids in w.ids)


def test_max_budget_forces_the_close() -> None:
    w = _wave(_biased("P"), [5, 2])
    assert [ids.index(T.end_think) for ids in w.ids] == [5, 2]
    assert w.forced == [True, True]
    for ids in w.ids:
        j = ids.index(T.end_think)
        assert ids[:j] == [T.stoi["P"]] * j


def test_no_eos_or_nested_think_inside_a_block() -> None:
    for tokname in ("<eos>", "<think>", "<pad>"):
        w = _wave(_biased(tokname), [7, 7])
        for ids in w.ids:
            j = ids.index(T.end_think)
            assert j == 7 and not ({T.eos, T.think, T.pad} & set(ids[:j]))


def test_budget_zero_starts_with_the_action() -> None:
    w = _wave(_biased("<eos>"), [0, 0])
    assert w.ids == [[T.eos], [T.eos]] and w.forced == [False, False]


def test_env_apply_receives_the_action_only(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []
    orig = D.state_env.Env.apply

    def spy(self: Any, act: Any) -> Any:
        seen.append(list(act))
        return orig(self, act)

    monkeypatch.setattr(D.state_env.Env, "apply", spy)
    res = think_generate(
        tiny_model(T.vocab_size),
        T,
        PROMPTS * 3,
        "structural",
        budgets=[3, 8],
        min_think=2,
        max_action=24,
        max_steps=4,
        max_rej=3,
        batch=4,
        seed=2,
    )
    assert seen and sum(r["think"] for r in res) >= 2 * len(res)
    for act in seen:
        assert not ({"<think>", "</think>", "<no>"} & set(act)), act


def test_redraw_token_accounting() -> None:
    st: dict[str, Any] = {}
    res = think_generate(
        tiny_model(T.vocab_size, seed=5),
        T,
        PROMPTS * 4,
        "structural",
        budgets=[0, 4, 9],
        max_action=20,
        max_steps=6,
        max_rej=4,
        batch=3,
        seed=3,
        stats=st,
    )
    assert sum(r["tokens"] for r in res) == st["sampled_tokens"]
    for r in res:
        assert r["tokens"] == r["think"] + r["close"] + r["action"]
        assert r["think"] <= r["draws"] * r["budget"] and r["forced"] <= r["draws"]
        assert r["draws"] == r["steps"] + len(r["causes"])  # accepted + rejected draws
        assert 0 <= r["rejected"] <= r["tokens"]
        if r["steps"] == 0:
            assert r["rejected"] == r["tokens"]
        if r["status"] == "fail:max_rej":
            assert r["rej"] == 4 and r["draws"] == r["steps"] + 5
        assert r["prefill"] >= r["draws"]
    assert st["redraws"] == sum(r["rej"] for r in res)
    assert [r["i"] for r in res] == list(range(len(res)))  # back in prompt order
    assert [r["budget"] for r in res] == [attempt_budget([0, 4, 9], 3, i) for i in range(len(res))]
    assert len({r["budget"] for r in res}) > 1


def test_logical_arm_and_plain_arm_stats() -> None:
    m = tiny_model(T.vocab_size, seed=6)
    for arm in ("plain", "logical"):
        st: dict[str, Any] = {}
        res = think_generate(
            m, T, PROMPTS, arm, budgets=4, max_action=16, max_steps=4, batch=2, stats=st
        )
        assert len(res) == 2 and all(r["status"] for r in res)
        if arm == "plain":
            assert all(
                r["rej"] == 0 and r["draws"] == r["steps"] + (r["status"] != "done") for r in res
            )


def _scripted(rid: str) -> tuple[ScriptedModel, dict[str, Any]]:
    rec = pilot_rec(rid)
    return ScriptedModel(T, script_from_pairs(trace_pairs(rec, T))), rec


@needs_lean
def test_scripted_trace_end_to_end_with_lean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model that writes the pilot trace: the sampler replays it, Lean accepts, the trace is
    kept."""
    isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(D.state_sample, "NAME_BASE_MAX", 0)
    for rec in pilot()[:3]:
        model = ScriptedModel(T, script_from_pairs(trace_pairs(rec, T)))
        res = think_generate(
            model, T, [rec["prompt"]] * 2, "plain", budgets=256, max_action=64, batch=1
        )
        res = judge(T, [rec["prompt"]] * 2, res)
        for r in res:
            assert r["status"] == "done" and r["ok"], r
            pairs = trace_pairs(rec, T)
            assert [bytes(s["state"]) for s in r["trace"]] == [p for p, _, _ in pairs]
            assert [list(s["think"]) for s in r["trace"]] == [
                list(q[1 : tl - 1]) for _, q, tl in pairs
            ]
            assert [list(s["applied"]) for s in r["trace"]] == [
                list(q[tl:-1]) for _, q, tl in pairs
            ]
            assert r["think"] == sum(tl - 2 for _, _, tl in pairs) and r["forced"] == 0
    model, rec = _scripted("k12_37120")
    nds = env_generate_think(model, T, [rec["prompt"]], budget=256)
    assert not nds[0].startswith("LEAN")


def test_scripted_trace_cut_by_a_small_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(D.state_sample, "NAME_BASE_MAX", 0)
    model, rec = _scripted("k12_37120")
    res = think_generate(model, T, [rec["prompt"]], "plain", budgets=4, max_action=64, batch=1)
    r = res[0]
    assert r["status"] != "done" and r["forced"] >= 1 and r["fail_step"] == r["steps"]


def test_no_think_mode_with_a_lean_staten_tokenizer() -> None:
    old = D.lean_tok.LeanTokenizer("lean_staten")
    res = think_generate(
        tiny_model(old.vocab_size), old, PROMPTS, "plain", max_action=16, max_steps=3
    )
    assert all(r["think"] == 0 and r["close"] == 0 and r["budget"] == 0 for r in res)
    with pytest.raises(AssertionError):
        think_generate(tiny_model(old.vocab_size), old, PROMPTS, think=True)
