"""Effort conditioning: the lean_thinke vocabulary, the checkpoint expansion, relabelled Stage-1
records, the sampler's effort prefill and ban, and EI's effort pairs and shortest selection."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest
import torch

from code.thinking import ckpt as C
from code.thinking import ei
from code.thinking.data import trace_pairs
from code.thinking.effort import (
    EFFORT_TOKENS,
    LEVELS,
    EffortTokenizer,
    effort_id,
    expand_to_effort,
    has_effort,
    level_for,
    record_level,
    relabel_records,
)
from code.thinking.sample import decode_wave, think_generate
from code.thinking.tok import D, ThinkTokenizer, make_tokenizer, rebase_names, remap_ids

from ._pipeline_helpers import bias_logits, isolate, pilot_rec, tiny_model

OLD = ThinkTokenizer()
E = EffortTokenizer()


def test_vocabulary_layout_and_names() -> None:
    assert E.vocab_size == 118 and E.ref0 == 54 and E.itos[54] == "n1" and E.itos[-1] == "n64"
    assert [E.itos[i] for i in E.effort_ids] == list(EFFORT_TOKENS) and max(E.effort_ids) < E.ref0
    assert E.itos[: OLD.stoi["h1"]] == OLD.itos[: OLD.stoi["h1"]]  # ids below h1 unchanged
    assert make_tokenizer("lean_thinke").itos == E.itos and has_effort(E) and not has_effort(OLD)
    ids = [*E.effort_ids, E.stoi["n5"]]
    assert rebase_names(ids, E, 3) == [*E.effort_ids, E.stoi["n2"]]  # effort ids are not names


def test_levels() -> None:
    assert [level_for(x) for x in (0, 1, 16, 17, 64, 65, 240, 999)] == [
        0,
        16,
        16,
        64,
        64,
        256,
        256,
        256,
    ]
    assert effort_id(E, 64) == E.stoi["<e64>"]
    with pytest.raises(ValueError):
        effort_id(E, 32)


def test_expand_keeps_old_logits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    isolate(tmp_path, monkeypatch)
    old = tiny_model(OLD.vocab_size, seed=3)
    p = tmp_path / "old.pt"
    C.save_ckpt(p, old, OLD.mode)
    q = expand_to_effort(p, tmp_path / "new.pt", seed=1)
    new, tok, extra = C.load_ckpt(q)
    assert tok.mode == "lean_thinke" and new.cfg["vocab"] == 118 and extra["effort_from"] == str(p)
    r = torch.tensor(remap_ids(OLD, tok))
    torch.manual_seed(7)
    x = torch.randint(0, OLD.vocab_size, (3, 25))
    with torch.no_grad():
        assert torch.allclose(new(r[x])[..., r], old(x), atol=1e-5)


def test_relabel_records_levels_and_ids() -> None:
    rec = pilot_rec("k12_37120")
    full = trace_pairs(rec, OLD)
    empty = trace_pairs({**rec, "steps": [{**s, "think": ""} for s in rec["steps"]]}, OLD)
    data = {"itos": list(OLD.itos), "mode": OLD.mode, "records": [full, empty], "report": {}}
    out = relabel_records(data)
    assert out["mode"] == "lean_thinke" and out["itos"] == list(E.itos)
    lv = record_level(full)
    assert lv in LEVELS and lv > 0 and record_level(empty) == 0
    assert out["report"]["effort_levels"][lv] == 1 and out["report"]["effort_levels"][0] == 1
    r = remap_ids(OLD, E)
    for (p0, q0, n0), (p1, q1, n1) in zip(full, out["records"][0], strict=True):
        assert list(p1) == [r[x] for x in p0] + [effort_id(E, lv)]
        assert list(q1) == [r[x] for x in q0] and n1 == n0
    assert all(p[-1] == effort_id(E, 0) for p, _, _ in out["records"][1])


def _effort_model(token: str) -> Any:
    b = torch.zeros(E.vocab_size)
    b[E.stoi[token]] = 60.0
    return bias_logits(tiny_model(E.vocab_size), b)


def test_effort_tokens_are_never_sampled() -> None:
    st = [
        E.encode_toks(D.state_env.Env(pilot_rec("k12_37120")["prompt"], canon=True).state_tokens())
    ]
    pids = [st[0] + [effort_id(E, 16), E.think]]
    for tokname in EFFORT_TOKENS:
        w = decode_wave(_effort_model(tokname), E, pids, [16], True, max_action=8, seed=1)
        assert not set(E.effort_ids) & set(w.ids[0])


def test_sampler_prefills_the_attempt_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[int]] = []
    real = decode_wave

    def spy(model: Any, tok: Any, pids: list[list[int]], *a: Any, **kw: Any) -> Any:
        seen.extend(pids)
        return real(model, tok, pids, *a, **kw)

    monkeypatch.setattr("code.thinking.sample.decode_wave", spy)
    prompts = [pilot_rec("k12_37120")["prompt"]] * 8
    think_generate(
        _effort_model("P"),
        E,
        prompts,
        "plain",
        [0, 16, 64, 256],
        0,
        True,
        0.8,
        False,
        8,
        2,
        1,
        8,
        3,
        {},
    )
    for p in seen:
        j = p.index(E.think)
        assert p[j - 1] in E.effort_ids
        if p[j - 1] == effort_id(E, 0):
            assert p[j + 1] == E.end_think
    assert {p[p.index(E.think) - 1] for p in seen} <= set(E.effort_ids)


def test_entry_pairs_end_the_state_with_the_effort_token() -> None:
    e = {
        "budget": 64,
        "think_format": "think",
        "steps": [{"state": "<st> ⊢ P <act>", "think": "n1 : P", "action": "exact n1"}],
    }
    ((p, q, n),) = ei.entry_pairs(e, E)
    assert p[-1] == effort_id(E, 64) and q[0] == E.think and n == 5


def test_pick_shortest_spreads_over_budgets() -> None:
    fs = [
        {"budget": 256, "tokens": 300},
        {"budget": 256, "tokens": 120},
        {"budget": 16, "tokens": 90},
        {"budget": 16, "tokens": 40},
        {"budget": 64, "tokens": 70},
    ]
    got = ei.pick(fs, 4, "shortest", random.Random(0))
    assert [(x["budget"], x["tokens"]) for x in got] == [(16, 40), (64, 70), (256, 120), (16, 90)]
    assert len(ei.pick(fs, 2, "random", random.Random(0))) == 2
