"""Auto effort: the model picks its own think level per step (`sample.choose_levels`), the
chosen-effort pair layout (`data.encode_pair_choice`), EI over the choice (`ei_auto`), and the
`eval --budgets auto` rows."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest
import torch

from code.thinking import ei, ei_auto, stats, train
from code.thinking import eval as ev
from code.thinking.data import FORMAT, encode_pair, encode_pair_choice, trace_pairs
from code.thinking.effort import EffortTokenizer, effort_id
from code.thinking.sample import AUTO, choose_levels, decode_wave, think_generate
from code.thinking.tok import D, ThinkTokenizer

from ._pipeline_helpers import ScriptedModel, bias_logits, isolate, pilot_rec, tiny_model

E = EffortTokenizer()
REC = "k12_37120"
CKPT = Path("/scratch/cowifornia/nd-rl/think/runs/ei_pilot/eiS1g3_h712/ckpts/eiS1g3_h712_r1.pt")
POOL = Path("/scratch/cowifornia/nd-rl/think/runs/pools/rl_hard12.jsonl")
LEAN = Path("~/.elan/bin/lean").expanduser()


def _state() -> list[int]:
    return E.encode_toks(D.state_env.Env(pilot_rec(REC)["prompt"], canon=True).state_tokens())


def _biased(token: str, strength: float = 60.0) -> Any:
    b = torch.zeros(E.vocab_size)
    b[E.stoi[token]] = strength
    return bias_logits(tiny_model(E.vocab_size), b)


# ----------------------------------------------------------------------------------- the choice


@pytest.mark.parametrize("token", ["P", "<eos>", "<think>", "<e16>"])
def test_choice_is_always_an_allowed_effort_level(token: str) -> None:
    """Whatever the model prefers, the pick is one of the allowed levels."""
    picks = choose_levels(_biased(token), E, [_state()] * 16, (0, 64, 256), 1.0, seed=3)
    assert set(picks) <= {0, 64, 256}


def test_choice_follows_the_model_within_the_allowed_levels() -> None:
    assert choose_levels(_biased("<e64>"), E, [_state()] * 8, (0, 64, 256), 0.8, seed=1) == [64] * 8
    got = choose_levels(_biased("<e256>"), E, [_state()] * 4, (0, 64, 256), greedy=True)
    assert got == [256] * 4
    # <e64> is not allowed: the pick falls among the allowed ones
    assert set(choose_levels(_biased("<e64>"), E, [_state()] * 16, (0, 256), 1.0)) <= {0, 256}


def test_choice_noise_is_keyed_by_seed_and_row() -> None:
    m = tiny_model(E.vocab_size, seed=5)
    rows = [_state()] * 32
    a = choose_levels(m, E, rows, (0, 64, 256), 5.0, seed=11)
    assert a == choose_levels(m, E, rows, (0, 64, 256), 5.0, seed=11)
    assert len(set(a)) > 1  # a near-flat choice at a high temperature varies across rows
    st: dict[str, Any] = {}
    choose_levels(m, E, rows[:3], (0, 64, 256), 1.0, seed=2, stats=st)
    assert st["sampled_tokens"] == 3 and sum(st["auto_levels"].values()) == 3
    assert st["choice_prefill_tokens"] == 3 * len(rows[0])


# ------------------------------------------------------------------------------- the sampler


def test_auto_prompts_caps_and_accounting(monkeypatch: pytest.MonkeyPatch) -> None:
    """A random model in auto mode: every step prompt ends in an allowed `<eB> <think>` (plus
    `</think>` at 0), think content stays within the picked level (exact forcing), no effort token
    is decoded, and the effort token is a sampled token, not prefill."""
    seen: list[tuple[list[int], int]] = []
    real = decode_wave

    def spy(model: Any, tok: Any, pids: list[list[int]], budgets: list[int], *a: Any, **kw: Any):
        seen.extend(zip(pids, budgets, strict=True))
        return real(model, tok, pids, budgets, *a, **kw)

    monkeypatch.setattr("code.thinking.sample.decode_wave", spy)
    prompts = [pilot_rec(REC)["prompt"]] * 12
    res = think_generate(
        tiny_model(E.vocab_size, seed=7),
        E,
        prompts,
        "plain",
        0,
        temperature=3.0,
        max_action=8,
        max_steps=3,
        batch=12,
        seed=4,
        force_mode="exact",
        auto_levels=(0, 256),
    )
    for p, b in seen:
        tail = [effort_id(E, b), E.think] + ([E.end_think] if b == 0 else [])
        assert b in (0, 256) and p[-len(tail) :] == tail
    for r in res:
        assert r["budget"] == AUTO and len(r["levels"]) == r["draws"] == r["choice"]
        assert set(r["levels"]) <= {0, 256}
        for s, lv in zip(r["trace"], r["levels"], strict=True):
            assert s["level"] == lv and len(s["think"]) <= lv
            assert not set(E.effort_ids) & set(s["gen"])
        assert r["tokens"] == sum(len(s["gen"]) for s in r["trace"]) + r["choice"]
        assert r["prefill"] == sum(len(s["prompt"]) for s in r["trace"]) - r["choice"]
    assert len({lv for r in res for lv in r["levels"]}) == 2  # both levels were picked


def test_scripted_auto_follows_the_picked_level_per_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model scripted to pick a level and then write that step of the pilot trace: the sampler
    picks the scripted level at every step, prefills it, and the proof completes."""
    monkeypatch.setattr(D.state_sample, "NAME_BASE_MAX", 0)
    rec = pilot_rec(REC)
    script: dict[bytes, list[int]] = {}
    want: list[int] = []
    for i, (p, q, tl) in enumerate(trace_pairs(rec, E)):
        content, action = list(q[1 : tl - 1]), list(q[tl:])
        lv = 0 if i % 3 == 0 or not content else (64 if len(content) < 64 else 256)
        block = [E.think, E.end_think] if lv == 0 else [E.think, *content, E.end_think]
        script[p] = [effort_id(E, lv), *block, *action]
        want.append(lv)
    assert {0} < set(want)
    res = think_generate(
        ScriptedModel(E, script),
        E,
        [rec["prompt"]] * 2,
        "plain",
        0,
        max_action=64,
        batch=2,
        force_mode="exact",
        auto_levels=(0, 64, 256),
    )
    for r in res:
        assert r["status"] == "done" and r["levels"] == want and r["forced"] == 0, r["status"]
        assert [s["level"] for s in r["trace"]] == want
        assert r["think"] == sum(len(s["think"]) for s in r["trace"])


def test_auto_rejects_what_it_does_not_support() -> None:
    m = tiny_model(E.vocab_size)
    prompts = [pilot_rec(REC)["prompt"]]
    with pytest.raises(ValueError, match="lean_thinke"):
        think_generate(tiny_model(ThinkTokenizer().vocab_size), ThinkTokenizer(), prompts,
                       auto_levels=(0, 64))  # fmt: skip
    with pytest.raises(ValueError, match="plain arm"):
        think_generate(m, E, prompts, "logical", auto_levels=(0, 64))
    with pytest.raises(ValueError, match="intervention"):
        think_generate(m, E, prompts, "plain", intervene="empty", auto_levels=(0, 64))
    with pytest.raises(ValueError, match="effort levels"):
        think_generate(m, E, prompts, "plain", auto_levels=(0, 32))
    with pytest.raises(ValueError, match="min_think"):
        think_generate(m, E, prompts, "plain", min_think=4, auto_levels=(0, 64))


def test_fixed_levels_are_unchanged_by_the_auto_code() -> None:
    """Without auto_levels the result dicts carry no auto fields and the budget is the level."""
    res = think_generate(
        tiny_model(E.vocab_size, seed=2),
        E,
        [pilot_rec(REC)["prompt"]] * 3,
        "plain",
        64,
        max_action=8,
        max_steps=2,
        seed=1,
    )
    for r in res:
        assert r["budget"] == 64 and "levels" not in r and "choice" not in r
        assert all("level" not in s for s in r["trace"])


# ----------------------------------------------------------------------------- the pair layout


def test_choice_pair_layout_and_training_mask() -> None:
    state, block, action = ["<st>", "⊢", "P", "<act>"], ["<think>", "P", "</think>"], ["exact"]
    p, q, tl = encode_pair_choice(E, state, 64, block, action)
    p0, q0, tl0 = encode_pair(E, [*state, "<e64>"], block, action)
    assert p + q == p0 + q0  # the same tokens: only the input / target split moves
    assert list(p) == E.encode_toks(state) and q[0] == effort_id(E, 64) and q[1] == E.think
    assert tl == 1 + len(block) and tl0 == len(block)
    x, m = train.batch_w([[(p, q, tl)]], [0], E, random.Random(0), "cpu", 1.0)
    assert int(x[0, len(p)]) == effort_id(E, 64)
    assert bool(m[0, len(p)]) and not bool(m[0, len(p) - 1])  # the effort token is a target
    _, mw = train.batch_w([[(p, q, tl)]], [0], E, random.Random(0), "cpu", 0.5)
    assert float(mw[0, len(p)]) == 0.5 and float(mw[0, len(p) + tl]) == 1.0
    with pytest.raises(ValueError):
        encode_pair_choice(ThinkTokenizer(), state, 64, block, action)


def test_entry_pairs_choice_uses_the_step_level() -> None:
    e: dict[str, Any] = {
        "budget": 64,
        "think_format": "think",
        "steps": [
            {"state": "<st> ⊢ P <act>", "think": "n1 : P", "action": "exact n1"},
            {"state": "<st> ⊢ P <act>", "think": "", "action": "exact n1", "level": 0},
        ],
    }
    (p1, q1, n1), (p2, q2, n2) = ei.entry_pairs(e, E, choice=True)
    assert q1[0] == effort_id(E, 64) and q2[0] == effort_id(E, 0)
    assert p1[-1] == E.stoi["<act>"] and n1 == 6 and n2 == 3
    ((p, q, n),) = ei.entry_pairs({**e, "steps": e["steps"][:1]}, E)  # default: unchanged
    assert p[-1] == effort_id(E, 64) and q[0] == E.think and n == 5


# ------------------------------------------------------------------------------- the label rule


def test_choice_label_rule() -> None:
    lab = ei_auto.choice_label
    assert lab([(0, 3, 10, 400.0), (64, 3, 10, 900.0), (256, 1, 10, 1500.0)]) == 0  # tie: cheaper
    assert lab([(0, 2, 10, 400.0), (64, 5, 10, 900.0), (256, 5, 10, 1500.0)]) == 64
    assert lab([(0, 0, 10, 400.0), (64, 0, 10, 900.0), (256, 0, 10, 1500.0)]) is None
    # a token cost: 0.5 - 0.3 * 0.9 = 0.23 < 0.4 - 0.3 * 0.4 = 0.28
    assert lab([(0, 4, 10, 400.0), (64, 5, 10, 900.0)], 0.3) == 0
    assert lab([(0, 4, 10, 400.0), (64, 5, 10, 900.0)], 0.0) == 64
    # a level without a success is never the label, however cheap
    assert lab([(0, 0, 10, 10.0), (256, 1, 10, 5000.0)], 10.0) == 256


def test_kept_entries_are_distinct_shortest_and_capped() -> None:
    def att(tokens: int, actions: list[str], ok: bool = True) -> dict[str, Any]:
        steps = [
            {"state": bytes(E.encode_toks(["<st>", "⊢", "P", "<act>"])), "think": b"",
             "applied": bytes(E.encode_toks(a.split())), "forced": False}
            for a in actions
        ]  # fmt: skip
        return {"ok": ok, "tokens": tokens, "base": 0, "budget": 64, "trace": steps}

    atts = [
        att(50, ["exact n1"]),
        att(30, ["exact n1"]),
        att(40, ["exact n2", "exact n1"]),
        att(10, ["exact n2"], ok=False),
        att(60, ["exact n3"]),
    ]
    got = ei_auto.kept_entries(atts, E, {"name": "t", "prompt": "THM P"}, 1, 2, "keep")
    assert [e["tokens"] for e in got] == [30, 40]
    assert len(ei_auto.kept_entries(atts, E, {"name": "t", "prompt": "THM P"}, 1, 9, "keep")) == 3


# ------------------------------------------------------------------------------ eval rows


def test_auto_rows_load_as_an_arm(tmp_path: Path) -> None:
    def x(ok: bool, levels: list[int], tokens: int) -> dict[str, Any]:
        d: dict[str, Any] = {c: 0 for c in ev.COUNTERS}
        d.update(ok=ok, status="done" if ok else "fail:S:x", steps=1, fail_step=None,
                 budget=AUTO, causes=[], text="t" if ok else None, levels=levels,
                 choice=len(levels), tokens=tokens)  # fmt: skip
        return d

    recs = [
        {"file": "f", "name": "a", "prompt": "THM P"},
        {"file": "f", "name": "b", "prompt": "Q"},
    ]
    res = [x(True, [0, 64], 100), x(False, [256], 300), x(False, [0], 50), x(False, [0], 60)]
    rows = ev.rows_for(recs, res, 2)
    assert rows[0]["levels"] == [[0, 64], [256]] and rows[0]["choice"] == [2, 1]
    assert ev.level_shares(res) == {"0": 60.0, "64": 20.0, "256": 20.0}
    f = tmp_path / "rows_plain_Bauto.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    arm = stats.load_arm("auto", [str(f)])
    assert arm.names == ["a", "b"] and arm.c.tolist() == [[1, 0]] and arm.n.tolist() == [[2, 2]]
    assert arm.tokens is not None and arm.tokens.tolist() == [[400.0, 110.0]]
    assert ev.parse_budgets("0,auto,64", True) == [0, AUTO, 64] and ev.parse_budgets(
        "x", False
    ) == [0]


# ------------------------------------------------------------------------ a round, end to end


@pytest.mark.slow
@pytest.mark.skipif(not (CKPT.exists() and POOL.exists() and LEAN.exists()), reason="no ckpt")
def test_ei_auto_round_then_auto_eval_on_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One EI-over-the-choice round on a handful of short pool targets (CPU, a few fine-tune
    steps), then an auto-effort read of the new checkpoint."""
    isolate(tmp_path, monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    rows = [json.loads(x) for x in POOL.read_text().splitlines() if x.strip()]
    pool = tmp_path / "pool.jsonl"
    pool.write_text(
        "".join(json.dumps(r) + "\n" for r in [r for r in rows if r["n_lines"] == 7][:4])
    )
    train_pkl = tmp_path / "train.pkl"
    import pickle

    pairs = trace_pairs(pilot_rec(REC), E)
    retained = [[(bytes([*p, effort_id(E, 64)]), q, n) for p, q, n in pairs]]
    data = {"format": FORMAT, "mode": E.mode, "itos": list(E.itos), "records": retained,
            "meta": [], "config": {}, "report": {}}  # fmt: skip
    train_pkl.write_bytes(pickle.dumps(data))
    out = tmp_path / "ei"
    hist = ei_auto.run(
        ei_auto.parser().parse_args(
            [
                "--init",
                str(CKPT),
                "--name",
                "A",
                "--out",
                str(out),
                "--targets",
                str(pool),
                "--train",
                str(train_pkl),
                "--rounds",
                "1",
                "--choice_k",
                "2,2,2",
                "--ft_steps",
                "3",
                "--ft_recs",
                "4",
                "--retain",
                "1",
                "--batch",
                "24",
                "--max_steps",
                "24",
                "--device",
                "cpu",
            ]  # fmt: skip
        )
    )
    lab = hist[0]["labels"]
    assert lab["labelled"] + lab["unlabelled"] == 4 and set(lab["labels"]) == {"0", "64", "256"}
    ck = out / "ckpts" / "A_r1.pt"
    if lab["labelled"]:
        assert ck.exists()
        found = [json.loads(x) for x in (out / "found_1.jsonl").read_text().splitlines()]
        assert found and all(e["label"] in (0, 64, 256) for e in found)
    summ = ev.run(
        ev.parser().parse_args(
            [
                "--ckpt",
                str(ck if ck.exists() else CKPT),
                "--sets",
                str(pool),
                "--out",
                str(tmp_path / "ev"),
                "--k",
                "2",
                "--budgets",
                "auto",
                "--arms",
                "plain",
                "--max_steps",
                "24",
                "--device",
                "cpu",
            ]  # fmt: skip
        )
    )
    cell = summ["cells"]["plain_Bauto"]
    assert abs(sum(cell["level_share_pct"].values()) - 100.0) < 1e-6
    arm = stats.load_arm("auto", [str(tmp_path / "ev" / "rows_plain_Bauto.jsonl")])
    assert arm.n.tolist() == [[2, 2, 2, 2]]
