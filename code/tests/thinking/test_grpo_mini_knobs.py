"""The mini-run knobs in grpo.py (Leon 2026-10-16): --loss_tokens trains one phase's scored tokens
(the divisor counts those alone; the entropy and the think ban mass stay over all), --adv_sign keeps
one sign of the outcome advantage (the zeroed items stay in the divisor, the step penalty stays).
Given at their defaults they change nothing but args.json: test_grpo_pilots.py's c4b8548 digests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from code.thinking import grpo
from code.thinking.sample import phase_bans

from ._grpo_digest import CONFIGS, run_digest, setup_go
from ._pipeline_helpers import tiny_model
from .test_grpo_pilots import EXTRA, GOLDEN
from .test_grpo_verifier import N_THINK, P, T
from .test_grpo_verifier import _att as _vatt

IDS = (T.think, T.end_think)


def test_the_defaults_given_change_nothing_but_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    flags = ["--loss_tokens", "all", "--adv_sign", "both"]
    monkeypatch.setitem(CONFIGS, "knobs", lambda iso: [*EXTRA["plain"](iso), *flags])
    got = run_digest(tmp_path, monkeypatch, "knobs")
    same = ("rollouts", "rows", "lines", "weights")
    assert {k: got[k] for k in same} == {k: GOLDEN["plain"][k] for k in same}
    assert got["args"] != GOLDEN["plain"]["args"]


def _items() -> list[grpo.Item]:
    """Two attempts' items with think blocks (N_THINK tokens each) and a budget-0 step."""
    att = [_vatt(True, []), _vatt(False, [], end_rejected=True)]
    items, _ = grpo.group_items(att, T, 0.0, "fail")
    assert items is not None
    b0 = (
        list(bytes([T.stoi["P"], T.think, T.end_think])),
        [T.stoi["Q"], T.eos],
        [True, True],
        0.25,
    )
    return [*items, b0]


def _only(items: list[grpo.Item], phase: str) -> list[grpo.Item]:
    out = []
    for p, g, m, a in items:
        n = grpo.think_len(p, g, IDS)
        out.append((p, g, [x and (k < n) == (phase == "think") for k, x in enumerate(m)], a))
    return out


def _grads(model: Any) -> list[torch.Tensor]:
    return [
        x.grad.clone() if x.grad is not None else torch.zeros_like(x) for x in model.parameters()
    ]


@pytest.mark.parametrize("phase", ["action", "think"])
@pytest.mark.parametrize("ppo", [False, True])
def test_loss_tokens_is_the_loss_of_that_phase_alone(phase: str, ppo: bool) -> None:
    """--loss_tokens PHASE: the loss and its gradient are the full loss's on items whose masks keep
    that phase alone (its divisor: those tokens); the entropy and the think ban mass stay over every
    scored token."""
    items = _items()
    assert [grpo.think_len(p, g, IDS) for p, g, _, _ in items] == [N_THINK] * 3 + [0]
    model = tiny_model(T.vocab_size, seed=7)
    bans = phase_bans(T, T.vocab_size, True)
    old = grpo.old_logps(model, items, 1.0, "cpu", think_ids=IDS, bans=bans) if ppo else None
    kw: dict[str, Any] = {"think_ids": IDS, "bans": bans}
    model.zero_grad()
    got = grpo.policy_loss(model, items, 1.0, "cpu", old=old, tokens=phase, **kw)
    g_got = _grads(model)
    sel = _only(items, phase)
    old_sel = None
    if old is not None:  # the old log-probs of the kept tokens, in mask order
        old_sel = []
        for (_, _, m, _), (_, _, ms, _), o in zip(items, sel, old, strict=True):
            keep = [k for k, x in enumerate(x for x in m if x)]
            pos = [ms[k] for k, x in enumerate(m) if x]
            old_sel.append(o[[k for k, y in zip(keep, pos, strict=True) if y]])
    model.zero_grad()
    want = grpo.policy_loss(model, sel, 1.0, "cpu", old=old_sel, **kw)
    g_want = _grads(model)
    assert got.tokens == want.tokens == sum(sum(m) for _, _, m, _ in sel)
    assert got.loss == pytest.approx(want.loss, rel=1e-6, abs=1e-9)
    assert all(torch.allclose(a, b, atol=1e-7) for a, b in zip(g_got, g_want, strict=True))
    full = grpo.policy_loss(model, items, 1.0, "cpu", old=old, **kw)
    assert got.ban_think == pytest.approx(full.ban_think) and got.tokens_think == full.tokens_think
    assert got.entropy == pytest.approx(full.entropy)
    assert got.entropy_action == pytest.approx(full.entropy_action)
    # the logged entropies are weighted by every scored token (`tokens_all`), not the trained ones
    assert got.tokens_all == full.tokens_all == full.tokens > got.tokens


def test_loss_tokens_think_at_budget_0_trains_nothing() -> None:
    b0 = _items()[-1:]
    model = tiny_model(T.vocab_size, seed=8)
    assert (
        grpo.policy_loss(model, b0, 1.0, "cpu", think_ids=IDS, tokens="think") == grpo.LossStats()
    )


def _oatt(i: int, ok: bool) -> dict[str, Any]:
    step = {"prompt": bytes([5, 6]), "gen": bytes([10 + i, 11 + i]), "forced": False}
    return {"status": "done" if ok else "fail:S:bad", "ok": ok, "tokens": 2, "trace": [step]}


def test_adv_sign_keeps_one_sign_of_the_outcome() -> None:
    att = [_oatt(0, True), _oatt(1, True), _oatt(2, False), _oatt(3, False)]
    base = [x[3] for x in grpo.group_items(att, T, 0.0, "fail")[0] or []]
    assert base == pytest.approx([0.5, 0.5, -0.5, -0.5])
    for sign, want in (("pos", [0.5, 0.5, 0.0, 0.0]), ("neg", [0.0, 0.0, -0.5, -0.5])):
        items, _ = grpo.group_items(att, T, 0.0, "fail", adv_sign=sign)
        assert [x[3] for x in items or []] == pytest.approx(want)  # every item stays
    # the unlikeliness deduction comes first: in a mostly solved group it can make a solved
    # attempt's advantage negative (rewards 0.7, 1, 1, 0: mean 0.675)
    att3 = [_oatt(0, True), _oatt(1, True), _oatt(2, True), _oatt(3, False)]
    pen = [0.3, 0.0, 0.0, 0.0]
    got = {
        s: [x[3] for x in grpo.group_items(att3, T, 0.0, "fail", unlikely=pen, adv_sign=s)[0] or []]
        for s in ("both", "pos", "neg")
    }
    assert got["both"] == pytest.approx([0.025, 0.325, 0.325, -0.675])
    assert got["pos"] == pytest.approx([0.025, 0.325, 0.325, 0.0])
    assert got["neg"] == pytest.approx([0.0, 0.0, 0.0, -0.675])
    # an all-solved group stays skipped
    assert (
        grpo.group_items([_oatt(i, True) for i in range(4)], T, 0.0, "fail", adv_sign="pos")[0]
        is None
    )


@pytest.mark.parametrize("sign", ["pos", "neg"])
def test_adv_sign_leaves_the_step_penalty(sign: str) -> None:
    """The step penalty is not an outcome advantage: its -P stays on the redrawn draws and on top of
    the draw that ended a failed attempt."""
    att = [_vatt(True, [0]), _vatt(False, [1], end_rejected=True)]
    src: list[tuple[int, str]] = []
    items, _ = grpo.group_items(att, T, 0.0, "fail", rej_penalty=P, src=src, adv_sign=sign)
    assert src == [(0, "acc"), (0, "rej"), (1, "acc"), (1, "acc"), (1, "rej")]
    a = [x[3] for x in items or []]
    if sign == "pos":
        assert a == pytest.approx([0.5, -P, 0.0, -P, -P])
    else:
        assert a == pytest.approx([0.0, -P, -0.5, -0.5 - P, -P])


def test_a_run_records_the_knobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    go = setup_go(tmp_path, monkeypatch)
    hist, out = go(["--iters", "2", "--loss_tokens", "action", "--adv_sign", "pos", "--seed", "1"])
    args = json.loads((out / "args.json").read_text())
    assert (args["loss_tokens"], args["adv_sign"]) == ("action", "pos") and len(hist) == 2
    _, out2 = go(["--iters", "1"], out="plain")
    args2 = json.loads((out2 / "args.json").read_text())
    assert "loss_tokens" not in args2 and "adv_sign" not in args2
