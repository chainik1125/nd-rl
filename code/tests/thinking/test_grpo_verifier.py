"""GRPO with the verifier: a guided arm keeps only accepted draws in the proof, redrawn draws are
recorded only on request, every rejected draw gets -P whatever the group's outcome, the skip rule,
the monitor's keys and shares, two guided and penalised iterations end to end, and a guided EI
round that fine-tunes on accepted proofs only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from code.thinking import ei, grpo
from code.thinking.data import trace_pairs
from code.thinking.sample import think_generate
from code.thinking.tok import D, ThinkTokenizer

from ._pipeline_helpers import (
    ScriptedModel,
    bias_logits,
    isolate,
    needs_lean,
    pilot_rec,
    script_from_pairs,
    tiny_model,
)
from .test_grpo import _tiny_run

T = ThinkTokenizer()
P = 0.3


def _guided(keep: bool, arm: str = "structural") -> list[dict[str, Any]]:
    b = torch.zeros(T.vocab_size)
    b[T.eos] = 3.0  # actions end within max_action, so the environment judges them
    model = bias_logits(tiny_model(T.vocab_size, seed=3), b)
    return think_generate(
        model,
        T,
        [pilot_rec("k12_37120")["prompt"]] * 6,
        arm,
        budgets=16,
        max_action=8,
        max_steps=3,
        max_rej=3,
        batch=6,
        seed=5,
        keep_rejected=keep,
    )


@pytest.mark.parametrize("arm", ["structural", "logical"])
def test_redrawn_draws_are_recorded_only_on_request(arm: str) -> None:
    plain, kept = _guided(False, arm), _guided(True, arm)
    assert all("redrawn" not in x for x in plain)
    assert [{k: v for k, v in x.items() if k != "redrawn"} for x in kept] == plain
    assert sum(x["rej"] for x in kept) > 0, "the tiny model should be rejected and redrawn"
    for x in kept:
        assert len(x["redrawn"]) == x["rej"]  # every redraw, in order; the last rejection ends
        assert all(r["cause"] for r in x["redrawn"]) and x["causes"][: x["rej"]] == [
            r["cause"] for r in x["redrawn"]
        ]
        accepted = [s for s in x["trace"] if not s.get("rejected")]
        assert len(accepted) == x["steps"]  # only accepted draws are proof steps
        assert all("applied" in s for s in accepted)
        ends = [s for s in x["trace"] if s.get("rejected")]
        assert len(ends) <= 1 and (not ends or x["trace"][-1] is ends[0])


def _step(think: list[int], action: list[int], **kw: Any) -> dict[str, Any]:
    state = [T.stoi["P"], T.stoi["Q"]]
    return dict(
        prompt=bytes([*state, T.think]),
        gen=bytes([*think, T.end_think, *action, T.eos]),
        forced=False,
        **kw,
    )


TH, ACT, BAD = [T.stoi["P"]] * 3, [T.stoi["Q"]] * 2, [T.stoi["R"]] * 2
N_THINK = len(TH) + 1  # a draw's think block and its </think>


def _att(ok: bool, redrawn: list[int], end_rejected: bool = False) -> dict[str, Any]:
    """One accepted step; optionally a rejected draw that ended the attempt (at step 1); one
    redrawn draw per entry of `redrawn`, at that step (its think as the accepted draw's, its
    action BAD)."""
    trace = [_step(TH, ACT, applied=b"")]
    if end_rejected:
        trace.append(_step(TH, BAD, rejected=True))
    return dict(
        ok=ok,
        tokens=10,
        status="done" if ok else ("fail:max_rej" if end_rejected else "fail:lean"),
        trace=trace,
        redrawn=[_step(TH, BAD, cause="S:x", step=t) for t in redrawn],
    )


def test_rejected_draws_get_minus_p() -> None:
    att = [_att(True, [0]), _att(False, [1, 1], end_rejected=True)]
    src: list[tuple[int, str]] = []
    items, masked = grpo.group_items(att, T, 0.0, "fail", rej_penalty=P, src=src)
    assert items is not None and masked == [False, False]
    advs = [round(it[3], 6) for it in items]
    # attempt 0: its step at +0.5, its redrawn draw at -P; attempt 1: its step at -0.5, the draw
    # that ended it at -0.5 - P, its two redrawn draws at -P
    assert advs == [0.5, -P, -0.5, round(-0.5 - P, 6), -P, -P]
    assert src == [(0, "acc"), (0, "rej"), (1, "acc"), (1, "acc"), (1, "rej"), (1, "rej")]
    m0, m1, m2 = items[1][2], items[4][2], items[5][2]
    # 'diverge' (default): the step-0 redraw shares its think block with the draw accepted there,
    # so only its action is scored; the step-1 redraws have no accepted draw: all of each
    assert not any(m0[:N_THINK]) and all(m0[N_THINK:]) and all(m1) and all(m2)
    for scope, want in (("draft", "all"), ("action", "action")):
        its, _ = grpo.group_items(att, T, 0.0, "fail", rej_penalty=P, rej_scope=scope)
        assert its is not None
        for it, kind in zip(its, [k for _, k in src], strict=True):
            m = it[2]
            if kind == "rej" and want == "action":
                assert not any(m[:N_THINK]) and all(m[N_THINK:])
            else:
                assert all(m)
    # P = 0: exactly today's items, no redrawn draw scored
    base, _ = grpo.group_items(att, T, 0.0, "fail")
    zero, _ = grpo.group_items(att, T, 0.0, "fail", rej_penalty=0.0)
    assert base == zero and base is not None and len(base) == 3


def test_a_redraw_identical_to_the_accepted_draw_is_not_penalised() -> None:
    x = _att(True, [])
    x["redrawn"] = [_step(TH, ACT, cause="S:x", step=0)]  # nothing departs from the accepted draw
    assert grpo.rejected_items(x, P, T) == []
    assert len(grpo.rejected_items(x, P, T, "draft")) == 1


def test_skip_rule() -> None:
    # no outcome signal and no rejected draw: skipped, with or without the penalty
    assert (
        grpo.group_items([_att(False, []), _att(False, [])], T, 0.0, "fail", rej_penalty=P)[0]
        is None
    )
    # all failed, with rejected draws: only the penalised draws are scored
    att = [_att(False, [0]), _att(False, [], end_rejected=True)]
    src: list[tuple[int, str]] = []
    items, _ = grpo.group_items(att, T, 0.0, "fail", rej_penalty=P, src=src)
    assert items is not None and [round(it[3], 6) for it in items] == [-P, -P]
    assert src == [(0, "rej"), (1, "acc")]  # the draw that ended attempt 1 is its own step
    assert grpo.group_items(att, T, 0.0, "fail")[0] is None  # today's rule without the penalty
    # all solved, with a redrawn draw: only that draw
    items, _ = grpo.group_items([_att(True, [0]), _att(True, [])], T, 0.0, "fail", rej_penalty=P)
    assert items is not None and [it[3] for it in items] == [-P]


def test_monitor_metrics() -> None:
    per = [
        [_att(True, [0]), _att(False, [1, 1], end_rejected=True)],
        [_att(True, []), _att(True, [])],
    ]
    per[0][0].update(draws=2, causes=["S:binder mismatch"], rejected=4, tokens=20)
    per[0][1].update(draws=4, causes=["S:binder mismatch", "L:app-arg", "S:unbound"], tokens=30)
    for x in per[1]:
        x.update(draws=1, causes=[], tokens=10)
    items: list[Any] = []
    src: list[tuple[int, int, str]] = []
    for g, att in enumerate(per):
        s: list[tuple[int, str]] = []
        its, _ = grpo.group_items(att, T, 0.0, "fail", rej_penalty=P, src=s)
        if its is not None:
            items += its
            src += [(g, j, k) for j, k in s]
    mo = grpo.monitor_metrics(per, [10, 13], [False, True], items, src, (T.think, T.end_think))
    assert set(mo["by_len"]) == {"10", "13"} and set(mo["by_pass"]) == {"(.25,.5]", "1"}
    assert mo["by_len"]["10"]["grad_share"] == pytest.approx(1.0)  # group 1 is skipped
    assert mo["by_len"]["13"] == {
        "groups": 1,
        "with_grad": 0.0,
        "pass": 1.0,
        "grad_share": 0.0,
        "think_share": 0.0,
        "rej_share": 0.0,
    }
    assert mo["mass_think"] + mo["mass_action"] == pytest.approx(1.0)
    assert mo["mass_rejected"] + mo["mass_accepted"] == pytest.approx(1.0)
    assert 0 < mo["mass_rejected"] < 1
    assert mo["rej_per_draw"] == pytest.approx(4 / 8) and mo["rej_per_attempt"] == 1.0
    assert mo["rej_causes"]["S:binder mismatch"] == pytest.approx(0.5)
    assert mo["rejected_token_share"] == pytest.approx(4 / 70)
    # one redrawn draw found the draw accepted at its state (step 0), two had none (step 1)
    assert mo["redrawn_drafts"] == 3 and mo["diverge_match"] == pytest.approx(1 / 3)
    assert mo["shared_prefix"] == pytest.approx(N_THINK / (N_THINK + len(BAD) + 1))
    assert grpo.pass_band(0.0) == "0" and grpo.pass_band(0.25) == "(0,.25]"
    assert grpo.pass_band(0.5) == "(.25,.5]" and grpo.pass_band(0.9) == "(.75,1)"
    assert grpo.target_length({"min_lines": 13, "n_lines": 9}) == 13
    assert grpo.target_length({"n_lines": 9}) == 9 and grpo.target_length({}) is None


def test_guided_penalised_iterations_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--iters", "2", "--arm", "logical", "--max_rej", "3", "--rej_penalty", "0.25"]
    hist, out, _ = _tiny_run(tmp_path, monkeypatch, [*argv, "--monitor"])
    assert len(hist) == 2
    for h in hist:
        mo = h["monitor"]
        assert mo["arm"] == "logical" and mo["rej_penalty"] == 0.25
        assert mo["rej_per_draw"] > 0 and mo["mass_rejected"] > 0
        assert sum(v["grad_share"] for v in mo["by_len"].values()) == pytest.approx(1.0)
        assert h["nonfinite"] == 0 and h["trained_tokens"] > 0
        assert 0 < h["tokens_accepted_share"] <= 1 and 0 <= h["clip_low_accepted"] <= 1
        assert mo["redrawn_drafts"] == 0 or 0 <= mo["diverge_match"] <= 1
    printed = capsys.readouterr().out
    assert "mon it 1:" in printed and "rejected-draw" in printed
    args = json.loads((out / "args.json").read_text())
    assert (args["arm"], args["rej_penalty"], args["max_rej"]) == ("logical", 0.25, 3)


@needs_lean
def test_guided_ei_round_trains_on_accepted_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One frozen EI round with the logical arm: the found records hold accepted steps only."""
    iso_env = isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(D.state_sample, "NAME_BASE_MAX", 0)
    recs = [pilot_rec("k12_37120"), pilot_rec("k12_16472")]
    script: dict[bytes, list[int]] = {}
    for r in recs:
        script.update(script_from_pairs(trace_pairs(r, T)))
    model = ScriptedModel(T, script)
    monkeypatch.setattr(ei, "load_ckpt", lambda path, dev: (model, T, {"think_format": "think"}))
    targets = iso_env / "targets.jsonl"
    targets.write_text(
        "".join(json.dumps({"name": r["id"], "prompt": r["prompt"]}) + "\n" for r in recs)
    )
    a = ei.parser().parse_args(
        [
            *("--init", "x.pt", "--name", "s", "--out", str(iso_env / "ei")),
            *("--targets", str(targets), "--train", "unused", "--rounds", "1", "--k", "2"),
            *("--budgets", "256", "--no_train", "--max_action", "64", "--batch", "2"),
            *("--arm", "logical", "--device", "cpu"),
        ]
    )
    hist = ei.run(a)
    assert hist[0]["targets_round"]["solved"] == 2
    found = [
        json.loads(line) for line in (iso_env / "ei" / "found_1.jsonl").read_text().splitlines()
    ]
    assert found and all(not s.get("rejected") for f in found for s in f["steps"])
    assert json.loads((iso_env / "ei" / "args.json").read_text())["arm"] == "logical"


def test_clip_low_alarms_read_the_accepted_tokens_under_the_penalty() -> None:
    row = {
        "clip_low": 0.05,
        "clip_low_accepted": 0.004,
        "ban_think": 0.0,
        "grad_norm_max": 1.0,
        "overlong_frac": 0.0,
        "nonfinite": 0,
        "alarm_cfg": {},
    }
    assert grpo.clip_low_basis(row) == (0.004, " (accepted tokens)")
    assert grpo.alarms(row, []) == []
    row["clip_low_accepted"] = 0.02
    assert grpo.alarms(row, []) == ["clip-low (accepted tokens) 0.0200 > 0.01"]
    del row["clip_low_accepted"]  # no penalty: today's alarm on every scored token
    assert grpo.clip_low_basis(row) == (0.05, "") and grpo.alarms(row, []) == [
        "clip-low 0.0500 > 0.01"
    ]


def test_policy_loss_splits_clip_low_by_item_kind() -> None:
    model = tiny_model(T.vocab_size, seed=4)
    st = _att(False, [1], end_rejected=True)
    items, _ = grpo.group_items([st, _att(True, [])], T, 0.0, "fail", rej_penalty=P)
    assert items is not None
    old = grpo.old_logps(model, items, 1.0, "cpu")
    # pretend the sampling policy put far more mass on every scored token: every ratio is tiny
    old = [o + 5.0 for o in old]
    rej = [it[3] == -P for it in items]
    s = grpo.policy_loss(model, items, 1.0, "cpu", old=old, rejected=rej)
    n_acc = sum(sum(it[2]) for it, r in zip(items, rej, strict=True) if not r)
    assert s.tokens_acc == n_acc and s.clip_low == 1.0 and s.clip_low_acc == 1.0
    s0 = grpo.policy_loss(model, items, 1.0, "cpu", old=old)
    assert s0.tokens_acc == 0 and s0.clip_low_acc == 0.0  # no flags: not computed


def test_the_penalty_does_not_dilute_the_accepted_tokens() -> None:
    """With `rejected` flags the loss is (sum over accepted + penalised tokens) / accepted tokens:
    the accepted part equals plain GRPO's loss on the accepted items alone."""
    model = tiny_model(T.vocab_size, seed=5)
    att = [_att(True, [0]), _att(False, [1, 1], end_rejected=True)]
    src: list[tuple[int, str]] = []
    items, _ = grpo.group_items(att, T, 0.0, "fail", rej_penalty=P, src=src)
    assert items is not None
    rej = [k == "rej" for _, k in src]
    acc_items = [it for it, r in zip(items, rej, strict=True) if not r]
    rej_items = [it for it, r in zip(items, rej, strict=True) if r]
    n_acc = sum(sum(it[2]) for it in acc_items)
    n_rej = sum(sum(it[2]) for it in rej_items)
    l_all = grpo.policy_loss(model, items, 1.0, "cpu", rejected=rej).loss
    l_acc = grpo.policy_loss(model, acc_items, 1.0, "cpu").loss  # plain GRPO's normaliser
    l_rej = grpo.policy_loss(model, rej_items, 1.0, "cpu").loss
    assert l_all * n_acc == pytest.approx(l_acc * n_acc + l_rej * n_rej, rel=1e-5)
    # without the flags: today's mean over every scored token
    l_mean = grpo.policy_loss(model, items, 1.0, "cpu").loss
    assert l_mean * (n_acc + n_rej) == pytest.approx(l_acc * n_acc + l_rej * n_rej, rel=1e-5)
