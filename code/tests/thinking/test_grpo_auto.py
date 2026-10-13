"""GRPO with auto effort (`--auto_levels`; Leon 2026-10-11): without it the runs are e027354's
(digests of tiny-model runs: rollouts, rows, log lines, weights, args); the effort token is scored
as the sampler drew it (the log-softmax restricted to the allowed effort tokens), gets a gradient
through the clipped ratio and counts in the loss normaliser, never in the think ban mass; the
restriction holds in the scorer, the KL anchor and the rollouts; the level shares and the choice
entropy are logged; the verifier stop, the canary, the KL anchor, the alarms, rollbacks, the
restore check and the stop run with it; the token cost is applied; redraws with auto are
refused."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from code.thinking import grpo
from code.thinking.canary import anchor_kl, choice_seqs, load_canary
from code.thinking.effort import EffortTokenizer, effort_id
from code.thinking.sample import AUTO, choose_levels, phase_bans

from ._grpo_digest import CONFIGS, run_digest, setup_go
from ._pipeline_helpers import tiny_model
from .test_grpo_guard import CHECK, Go, _canary_file

E = EffortTokenizer()
LV = (0, 64, 256)
IDS = [effort_id(E, b) for b in LV]
P = 0.1
THINK_IDS = (E.think, E.end_think)

# `_grpo_digest.run_digest` on a clean export of e027354 (`git archive e027354 code`, its
# docstring): without --auto_levels every part is that commit's
GOLDEN = {
    "guarded": {
        "rollouts": "26bce0a1a4b904364ff844fc851fdee3ec72f1642ea7311ef3b8ced0b019bdb4",
        "rows": "f00e051bf99bc95a56c913cc556957b6a6ee1db069b4da38d54f2f47941245ad",
        "lines": "30738e3973988116d73e13fe5537f9d02f9749e2ba7d1c268ed61e65f7402375",
        "weights": "c5e3c0c060bd913481cb742d90d07ab8a09cc5b00cdfbf181b899ae18a78620e",
        "args": "f1e37f20edb9142f424175253715a729a536b7b32d8d69a4f24b814281f2e4bf",
    },
    "verifier_stop": {
        "rollouts": "a7b347d480893c5c2f3fe646fcc125592f49120262d31115878a206d7589fcd5",
        "rows": "38dc6329f884377893ea3442acd841458cc2c644eca73e392a519fbd89e760ee",
        "lines": "5fd7de2e0afeb36f6927de2bd8625d20dd6bfdfdf53714760654c509d406f982",
        "weights": "f4e31354738a6e0b7e4f990fa7f8d0261bc5af40488fdfeb82a82d15e316f71c",
        "args": "53b771a05a540309fb70769decf19a6b6631b241fd67157f7bffc71770091f78",
    },
    "verifier_redraw": {
        "rollouts": "4a203bd571220088b06d41ee4bebcb47be01fc8121cecdb2243ab3238bb753a1",
        "rows": "8760a1a3a0993dddbfab2f7a195c9064857732ee7cd98e76cd9aa8aaf9b7e71b",
        "lines": "482c3b71607252158aaacc1ddbe3836ab82e29a4e6998f92ed34c9cb5a25aecb",
        "weights": "7c326ff3339f24d81bbe0cb05ab88e2c7692d0bb61774b88ce32c2e7bd3b6220",
        "args": "50651629657b725a0ed35657d2ca732fc9d1ad1413878c2cddca818aea21d166",
    },
}


@pytest.fixture
def go(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Go:
    """test_grpo_guard's `go`: the tiny effort model on an <eos>-biased init, the hash judge."""
    return setup_go(tmp_path, monkeypatch)


@pytest.mark.parametrize("config", sorted(CONFIGS))
def test_default_path_is_e027354(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: str
) -> None:
    assert run_digest(tmp_path, monkeypatch, config) == GOLDEN[config]


# --- scoring the level choice ---------------------------------------------------------------


def _state() -> list[int]:
    return E.encode_toks(["<st>", "h1", ":", "P", "<nl>", "⊢", "(", "P", "→", "Q", ")", "<act>"])


def _bans(levels: tuple[int, ...] = LV) -> grpo.Bans:
    return (*phase_bans(E, E.vocab_size, True), grpo.choice_ban(E, E.vocab_size, levels))


def _choice(level: int = 64, adv: float = 1.0) -> grpo.Item:
    return (_state(), [effort_id(E, level)], [True], adv)


def test_the_effort_token_is_scored_as_the_sampler_draws_it() -> None:
    """The old log-prob of a level choice is log softmax(z / T) over the allowed effort tokens at
    the state's last position: what `choose_levels` samples from (its greedy pick is the argmax).
    The full vocabulary would score it lower; the phase masks alone ban it (-inf)."""
    model = tiny_model(E.vocab_size, seed=4)
    old = grpo.old_logps(model, [_choice()], 0.8, "cpu", think_ids=THINK_IDS, bans=_bans())
    with torch.no_grad():
        z = model(torch.tensor([_state()]))[0, -1].float() / 0.8
    want = torch.log_softmax(z[IDS], -1)
    assert float(old[0][0]) == pytest.approx(float(want[1]), abs=1e-5)
    assert float(want[1]) > float(torch.log_softmax(z, -1)[IDS[1]])
    pick = choose_levels(model, E, [_state()], LV, 0.8, greedy=True)
    assert pick == [LV[int(want.argmax())]]
    bare = grpo.old_logps(model, [_choice()], 0.8, "cpu", bans=phase_bans(E, E.vocab_size, True))
    assert float(bare[0][0]) == -math.inf


def test_the_effort_token_gets_a_gradient_through_the_clipped_ratio() -> None:
    """On-policy (r = 1), A > 0: a gradient on the allowed effort tokens' head rows only (the
    restriction), and a step raises the picked level's restricted log-prob. r = e (above 1 +
    eps_high) with A > 0 is clipped: no gradient, counted in clip-high; r = 1 / e is clip-low."""
    model = tiny_model(E.vocab_size, seed=4)
    item, bans = _choice(64, 1.0), _bans()
    old = grpo.old_logps(model, [item], 1.0, "cpu", bans=bans)
    model.zero_grad(set_to_none=True)
    s = grpo.policy_loss(model, [item], 1.0, "cpu", old=old, bans=bans, think_ids=THINK_IDS)
    assert (s.tokens, s.tokens_choice, s.tokens_think, s.nonfinite) == (1, 1, 0, 0)
    assert s.ratio_max == pytest.approx(1.0) and 0 < s.entropy_choice <= math.log(3) + 1e-6
    assert s.entropy == pytest.approx(s.entropy_choice) and s.entropy_action == 0.0
    g = model.head.weight.grad
    assert g is not None
    others = [i for i in range(E.vocab_size) if i not in IDS]
    assert float(g[IDS].abs().sum()) > 0 and float(g[others].abs().sum()) == 0.0
    with torch.no_grad():
        for q in model.parameters():
            if q.grad is not None:
                q -= 0.5 * q.grad
    after = grpo.old_logps(model, [item], 1.0, "cpu", bans=bans)
    assert float(after[0][0]) > float(old[0][0])
    model.zero_grad(set_to_none=True)
    hi = grpo.policy_loss(model, [item], 1.0, "cpu", old=[after[0] - 1.0], bans=bans)
    assert hi.clip_high == 1.0 and hi.clip_low_choice == 0.0
    assert all(float(q.grad.abs().sum()) == 0.0 for q in model.parameters() if q.grad is not None)
    lo = grpo.policy_loss(model, [item], 1.0, "cpu", old=[after[0] + 1.0], bans=bans)
    assert lo.clip_low == lo.clip_low_choice == 1.0


def _auto_step(level: int, rejected: bool = False) -> dict[str, Any]:
    """A step of an auto attempt as the sampler records it: prompt `state <eB> <think>` (+
    `</think>` at 0), the decoded think + action, the picked `level`."""
    st = _state()
    tail = [E.think] + ([E.end_think] if level == 0 else [])
    think = [] if level == 0 else [E.stoi["P"]] * 3 + [E.end_think]
    d: dict[str, Any] = dict(
        state=bytes(st),
        prompt=bytes([*st, effort_id(E, level), *tail]),
        gen=bytes([*think, E.stoi["exact"], E.eos]),
        forced=False,
        level=level,
    )
    d.update(rejected=True) if rejected else d.update(applied=b"")
    return d


def _auto_att(ok: bool, levels: list[int], end: int | None = None) -> dict[str, Any]:
    trace = [_auto_step(lv) for lv in levels] + ([] if end is None else [_auto_step(end, True)])
    lv = [s["level"] for s in trace]
    return dict(ok=ok, tokens=10 * len(trace), status="done" if ok else "fail:max_rej",
                budget=AUTO, levels=lv, choice=len(lv), trace=trace, redrawn=[])  # fmt: skip


def test_choice_items_carry_the_attempt_advantage() -> None:
    """Every step's effort token (the ending draw's too) is a one-token item with the attempt's
    advantage; the step penalty stays on the ending draw's think + action. Fixed-level attempts
    give none."""
    att = [_auto_att(True, [0, 64]), _auto_att(False, [256], end=64)]
    src: list[tuple[int, str]] = []
    items, _ = grpo.group_items(att, E, 0.0, "fail", rej_penalty=P, src=src)
    assert items is not None
    assert [k for _, k in src] == ["acc", "acc", "choice", "choice"] + ["acc", "acc"] + [
        "choice",
        "choice",
    ]
    ch = [it for it, (_, k) in zip(items, src, strict=True) if k == "choice"]
    assert [it[1] for it in ch] == [[effort_id(E, b)] for b in (0, 64, 256, 64)]
    assert [it[3] for it in ch] == [0.5, 0.5, -0.5, -0.5]
    assert all(it[0] == _state() and it[2] == [True] for it in ch)
    acc = [round(it[3], 6) for it, (_, k) in zip(items, src, strict=True) if k == "acc"]
    assert acc == [0.5, 0.5, -0.5, round(-0.5 - P, 6)]
    fixed = [{k: v for k, v in s.items() if k != "level"} for s in att[0]["trace"]]
    assert grpo.choice_items({"trace": fixed}, 0.5, E) == []


def test_choice_tokens_are_in_the_normaliser_and_out_of_the_think_ban() -> None:
    """With the step penalty's flags the loss is the sum over every scored token / (accepted +
    level-choice tokens); the think ban mass and the think token count do not change when level
    choices are added, and `entropy_action` leaves them out."""
    model = tiny_model(E.vocab_size, seed=5)
    att = [_auto_att(True, [64, 256]), _auto_att(False, [64], end=0)]
    src: list[tuple[int, str]] = []
    items, _ = grpo.group_items(att, E, 0.0, "fail", rej_penalty=P, src=src)
    assert items is not None
    rej_draw = (_state() + [effort_id(E, 64), E.think], [E.stoi["P"], E.end_think], [True] * 2, -P)
    items, kinds = [*items, rej_draw], [k for _, k in src] + ["rej"]
    bans = _bans()

    def loss(sel: list[str], flags: bool) -> tuple[float, int, Any]:
        its = [it for it, k in zip(items, kinds, strict=True) if k in sel]
        rej = [k == "rej" for k in kinds if k in sel] if flags else None
        old = grpo.old_logps(model, its, 1.0, "cpu", think_ids=THINK_IDS, bans=bans)
        s = grpo.policy_loss(
            model, its, 1.0, "cpu", old=old, think_ids=THINK_IDS, bans=bans, rejected=rej
        )
        return s.loss, sum(sum(it[2]) for it in its), s

    l_all, _, s_all = loss(["acc", "choice", "rej"], True)
    parts = [loss([k], False)[:2] for k in ("acc", "choice", "rej")]
    n_norm = parts[0][1] + parts[1][1]
    assert l_all * n_norm == pytest.approx(sum(lk * nk for lk, nk in parts), rel=1e-5)
    assert s_all.tokens_choice == parts[1][1] == 4 and s_all.tokens_acc == n_norm
    _, _, s_no = loss(["acc", "rej"], True)
    assert s_no.tokens_think == s_all.tokens_think > 0
    assert s_no.ban_think == pytest.approx(s_all.ban_think, rel=1e-6) and s_all.ban_think > 0
    assert s_no.entropy_action == pytest.approx(s_all.entropy_action, rel=1e-5)
    assert grpo.ban_mass(model, items, 1.0, "cpu", 50_000, THINK_IDS, bans) == pytest.approx(
        s_all.ban_think, rel=1e-6
    )


def test_the_kl_anchor_reads_the_level_choice(tmp_path: Path) -> None:
    """`choice_seqs` cuts each B0 anchor step at its state; `anchor_kl` there with the choice
    mask is KL(pi_ref || pi) over the allowed levels: 0 at the reference, positive elsewhere, and
    its gradient reaches the allowed effort tokens' head rows only."""
    seqs = load_canary(_canary_file(tmp_path / "c.jsonl"), E, True, 0).anchor
    cs = choice_seqs(seqs, E)
    assert len(cs) == len(seqs)
    for (p, t), (q, _) in zip(cs, seqs, strict=True):
        assert q == [*p, effort_id(E, 0), E.think, E.end_think] and t == [effort_id(E, 0)]
    ban = grpo.choice_ban(E, E.vocab_size, LV)
    model, ref = tiny_model(E.vocab_size, seed=7), tiny_model(E.vocab_size, seed=8)
    assert abs(anchor_kl(model, copy.deepcopy(model), cs, 1.0, ban, "cpu", backward=False)) < 1e-6
    want = 0.0
    with torch.no_grad():
        for p, _ in cs:
            lr_, lp = (
                torch.log_softmax(m(torch.tensor([p]))[0, -1, IDS].float(), -1)
                for m in (ref, model)
            )
            want += float((lr_.exp() * (lr_ - lp)).sum())
    model.zero_grad(set_to_none=True)
    kl = anchor_kl(model, ref, cs, 1.0, ban, "cpu", coef=1.0)
    assert kl == pytest.approx(want / len(cs), rel=1e-5) and kl > 0
    g = model.head.weight.grad
    assert g is not None
    others = [i for i in range(E.vocab_size) if i not in IDS]
    assert float(g[IDS].abs().sum()) > 0 and float(g[others].abs().sum()) == 0.0


# --- the token cost -------------------------------------------------------------------------


def test_choice_lambda_is_ei_auto_token_cost() -> None:
    """Every attempt pays lambda x tokens / 1,000, so a group's mean reward is ei_auto's label
    score (pass rate - lambda x mean tokens / 1,000), and an all-solved group whose lengths differ
    now gives a gradient that favours the shorter proof."""
    att = [dict(ok=True, tokens=300), dict(ok=False, tokens=100), dict(ok=True, tokens=900)]
    r = grpo.rewards(att, 0.0, choice_lambda=0.5)
    assert r == pytest.approx([1 - 0.15, -0.05, 1 - 0.45]) and grpo.rewards(att, 0.0) == [1, 0, 1]
    assert sum(r) / 3 == pytest.approx(2 / 3 - 0.5 * (1300 / 3) / 1000)
    solved = [_auto_att(True, [0]), _auto_att(True, [64, 256])]  # 10 and 20 tokens
    assert grpo.group_items(solved, E, 0.0, "fail")[0] is None
    items, _ = grpo.group_items(solved, E, 0.0, "fail", choice_lambda=0.5)
    assert items is not None
    advs = sorted({round(it[3], 9) for it in items})
    assert advs == pytest.approx([-0.0025, 0.0025])  # (20 - 10) x 0.5 / 1,000 / 2


# --- end to end on the tiny model -----------------------------------------------------------


def _spy(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Collect every attempt the GRPO sampler returns."""
    got: list[dict[str, Any]] = []
    real = grpo.think_generate

    def spy(*args: Any, **kw: Any) -> Any:
        res = real(*args, **kw)
        got.extend(res)
        return res

    monkeypatch.setattr(grpo, "think_generate", spy)
    return got


def test_auto_iterations_pick_allowed_levels_and_log_them(
    go: Go, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--auto_levels 0,64 on the tiny model: every draw is at 0 or 64 (16 is a level, but not
    allowed) with `<eB>` at the state's length in its prompt; groups are (target, auto); the rows
    log the level shares, the choice entropy and the scored level choices; the it line shows them;
    the monitor splits the choice mass off the action's."""
    got = _spy(monkeypatch)
    hist, out = go(["--iters", "2", "--auto_levels", "0,64", "--monitor", "--overlong", "fail"])
    assert any(h["steps"] for h in hist)
    assert {b for x in got for b in x["levels"]} == {0, 64}
    assert all(x["budget"] == AUTO and len(x["levels"]) == x["draws"] for x in got)
    for x in got:
        for s in x["trace"]:
            assert s["prompt"][len(s["state"])] == effort_id(E, s["level"])
    for i, h in enumerate(hist):  # one sampler call per iteration: 2 groups x 4 attempts
        lv = [b for x in got[8 * i : 8 * i + 8] for b in x["levels"]]
        assert h["level_share"]["0"] == pytest.approx(lv.count(0) / len(lv))
        assert set(h["level_share"]) == {"0", "64"} and h["auto_levels"] == [0, 64]
        assert sum(h["level_share"].values()) == pytest.approx(1.0)
        assert 0 < h["choice_entropy"] <= math.log(2) + 1e-6
        assert 0 <= h["clip_low_choice"] <= 1 and h["choice_cost"] == 0.0 and h["nonfinite"] == 0
        if h["steps"]:  # a group with a gradient: its effort tokens were scored
            assert h["tokens_choice"] > 0 and 0 < h["entropy_choice"] <= math.log(2) + 1e-6
            assert h["trained_tokens"] > h["tokens_choice"] and h["monitor"]["mass_choice"] > 0
        assert set(h["pass1_by_budget"]) == {"auto"} == set(h["capped_by_budget"])
        assert h["curriculum"]["pairs"] == 2 and set(h["think_block_by_budget"]) <= {"64"}
        assert set(h["len_dev_by_budget"]) <= {"auto"}
        mo = h["monitor"]
        assert mo["mass_think"] + mo["mass_action"] + mo["mass_choice"] == pytest.approx(
            1.0 if h["steps"] else 0.0
        )
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("it ")]
    assert len(lines) == 2
    assert all(" levels e0 " in x and " e64 " in x and " ch-ent " in x for x in lines)


def test_verifier_stop_with_auto(go: Go, monkeypatch: pytest.MonkeyPatch) -> None:
    """The logical arm without redraws: a rejected step ends the attempt and gets -P; the effort
    tokens are scored; no redrawn draws."""
    got = _spy(monkeypatch)
    argv = ["--iters", "2", "--auto_levels", "0,16", "--arm", "logical", "--max_rej", "0"]
    hist, _ = go([*argv, "--rej_penalty", "0.1", "--monitor"])
    assert any(x["status"] == "fail:max_rej" for x in got) and all(x["rej"] == 0 for x in got)
    for h in hist:
        assert h["tokens_choice"] > 0 and h["nonfinite"] == 0
        assert 0 < h["tokens_accepted_share"] <= 1 and 0 <= h["clip_low_accepted"] <= 1
        mo = h["monitor"]
        assert mo["arm"] == "logical" and mo["redrawn_drafts"] == 0 and mo["rej_per_draw"] > 0


def test_auto_with_the_canary_and_the_kl_anchor(go: Go, capsys: pytest.CaptureFixture[str]) -> None:
    ref = go.iso / "g" / "ckpts" / "g_it0.pt"
    argv = ["--iters", "2", "--auto_levels", "0,16", "--canary", str(go.iso / "canary.jsonl")]
    hist, _ = go([*argv, "--canary_n", "4", "--kl_anchor_coef", "0.5", "--kl_anchor_ref", str(ref)])
    for h in hist:
        assert h["canary"]["steps"] == 4 and h["canary"]["nll"] > 0
        assert math.isfinite(h["kl_anchor_choice"]) and h["kl_anchor_choice"] >= 0
    assert hist[1]["kl_anchor_choice"] > 0 and hist[1]["kl_anchor"] > 0  # off pi_ref
    printed = capsys.readouterr().out
    assert "canary it 2:" in printed and " choice " in printed.split("guard it 2:")[1]


def test_auto_alarms_rollback_restore_check_and_stop(
    go: Go, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The <eos>-biased tiny model raises the think ban alarm (the effort tokens are not think
    tokens). With --rollback_on_alarm it rolls back; with the restore check it is let through
    (its restore point is over the limit too); over --ban_stop the run ends with exit code 3."""
    hist, out = go(["--iters", "2", "--auto_levels", "0,16", "--rollback_on_alarm"], out="rb")
    assert hist[0]["steps"] and hist[0]["alarms"][0].startswith("think banned mass")
    assert [h["rollback"]["n"] for h in hist] == [1, 2] and hist[1]["rollback"]["lr_scale"] == 0.25
    monkeypatch.setattr(grpo, "alarms", lambda row, prev: [m for m in [grpo.ban_alarm(row)] if m])
    hist, out = go(["--iters", "2", "--auto_levels", "0,16", *CHECK], out="check")
    bc = hist[0]["ban_check"]
    assert bc["rollback"] is False and bc["restore_mass"] > grpo.ALARM_BAN
    assert bc["mass"] == hist[0]["ban_think"] and hist[0]["rollback"] is None
    assert "ALARM it 1: no rollback: the restore point (it 0)" in capsys.readouterr().out
    argv = ["--iters", "3", "--auto_levels", "0,16", *CHECK, "--ban_stop", "0.005"]
    assert grpo.main(go.argv(argv, out="stop")) == 3


def test_auto_token_cost_end_to_end(go: Go) -> None:
    hist, _ = go(["--iters", "1", "--auto_levels", "0,16", "--choice_lambda", "0.5"])
    h = hist[0]
    assert h["choice_cost"] == pytest.approx(0.5 * h["tokens_per_attempt"] / 1000)


@pytest.mark.parametrize(
    ("argv", "msg"),
    [
        (["--arm", "logical", "--max_rej", "3"], "does not support redraws"),
        (["--arm", "structural"], "does not support redraws"),  # the default --max_rej 10
        (["--len_target_coef", "0.2"], "len_target_coef"),
        (["--dump_batches", "dumps"], "dump_batches"),
    ],
)
def test_auto_refusals(go: Go, argv: list[str], msg: str) -> None:
    """Refused before anything is written."""
    with pytest.raises(SystemExit, match=msg):
        go(["--iters", "1", "--auto_levels", "0,16", *argv])
    assert not (go.iso / "g").exists()


@pytest.mark.parametrize(
    ("argv", "msg"),
    [
        (["--auto_levels", "0,32"], "distinct effort levels"),
        (["--auto_levels", "0,0"], "distinct effort levels"),
        (["--auto_levels", "x"], "distinct effort levels"),
        (["--choice_lambda", "0.5"], "needs --auto_levels"),
        (["--auto_levels", "0,16", "--choice_lambda", "-1"], "must be >= 0"),
    ],
)
def test_bad_auto_options(go: Go, argv: list[str], msg: str) -> None:
    with pytest.raises(SystemExit, match=msg):
        go(["--iters", "1", *argv])
    assert not (go.iso / "g").exists()


def test_auto_needs_effort(go: Go) -> None:
    argv = [x for x in go.argv(["--iters", "1", "--auto_levels", "0,16"]) if x != "--effort"]
    with pytest.raises(SystemExit, match="needs --effort"):
        grpo.run(grpo.parser().parse_args(argv))
