"""--len_penalty's length-only groups (Leon 2026-10-16, for the mini-run knobs lpm05 / lp05 / lp15).
An all-solved group's rewards 1 - lp x tokens / 1000 differ by length alone. Such a group is still
trained (the length term is the penalty's point), but it has no outcome signal: the unlikeliness
reward (and --refill) read the outcomes without the penalty, so its ranks never apply there, and
the loss divides by the other items' tokens, so its gradient comes on top of the outcome signal
(as the step penalty's does) instead of diluting it, an lr cut in disguise. With --len_penalty 0
nothing changes (test_grpo_pilots.py: the c4b8548 digests, "mixed0" included)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from code.thinking import grpo

from ._grpo_digest import setup_go
from ._pipeline_helpers import tiny_model
from .test_grpo_pilots import Pilot
from .test_grpo_verifier import T


def _att(i: int, ok: bool, tokens: int) -> dict[str, Any]:
    step = {"prompt": bytes([5, 6]), "gen": bytes([10 + i, 11 + i]), "forced": False}
    return {"status": "done" if ok else "fail:S:bad", "ok": ok, "tokens": tokens, "trace": [step]}


def _adv(items: list[grpo.Item] | None) -> list[float]:
    assert items is not None
    return [a for *_, a in items]


@pytest.mark.parametrize("lp", [0.05, -0.05, 0.15])
def test_an_all_solved_group_is_length_only(lp: float) -> None:
    att = [_att(i, True, n) for i, n in enumerate((100, 200, 300, 400))]
    sig: list[bool] = []
    src: list[tuple[int, str]] = []
    items, _ = grpo.group_items(att, T, lp, "fail", signal=sig, src=src)
    # Dr. GRPO: r - the mean, no std; the mean is 250 tokens
    assert sig == [False] and src == [(j, "len") for j in range(4)]
    assert _adv(items) == pytest.approx([-lp * (n - 250) / 1000 for n in (100, 200, 300, 400)])
    # the unlikeliness deductions never apply to it
    got, _ = grpo.group_items(att, T, lp, "fail", unlikely=[0.25, 0.1, 0.0, 0.0])
    assert _adv(got) == pytest.approx(_adv(items))
    # without the penalty, or with equal lengths, or all failed: skipped, as before
    assert grpo.group_items(att, T, 0.0, "fail", signal=sig)[0] is None
    same = [_att(i, True, 300) for i in range(4)]
    assert grpo.group_items(same, T, lp, "fail", signal=sig)[0] is None
    failed = [_att(i, False, n) for i, n in enumerate((100, 200, 300, 400))]
    assert grpo.group_items(failed, T, lp, "fail", signal=sig)[0] is None
    assert sig == [False, False, False, False]


def test_a_mixed_group_keeps_its_signal_and_its_deductions() -> None:
    att = [_att(0, True, 100), _att(1, True, 300), _att(2, False, 200), _att(3, False, 50)]
    sig: list[bool] = []
    src: list[tuple[int, str]] = []
    items, _ = grpo.group_items(att, T, 0.1, "fail", signal=sig, src=src, unlikely=[0.2, 0, 0, 0])
    assert sig == [True] and {k for _, k in src} == {"acc"}
    r = [1 - 0.01 - 0.2, 1 - 0.03, 0.0, 0.0]  # the length term stays additive in its rewards
    m = sum(r) / 4
    assert _adv(items) == pytest.approx([x - m for x in r])


def test_length_only_items_are_on_top_of_the_divisor() -> None:
    """With `on_top` the loss is (sum over every scored token) / the other items' tokens: the
    other items' part is their own plain loss, the length-only part comes on top."""
    model = tiny_model(T.vocab_size, seed=6)
    mixed = [_att(0, True, 100), _att(1, False, 100)]
    solved = [_att(2, True, 100), _att(3, True, 300)]
    a, _ = grpo.group_items(mixed, T, 0.05, "fail")
    b, _ = grpo.group_items(solved, T, 0.05, "fail")
    assert a is not None and b is not None
    items = a + b
    top = [False] * len(a) + [True] * len(b)
    n_a, n_b = (sum(sum(it[2]) for it in x) for x in (a, b))
    l_all = grpo.policy_loss(model, items, 1.0, "cpu", on_top=top).loss
    l_a = grpo.policy_loss(model, a, 1.0, "cpu").loss
    l_b = grpo.policy_loss(model, b, 1.0, "cpu").loss
    assert l_all * n_a == pytest.approx(l_a * n_a + l_b * n_b, rel=1e-5)
    # flags all off: the plain mean over every scored token, as without them
    l_off = grpo.policy_loss(model, items, 1.0, "cpu", on_top=[False] * len(items)).loss
    assert l_off == pytest.approx(grpo.policy_loss(model, items, 1.0, "cpu").loss, rel=1e-6)
    # with the step penalty's flags too: both kinds out of the divisor
    l_both = grpo.policy_loss(
        model, items, 1.0, "cpu", rejected=[True] + [False] * (len(items) - 1), on_top=top
    ).loss
    n_a1 = n_a - sum(items[0][2])
    l_1 = grpo.policy_loss(model, items[:1], 1.0, "cpu").loss
    l_a2 = grpo.policy_loss(model, a[1:], 1.0, "cpu").loss
    want = l_1 * sum(items[0][2]) + l_a2 * n_a1 + l_b * n_b
    assert l_both * n_a1 == pytest.approx(want, rel=1e-5)


@pytest.fixture
def lpilot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Pilot:
    return Pilot(setup_go(tmp_path, monkeypatch), monkeypatch)


def test_a_run_logs_its_length_only_groups(
    lpilot: Pilot, capsys: pytest.CaptureFixture[str]
) -> None:
    """Three always-solved targets: their groups are length-only (when their lengths differ), out
    of the unlikeliness reward's groups, and counted in row['len_only'], the monitor and a line."""
    argv = ["--iters", "3", "--prompts", "4", "--seed", "1", "--overlong", "fail", "--monitor"]
    plain, _ = lpilot([*argv, "--unlikely_beta", "0.25"], out="plain")
    assert not any("len_only" in h or "len_only_groups" in h["monitor"] for h in plain)
    hist, _ = lpilot([*argv, "--unlikely_beta", "0.25", "--len_penalty", "0.05"], out="lp")
    printed = capsys.readouterr().out
    n_len = 0
    for h, c in zip(hist, lpilot.calls, strict=True):
        per = c["per"]
        solved = [all(x["ok"] for x in g) and len({x["tokens"] for x in g}) > 1 for g in per]
        signal = [len({bool(x["ok"]) for x in g}) > 1 for g in per]
        assert h["len_only"]["groups"] == sum(solved) == h["monitor"]["len_only_groups"]
        assert h["unlikely"]["groups"] == sum(signal)
        assert h["groups_skipped"] == pytest.approx(1 - (sum(solved) + sum(signal)) / len(per))
        assert f"len it {h['iter']}: {sum(solved)} length-only groups of {len(per)}" in printed
        n_len += sum(solved)
    assert n_len  # the run had some
