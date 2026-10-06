"""GRPO: rewards and advantages, per-step items (forced close masked, the failing draw kept), the
alignment of the old log-probs with the scored tokens, the PPO surrogate (REINFORCE at ratio 1, the
clip-higher clip), the logged entropy, the direction of an update, the learnability curriculum, and
two end-to-end iterations on a tiny model."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import pytest
import torch

from code.thinking import grpo
from code.thinking.ckpt import save_ckpt
from code.thinking.data import trace_pairs
from code.thinking.sample import LINE_CAP, think_generate
from code.thinking.tok import D, ThinkTokenizer

from ._pipeline_helpers import (
    ScriptedModel,
    bias_logits,
    isolate,
    pilot_rec,
    script_from_pairs,
    tiny_model,
)

T = ThinkTokenizer()
TEMP = 0.8


def test_rewards_and_advantages() -> None:
    att = [{"ok": True, "tokens": 500}, {"ok": False, "tokens": 50}, {"ok": True, "tokens": 0}]
    assert grpo.rewards(att, 0.0) == [1.0, 0.0, 1.0]
    assert grpo.rewards(att, 0.2) == pytest.approx([0.9, 0.0, 1.0])
    adv = grpo.advantages([1.0, 0.0, 1.0, 0.0])
    assert adv == [0.5, -0.5, 0.5, -0.5]
    assert grpo.advantages([1.0, 1.0]) is None and grpo.advantages([0.0, 0.0, 0.0]) is None


def test_accepted_attempt_items_are_the_sampled_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(D.state_sample, "NAME_BASE_MAX", 0)
    rec = pilot_rec("k12_22530")
    pairs = trace_pairs(rec, T)
    model = ScriptedModel(T, script_from_pairs(pairs))
    (x,) = think_generate(model, T, [rec["prompt"]], "plain", budgets=256, max_action=64, batch=1)
    assert x["status"] == "done"
    items = grpo.attempt_items(x, 0.5, T)
    assert len(items) == len(pairs)
    for (p, g, m, a), (sp, sq, _) in zip(items, pairs, strict=True):
        assert bytes(p) == sp + bytes([T.think])  # state + the prefilled <think>
        assert [T.think, *g] == list(sq) and all(m) and a == 0.5


@pytest.mark.parametrize(("force_mode", "close_at"), [("exact", 5), ("line", 5 + LINE_CAP)])
def test_failing_draw_is_kept_and_a_forced_close_is_masked(force_mode: str, close_at: int) -> None:
    """A think block with no `<nl>`: 'exact' forces the close at B = 5, 'line' at the cap B + 64."""
    b = torch.zeros(T.vocab_size)
    b[T.stoi["P"]] = 60.0  # thinks "P P P ..." until the budget forces </think>, then a bad action
    model = bias_logits(tiny_model(T.vocab_size), b)
    (x,) = think_generate(
        model,
        T,
        [pilot_rec("k12_37120")["prompt"]],
        "plain",
        budgets=5,
        max_action=8,
        batch=1,
        force_mode=force_mode,
    )
    assert x["status"].startswith("fail")
    (step,) = x["trace"]
    assert step["rejected"] and step["forced"]
    ((p, g, m, a),) = grpo.attempt_items(x, -1.0, T)
    j = g.index(T.end_think)
    assert j == close_at and m[j] is False and sum(m) == len(g) - 1 and a == -1.0


# --- overlong masking -------------------------------------------------------------------------


def _att(i: int, status: str, ok: bool, forced: bool = False) -> dict[str, Any]:
    """A one-step attempt whose generated ids identify it: [10 + i, 11 + i]."""
    step = {"prompt": bytes([5, 6]), "gen": bytes([10 + i, 11 + i]), "forced": forced}
    return {"status": status, "ok": ok, "tokens": 2, "think": 0, "trace": [step]}


def _who(items: list[grpo.Item] | None) -> dict[int, float]:
    """attempt -> its advantage, from the items."""
    return {} if items is None else {g[0] - 10: a for _, g, _, a in items}


def test_overlong_attempts_get_no_gradient_and_leave_the_baseline() -> None:
    att = [
        _att(0, "done", True),
        _att(1, "fail:S:bad", False),  # a verdict
        _att(2, "fail:step_cap", False),  # caps, no verdict
        _att(3, "fail:truncated", False),
        _att(4, "fail:context", False),
        _att(5, "done", False, forced=True),  # Lean rejected; a capped think block is no cap
    ]
    items, masked = grpo.group_items(att, T, 0.0, "mask")
    assert masked == [False, False, True, True, True, False]
    # baseline over the 3 unmasked rewards (1, 0, 0); the masked attempts have no items at all
    assert _who(items) == pytest.approx({0: 2 / 3, 1: -1 / 3, 5: -1 / 3})
    items, masked = grpo.group_items(att, T, 0.0, "fail")  # the old behaviour: capped = failed
    assert not any(masked)
    assert _who(items) == pytest.approx({0: 5 / 6, **{k: -1 / 6 for k in range(1, 6)}})


@pytest.mark.parametrize(
    "statuses",
    [
        [("done", True), ("fail:step_cap", False), ("fail:truncated", False)],  # 1 unmasked
        [("done", True), ("done", True), ("fail:truncated", False)],  # unmasked all equal
    ],
)
def test_groups_with_too_few_unmasked_attempts_are_skipped(statuses: list[Any]) -> None:
    att = [_att(i, st, ok) for i, (st, ok) in enumerate(statuses)]
    assert grpo.group_items(att, T, 0.0, "mask")[0] is None
    assert len(_who(grpo.group_items(att, T, 0.0, "fail")[0])) == 3  # unmasked, all scored
    assert grpo.advantages([1.0, 0.0], [True, False]) is None


def test_the_sampler_cap_statuses_are_overlong(monkeypatch: pytest.MonkeyPatch) -> None:
    """The statuses think_generate gives an attempt a cap ends match OVERLONG."""
    b = torch.zeros(T.vocab_size)
    b[T.stoi["P"]] = 60.0  # never writes <eos>: the action hits max_action
    model = bias_logits(tiny_model(T.vocab_size), b)
    prompt = pilot_rec("k12_37120")["prompt"]
    (x,) = think_generate(model, T, [prompt], "plain", budgets=5, max_action=8, batch=1)
    assert x["status"] == "fail:truncated" and grpo.is_overlong(x)
    monkeypatch.setattr(D.state_sample, "NAME_BASE_MAX", 0)
    rec = pilot_rec("k12_22530")
    scripted = ScriptedModel(T, script_from_pairs(trace_pairs(rec, T)))
    (y,) = think_generate(scripted, T, [rec["prompt"]], "plain", budgets=256, max_steps=1, batch=1)
    assert y["status"] == "fail:step_cap" and grpo.is_overlong(y)
    (z,) = think_generate(scripted, T, [rec["prompt"]], "plain", budgets=256, batch=1)
    assert z["status"] == "done" and not grpo.is_overlong(z)


# --- log-probs, alignment and the PPO surrogate ------------------------------------------------


def _direct_logp(model: Any, p: list[int], g: list[int], temp: float = 1.0) -> torch.Tensor:
    """log pi_T of every token of g after p, one sequence at a time (no padding, no chunks)."""
    with torch.no_grad():
        x = torch.tensor([p + g])
        lp = torch.log_softmax(model(x[:, :-1]).float() / temp, -1)[0]
        return torch.stack([lp[len(p) - 1 + t, tok] for t, tok in enumerate(g)])


def _items(seed: int = 0, n: int = 9) -> list[grpo.Item]:
    """Items of mixed prompt / generation lengths, some tokens masked, mixed advantages."""
    rng = random.Random(seed)
    out: list[grpo.Item] = []
    for _ in range(n):
        p = [rng.randrange(2, 40) for _ in range(rng.randint(1, 12))]
        g = [rng.randrange(2, 40) for _ in range(rng.randint(1, 9))]
        mk = [True] + [rng.random() < 0.75 for _ in g[1:]]
        out.append((p, g, mk, rng.choice([-0.75, -0.25, 0.25, 0.75])))
    return out


def _grads(model: Any, items: list[grpo.Item], **kw: Any) -> tuple[list[torch.Tensor], Any]:
    model.zero_grad(set_to_none=True)
    st = grpo.policy_loss(model, items, TEMP, "cpu", 30, **kw)
    g = [
        torch.zeros_like(q) if q.grad is None else q.grad.detach().clone()
        for q in model.parameters()
    ]
    return g, st


def _old_at_ratio(model: Any, items: list[grpo.Item], r: float) -> list[torch.Tensor]:
    """Old log-probs that put the ratio pi / pi_old at r on every scored token."""
    return [o - math.log(r) for o in grpo.old_logps(model, items, TEMP, "cpu")]


def test_old_logps_align_with_the_scored_tokens_across_chunks() -> None:
    model = tiny_model(T.vocab_size)
    items = _items()
    assert len(grpo._chunks(items, 30)) > 1  # length-sorted chunks reorder the items
    want = [_direct_logp(model, p, g, TEMP)[torch.tensor(mk)] for p, g, mk, _ in items]
    for max_tokens in (100_000, 30):
        old = grpo.old_logps(model, items, TEMP, "cpu", max_tokens)
        assert len(old) == len(items)
        for o, w in zip(old, want, strict=True):
            assert o.shape == w.shape and torch.allclose(o, w, atol=1e-5)


def test_ppo_at_ratio_one_is_reinforce() -> None:
    """old = the current policy: r = 1, so the clipped surrogate's gradient is REINFORCE's and
    nothing is clipped."""
    model = tiny_model(T.vocab_size)
    items = _items()
    old = grpo.old_logps(model, items, TEMP, "cpu")
    g_pg, s_pg = _grads(model, items)
    g_ppo, s_ppo = _grads(model, items, old=old)
    assert s_pg.tokens == s_ppo.tokens == sum(sum(mk) for _, _, mk, _ in items)
    assert s_pg.clip_frac == s_ppo.clip_frac == 0.0
    assert any(x.abs().sum() > 0 for x in g_pg)
    for a, b in zip(g_pg, g_ppo, strict=True):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize(
    ("r", "adv", "eps_high", "factor"),
    [
        (math.e, 0.5, 0.28, 0.0),  # far above 1 + eps, A > 0: the clipped branch binds
        (math.e, -0.5, 0.28, math.e),  # A < 0: the unclipped r A is the minimum
        (1.25, 0.5, 0.28, 1.25),  # clip-higher: inside [0.8, 1.28], still learning
        (1.25, 0.5, 0.2, 0.0),  # the symmetric clip would have stopped it
        (1.25, -0.5, 0.2, 1.25),
        (0.75, -0.5, 0.28, 0.0),  # below 1 - eps_low, A < 0: clipped
        (0.75, 0.5, 0.28, 0.75),  # A > 0: unclipped
    ],
)
def test_ppo_clip_higher(r: float, adv: float, eps_high: float, factor: float) -> None:
    """The gradient is factor x REINFORCE's (0: clipped, no gradient). The clip fraction counts
    the tokens outside [1 - eps_low, 1 + eps_high], whichever branch binds."""
    model = tiny_model(T.vocab_size)
    items = [(p, g, mk, adv) for p, g, mk, _ in _items()]
    g_ppo, st = _grads(model, items, old=_old_at_ratio(model, items, r), eps_high=eps_high)
    g_pg, _ = _grads(model, items)
    high, low = float(r > 1 + eps_high), float(r < 1 - 0.2)
    assert (st.clip_high, st.clip_low, st.clip_frac) == (high, low, high + low)
    for a, b in zip(g_pg, g_ppo, strict=True):
        if factor == 0.0:
            assert int(torch.count_nonzero(b)) == 0
        else:
            assert torch.allclose(b, factor * a, rtol=1e-4, atol=1e-7)


def test_entropy_is_the_mean_over_the_scored_tokens() -> None:
    model = tiny_model(T.vocab_size)
    items = _items()
    tot, n = 0.0, 0
    for p, g, mk, _ in items:
        with torch.no_grad():
            x = torch.tensor([p + g])
            lp = torch.log_softmax(model(x[:, :-1]).float() / TEMP, -1)[0, len(p) - 1 :]
        h = -(lp.exp() * lp).sum(-1)[torch.tensor(mk)]
        tot, n = tot + float(h.sum()), n + len(h)
    _, st = _grads(model, items)
    assert st.entropy == pytest.approx(tot / n, rel=1e-5)
    assert 0 < st.entropy < math.log(T.vocab_size)
    b = torch.zeros(T.vocab_size)
    b[T.stoi["P"]] = 60.0
    _, peaked = _grads(bias_logits(tiny_model(T.vocab_size), b), items)
    assert peaked.entropy < 1e-6  # a collapsed policy


def test_think_len() -> None:
    ids = (T.think, T.end_think)
    st, q, a_, e = T.stoi["⊢"], T.stoi["Q"], T.stoi["exact"], T.eos
    assert grpo.think_len([st, T.think], [q, q, T.end_think, a_, e], ids) == 3  # `</think>` in
    assert grpo.think_len([st, T.think, T.end_think], [a_, e], ids) == 0  # budget 0
    assert grpo.think_len([st, T.think], [q, q], ids) == 2  # no close: all think
    assert grpo.think_len([st, T.think], [q, T.end_think, a_], None) == 0  # no think vocabulary


def test_entropy_splits_into_think_and_action_tokens() -> None:
    model = tiny_model(T.vocab_size)
    ids = (T.think, T.end_think)
    st, q, a_, e = T.stoi["⊢"], T.stoi["Q"], T.stoi["exact"], T.eos
    items: list[grpo.Item] = [
        ([st, T.think], [q, q, T.end_think, a_, e], [True, True, False, True, True], 0.5),
        ([st, q, T.think, T.end_think], [a_, a_, e], [True] * 3, -0.5),  # budget 0: action only
    ]
    sums = {True: [0.0, 0], False: [0.0, 0]}
    for p, g, mk, _ in items:
        with torch.no_grad():
            x = torch.tensor([p + g])
            lp = torch.log_softmax(model(x[:, :-1]).float() / TEMP, -1)[0, len(p) - 1 :]
        h = -(lp.exp() * lp).sum(-1)
        n_think = grpo.think_len(p, g, ids)
        for t, keep in enumerate(mk):
            if keep:
                sums[t < n_think][0] += float(h[t])
                sums[t < n_think][1] += 1
    _, stats = _grads(model, items, think_ids=ids)
    assert stats.tokens_think == sums[True][1] == 2  # the masked (forced) `</think>` is out
    assert stats.tokens == 7
    assert stats.entropy_think == pytest.approx(sums[True][0] / 2, rel=1e-5)
    assert stats.entropy_action == pytest.approx(sums[False][0] / 5, rel=1e-5)
    both = (2 * stats.entropy_think + 5 * stats.entropy_action) / 7
    assert stats.entropy == pytest.approx(both, rel=1e-5)
    _, plain = _grads(model, items)  # without think_ids every token is an action token
    assert plain.tokens_think == 0 and plain.entropy_action == pytest.approx(plain.entropy)


@pytest.mark.parametrize(
    ("argv", "want"),
    [
        ([], (0.2, 0.28)),
        (["--clip_eps", "0.1"], (0.1, 0.1)),
        (["--clip_eps", "0.1", "--clip_eps_high", "0.3"], (0.1, 0.3)),
        (["--clip_eps_low", "0.15"], (0.15, 0.28)),
    ],
)
def test_clip_range(argv: list[str], want: tuple[float, float]) -> None:
    a = grpo.parser().parse_args(
        ["--init", "i", "--name", "n", "--out", "o", "--targets", "t"] + argv
    )
    assert grpo.clip_range(a) == pytest.approx(want)


@pytest.mark.parametrize("ppo", [False, True])
@pytest.mark.parametrize("adv", [1.0, -1.0])
def test_update_direction(adv: float, ppo: bool) -> None:
    model = tiny_model(T.vocab_size)
    p, g = [T.stoi["⊢"], T.stoi["P"], T.think], [T.stoi["Q"], T.end_think, T.stoi["exact"]]
    items: list[grpo.Item] = [(p, g, [True] * 3, adv)]
    before = float(_direct_logp(model, p, g).sum())
    old = grpo.old_logps(model, items, 1.0, "cpu") if ppo else None
    opt = torch.optim.SGD(model.parameters(), lr=0.01)  # 0.5 overshoots on the tiny model
    opt.zero_grad()
    st = grpo.policy_loss(model, items, 1.0, "cpu", old=old)
    opt.step()
    after = float(_direct_logp(model, p, g).sum())
    assert st.tokens == 3 and st.clip_frac == 0.0 and (after - before) * adv > 0


# --- curriculum -------------------------------------------------------------------------------


def test_curriculum_unseen_pairs_are_at_one_half() -> None:
    c = grpo.Curriculum(3, [0, 16], floor=0.03, decay=0.7, seed=0)
    assert c.keys == [(0, 0), (0, 16), (1, 0), (1, 16), (2, 0), (2, 16)]
    assert all(c.p(i) == 0.5 for i in range(len(c.keys)))
    assert all(c.weight(i) == pytest.approx(0.28) for i in range(len(c.keys)))
    assert c.summary() == {"seen": 0, "pairs": 6, "frontier": 0.0, "mean_p_by_budget": {}}


def test_curriculum_frontier_pairs_outweigh_solved_and_unsolved() -> None:
    c = grpo.Curriculum(3, [64], floor=0.03, decay=0.7, seed=0)
    for _ in range(3):
        c.update((0, 64), 16, 16)  # always solved
        c.update((1, 64), 0, 16)  # never solved
        c.update((2, 64), 8, 16)  # the frontier
    w = [c.weight(c.index[(t, 64)]) for t in range(3)]
    assert w[2] > w[0] and w[2] > w[1]
    assert c.p(0) > 0.95 and c.p(1) < 0.05 and c.p(2) == pytest.approx(0.5)
    assert min(w) > 0.03  # the floor keeps them drawable
    s = c.summary()
    assert s["seen"] == 3 and s["frontier"] == pytest.approx(1 / 3)


def test_curriculum_counts_decay_per_update() -> None:
    c = grpo.Curriculum(2, [0], floor=0.0, decay=0.5, seed=0)
    c.update((0, 0), 16, 16)
    c.update((0, 0), 0, 16)  # s = 16 * 0.5 + 0, n = 16 * 0.5 + 16
    assert (c.s[0], c.n[0]) == (8.0, 24.0) and c.p(0) == pytest.approx(9 / 26)
    assert (c.s[1], c.n[1]) == (0.0, 0.0) and c.p(1) == 0.5  # untouched pairs keep their counts
    for _ in range(40):  # old evidence fades: the pass rate tracks the recent groups
        c.update((0, 0), 4, 16)
    assert c.p(0) == pytest.approx((8 + 1) / (32 + 2), abs=1e-6)


def test_curriculum_draw_is_distinct_and_covers_everything() -> None:
    c = grpo.Curriculum(5, [0, 16, 64], floor=0.03, decay=0.7, seed=1)
    c.update((0, 16), 16, 16)
    for m in (1, 4, 10):
        d = c.draw(m)
        assert len(d) == m == len(set(d)) and set(d) <= set(c.keys)
    for m in (15, 40):
        d = c.draw(m)
        assert len(d) == 15 and set(d) == set(c.keys)


def test_curriculum_draw_favours_the_frontier() -> None:
    c = grpo.Curriculum(40, [0], floor=0.03, decay=0.7, seed=0)
    for t in range(40):
        for _ in range(3):
            # 10 frontier pairs (8/16), 15 always solved, 15 never solved
            c.update((t, 0), 8 if t < 10 else (16 if t < 25 else 0), 16)
    counts = [0] * 40
    for _ in range(400):
        for t, _ in c.draw(5):
            counts[t] += 1
    frontier = sum(counts[:10]) / sum(counts)
    # weights 0.28 (frontier) vs ~0.056 (rest): a uniform draw would give 0.25
    assert frontier > 0.5
    assert min(counts[10:]) > 0  # the floor still brings the others back


# --- end to end ---------------------------------------------------------------------------------


def test_two_iterations_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    iso = isolate(tmp_path, monkeypatch)
    model = tiny_model(T.vocab_size, seed=2)
    ck = iso / "init.pt"
    save_ckpt(str(ck), model, T.mode, extra={"think_format": "think"})
    recs = [pilot_rec("k12_37120"), pilot_rec("k12_16472"), pilot_rec("k12_22530")]
    targets = iso / "targets.jsonl"
    targets.write_text("".join(json.dumps({"prompt": r["prompt"]}) + "\n" for r in recs))
    flip = random.Random(0)

    def fake_judge(tok: Any, prompts: Any, res: list[dict[str, Any]], lean: Any = None) -> Any:
        for r in res:
            r["ok"] = flip.random() < 0.5
        return res

    modes: list[str] = []
    real_generate = grpo.think_generate

    def spy_generate(*args: Any, **kw: Any) -> Any:
        modes.append(kw["force_mode"])
        return real_generate(*args, **kw)

    real_load = grpo.load_ckpt

    def eos_biased_load(path: Any, device: str = "cpu") -> Any:
        """A random model's action rarely ends within max_action (overlong, masked): favour <eos>
        (banned inside the think block) so most attempts end in a verdict."""
        m, tok, extra = real_load(path, device)
        b = torch.zeros(m.cfg["vocab"])
        b[tok.eos] = 3.0
        return bias_logits(m, b), tok, extra

    monkeypatch.setattr(grpo, "judge", fake_judge)
    monkeypatch.setattr(grpo, "think_generate", spy_generate)
    monkeypatch.setattr(grpo, "load_ckpt", eos_biased_load)
    a = grpo.parser().parse_args(
        [
            "--init",
            str(ck),
            "--name",
            "g",
            "--out",
            str(iso / "g"),
            "--targets",
            str(targets),
            "--budgets",
            "0,16",
            "--effort",
            "--iters",
            "2",
            "--prompts",
            "2",
            "--group",
            "4",
            "--updates",
            "2",
            "--max_action",
            "8",
            "--max_steps",
            "3",
            "--batch",
            "8",
            "--save_every",
            "1",
            "--device",
            "cpu",
            "--lr",
            "1e-3",
        ]
    )
    hist = grpo.run(a)
    assert len(hist) == 2 and all(h["trained_tokens"] > 0 for h in hist)
    assert all(h["steps"] == 2 and 0.0 <= h["clip_frac"] <= h["clip_frac_max"] <= 1.0 for h in hist)
    assert all(h["grad_norm"] > 0 and h["attempts_per_s"] > 0 for h in hist)
    for h in hist:
        assert 0 < h["entropy"] < math.log(T.vocab_size + 4)  # the effort vocabulary
        assert h["entropy_think"] > 0 and h["entropy_action"] > 0
        assert h["clip_low"] + h["clip_high"] == pytest.approx(h["clip_frac"])
        assert (
            set(h["think_by_budget"]) == set(h["overlong_by_budget"]) == set(h["pass1_by_budget"])
        )
        assert h["think_by_budget"].get("0", 0.0) == 0.0 < h["think_by_budget"]["16"]
        assert 0.0 <= h["overlong_frac"] < 1.0
    assert any(h["overlong_frac"] > 0 for h in hist)  # some actions still hit max_action
    args = json.loads((iso / "g" / "args.json").read_text())
    assert (args["clip_eps_low"], args["clip_eps_high"]) == (0.2, 0.28)  # resolved defaults
    assert (args["temperature"], args["overlong"]) == (1.0, "mask")
    assert modes and set(modes) == {"line"} == {args["force_mode"]}  # the sampler's default
    rows = [json.loads(x) for x in (iso / "g" / "metrics.jsonl").read_text().splitlines()]
    assert [r["iter"] for r in rows] == [1, 2]
    assert set(rows[0]["pass1_by_budget"]) <= {"0", "16"}
    assert rows[1]["curriculum"]["seen"] >= 2 and rows[1]["curriculum"]["pairs"] == 6
    assert (iso / "g" / "ckpts" / "g_it0.pt").exists()  # the effort expansion of the init
    assert (iso / "g" / "ckpts" / "g_it2.pt").exists()
