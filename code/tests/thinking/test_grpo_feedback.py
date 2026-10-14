"""In-context verifier feedback (`--feedback context`, feedback.py; Leon 2026-10-14): the dead-end
line of every action kind (the trace DSL's own skeleton parser reads each), the sampler's retries
with the lines in the think block (B > 0 and B = 0), the per-step retry cap, the counters, the
default path (feedback off, and feedback at max_rej 0, are today's runs), the training items
(feedback tokens out of the loss, the ratio and the think ban mass; a retry's think tokens scored
as think tokens; every rejected draft at -P, the 'diverge' scope matched by state), and the
logged summary of a tiny GRPO run and of an eval cell."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import torch

from code.thinking import eval as ev
from code.thinking import feedback as F
from code.thinking import grpo, stateview
from code.thinking.ckpt import save_ckpt
from code.thinking.dan import load, trace_dsl
from code.thinking.sample import phase_bans, think_generate
from code.thinking.tok import ThinkTokenizer

from ._grpo_digest import CONFIGS, run_digest, setup_go
from ._pipeline_helpers import bias_logits, isolate, pilot_rec, tiny_model
from .test_grpo_guard import _crc_judge
from .test_grpo_sched import GOLDEN as SCHED_GOLDEN

D = load()
T = ThinkTokenizer()
P = 0.25
PROMPT = pilot_rec("k12_37120")[
    "prompt"
]  # ( ~ S ), ( P > ( ( ~ P ) & R ) ) |- ( Q > ( ... & ( ~ S ) ) )
GOAL = "( Q → ( ( P → ( ( ¬ P ) ∧ R ) ) ∧ ( ¬ S ) ) )"
CONJ = "( ( P → ( ( ¬ P ) ∧ R ) ) ∧ ( ¬ S ) )"
VALID = [  # k12_37120's proof, one action per state
    "have n1 : ( ¬ S ) := h1 ;",
    "have n2 : ( P → ( ( ¬ P ) ∧ R ) ) := h2 ;",
    f"have n3 : {GOAL} := ( fun ( n4 : Q ) => by",
    f"have n5 : {CONJ} := ⟨ n2 , n1 ⟩ ;",
    "exact n5",
    "exact n3",
]
NL = T.stoi["<nl>"]


def _env(prompt: str, acts: list[str]) -> Any:
    env = D.state_env.Env(prompt, canon=True, base=0, assign=True)
    for a in acts:
        ok, why = env.apply(a.split())
        assert ok, (a, why)
    return env


def _toks(s: str) -> list[str]:
    return s.split()


def _ids(s: str) -> list[int]:
    return T.encode_toks(_toks(s))


# --- the rendering ------------------------------------------------------------------------------

PREM = VALID[:2]  # the premises restated: n1 : ( ¬ S ), n2 : ( P → ( ( ¬ P ) ∧ R ) )
RENDER = [
    # (actions before, the rejected action, its line)
    (PREM, "have n3 : ( ( ¬ P ) ∧ R ) := n2 n1 ;", "⊢ ( ( ¬ P ) ∧ R ) := n2 n1 <no>"),
    (PREM, "have n3 : R := n2 .2 ;", "⊢ R := n2 .2 <no>"),
    (PREM, "have n3 : S := n1 .elim ;", "⊢ S := n1 .elim <no>"),
    (PREM, "have n3 : ( ( ¬ S ) ∧ P ) := ⟨ n1 , n2 ⟩ ;", "⊢ ( ( ¬ S ) ∧ P ) := ⟨ n1 , n2 ⟩ <no>"),
    (PREM, "have n3 : ( S ∨ Q ) := Or.inl n1 ;", "⊢ ( S ∨ Q ) := Or.inl n1 <no>"),
    (PREM, "have n3 : ( Q ∨ S ) := Or.inr n9 ;", "⊢ ( Q ∨ S ) := Or.inr n9 <no>"),  # unbound
    (
        PREM,
        "have n3 : S := Classical.byContradiction ( fun hh => n1 hh ) ;",
        "⊢ S := Classical.byContradiction n1 <no>",  # the think surface (progtrace.r_skel)
    ),
    ([], "have n1 : S := h1 ;", "⊢ S := h1 <no>"),  # a premise line, the wrong formula
    ([], "have n1 : ( ¬ S ) := h2 ;", "⊢ ( ¬ S ) := h2 <no>"),  # ... the wrong position
    (
        PREM,
        "have n3 : ( Q → S ) := ( fun ( n4 : Q ) => by",
        "⊢ ( Q → S ) := fun ( n4 : Q ) => ⊢ S <no>",
    ),
    (
        PREM,
        "have n3 : ( ¬ Q ) := ( fun ( n4 : Q ) => by",
        "⊢ ( ¬ Q ) := fun ( n4 : Q ) => ⊢ False <no>",
    ),
    (PREM, "have n3 : Q := ( fun ( n4 : P ) => by", "⊢ Q := fun ( n4 : P ) => ⊢ Q <no>"),
    (PREM, "exact n2", f"⊢ {GOAL} := n2 <no>"),
]
UNRENDERED = [
    "have n3 : ( P ∧ Q ) := ⟨ ⊢ P , ⊢ Q ⟩ ;",  # think tokens in the action
    "have n3 : R := n2 .2",  # no semicolon
    "have n3 : R := n2 n1 n1 ;",
    "have n3 : ( P ∧ := n2 ;",
    "have n3 :",
    "have n3 : R := ( fun ( n4 : P ) => by ;",
    "have n3 : R := Or.elim ( fun ( n4 : P ) => by",
    "have n3 : R := Or.elim n1 ( fun ( n4 : S ) => by",  # n1 is no disjunction: no second branch
    "have n3 : R := Or.elim n9 ( fun ( n4 : S ) => by",  # n9 is not in scope
    "exact ( n2 )",
    "exact",
    "",
]


def _skel_parses(line: list[str]) -> None:
    """The trace DSL's skeleton parser reads the line's reduction whole, and F is a formula."""
    TD = trace_dsl()
    assert line[0] == "⊢" and line[-1] == "<no>" and ":=" in line
    i = line.index(":=")
    _, used = D.state_env.parse_ftoks(line, 1)
    assert used == i - 1
    sk = line[i + 1 : -1]
    p = TD.Skel(sk)
    p.skel()
    assert p.i == len(sk), sk


@pytest.mark.parametrize(("before", "act", "want"), RENDER)
def test_dead_end_line_of_every_action_kind(before: list[str], act: str, want: str) -> None:
    env = _env(PROMPT, before)
    line = F.render(env, act.split())
    assert line is not None and " ".join(line) == want
    assert all(t in T.stoi for t in line)
    _skel_parses(line)


def test_or_elim_names_the_second_branch() -> None:
    env = _env("THM ( P v Q ) SEQ ( Q v P ) PRF", ["have n1 : ( P ∨ Q ) := h1 ;"])
    line = F.render(env, _toks("have n2 : ( Q ∨ P ) := Or.elim n1 ( fun ( n3 : Q ) => by"))
    assert line is not None
    want = (
        "⊢ ( Q ∨ P ) := Or.elim n1 ( fun ( n3 : Q ) => ⊢ ( Q ∨ P ) ) "
        "( fun ( n4 : Q ) => ⊢ ( Q ∨ P ) ) <no>"
    )
    assert " ".join(line) == want
    _skel_parses(line)


@pytest.mark.parametrize("act", UNRENDERED)
def test_malformed_actions_have_no_line(act: str) -> None:
    assert F.render(_env(PROMPT, PREM), act.split()) is None


def test_block_joins_lines_as_a_think_block() -> None:
    assert F.block([], NL) == []
    assert F.block([[1, 2], [3]], NL) == [1, 2, NL, 3]


# --- the sampler: a scripted policy that sees its feedback ---------------------------------------


class FBScripted(torch.nn.Module):
    """After the prompt of a known state, writes `script[(state, n)]` (all logit mass on it), n
    the feedback lines (`<no>` tokens) prefilled after the state (`key='lines'`: a policy that
    reads its feedback) or the draws made at that state before (`key='draws'`); `<eos>`
    otherwise. Per-row history and prompt length live in the KV cache as tensors (the sampler
    compacts them)."""

    def __init__(
        self, tok: Any, script: dict[tuple[bytes, int], list[int]], key: str = "lines"
    ) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Identity()])
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.tok, self.script, self.key = tok, script, key
        self.act = tok.stoi["<act>"]
        self.seen: dict[bytes, int] = {}

    def forward(
        self, idx: torch.Tensor, pos: Any = None, mask: Any = None, caches: Any = None
    ) -> Any:
        B, L = idx.shape
        c = caches[0]
        first = "hist" not in c
        if first:
            c["hist"] = torch.full((B, c["max"] + L), -1, dtype=torch.long)
            c["len"] = torch.zeros(B, dtype=torch.long)
            c["plen"] = torch.zeros(B, dtype=torch.long)
            c["n"] = torch.zeros(B, dtype=torch.long)
        for b in range(B):
            toks = [int(x) for x in idx[b] if L == 1 or int(x) != self.tok.pad]
            n = int(c["len"][b])
            c["hist"][b, n : n + len(toks)] = torch.tensor(toks, dtype=torch.long)
            c["len"][b] = n + len(toks)
            if first:
                c["plen"][b] = len(toks)
                j = len(toks) - 1 - toks[::-1].index(self.act)
                st = bytes(toks[: j + 1])
                c["n"][b] = (
                    toks[j + 1 :].count(self.tok.no)
                    if self.key == "lines"
                    else self.seen.get(st, 0)
                )
                self.seen[st] = self.seen.get(st, 0) + 1
        out = torch.full((B, 1, self.tok.vocab_size), -30.0)
        for b in range(B):
            h = c["hist"][b, : int(c["len"][b])].tolist()
            pl = int(c["plen"][b])
            j = pl - 1 - h[:pl][::-1].index(self.act)
            cont = self.script.get((bytes(h[: j + 1]), int(c["n"][b])), [])
            gen = h[pl:]
            out[b, 0, cont[len(gen)] if len(gen) < len(cont) else self.tok.eos] = 30.0
        return out


THINK = [T.stoi["P"]]  # every scripted draw's think content (B > 0)
BAD = {  # state index -> the actions rejected there, in order (one per feedback line already seen)
    0: ["have n1 : ( ¬ S ) := h2 ;", "have n1 : S := h1 ;"],  # S:PR position, then L:
    3: [f"have n5 : {CONJ} := ⟨ n1 , n2 ⟩ ;"],  # L:and-intro
}
LINES = {
    0: [_ids("⊢ ( ¬ S ) := h2 <no>"), _ids("⊢ S := h1 <no>")],
    3: [_ids(f"⊢ {CONJ} := ⟨ n1 , n2 ⟩ <no>")],
}


def _states() -> list[list[int]]:
    env = D.state_env.Env(PROMPT, canon=True, base=0, assign=True)
    out = []
    for a in VALID:
        out.append(T.encode_toks(stateview.render(env)))
        assert env.apply(a.split())[0]
    return out


def _script(budget: int, bad: dict[int, list[str]]) -> dict[tuple[bytes, int], list[int]]:
    head = [] if budget == 0 else [*THINK, T.end_think]
    out = {}
    for t, s in enumerate(_states()):
        acts = [*bad.get(t, []), VALID[t]]
        for n, a in enumerate(acts):
            out[(bytes(s), n)] = [*head, *_ids(a), T.eos]
    return out


def _sample(
    budget: int, max_rej: int, bad: dict[int, list[str]] = BAD, n: int = 2, **kw: Any
) -> list[dict[str, Any]]:
    kw = {"feedback": "context", "keep_rejected": True, **kw}
    return think_generate(
        FBScripted(T, _script(budget, bad)),
        T,
        [PROMPT] * n,
        "logical",
        budgets=budget,
        max_action=64,
        max_steps=10,
        max_rej=max_rej,
        batch=n,
        seed=3,
        **kw,
    )


@pytest.fixture
def base0(monkeypatch: pytest.MonkeyPatch) -> None:
    """Name base 0 for every attempt: the scripted states' names are VALID's."""
    monkeypatch.setattr(D.state_sample, "NAME_BASE_MAX", 0)


scripted = pytest.mark.usefixtures("base0")


def _prefix(budget: int) -> list[int]:
    return [T.think] + ([T.end_think] if budget == 0 else [])


@scripted
@pytest.mark.parametrize("budget", [16, 0])
def test_retries_see_the_rejected_actions_in_the_think_block(budget: int) -> None:
    res = _sample(budget, max_rej=2)
    S = _states()
    for r in res:
        assert r["status"] == "done" and r["steps"] == 6, r["status"]
        assert [c[:2] for c in r["causes"]] == ["S:", "L:", "L:"]
        tr = r["trace"]
        # the accepted step 0 came after two rejections: both lines, then the model's own tokens
        tail = [NL] if budget else [T.end_think]
        want0 = S[0] + [T.think] + F.block(LINES[0], NL) + tail
        assert list(tr[0]["prompt"]) == want0
        assert tr[0]["fb"] == len(F.block(LINES[0], NL)) + (budget > 0)
        head = [] if budget == 0 else [*THINK, T.end_think]
        assert list(tr[0]["gen"]) == [*head, *_ids(VALID[0]), T.eos]  # no feedback token in gen
        assert list(tr[3]["prompt"]) == S[3] + [T.think] + F.block(LINES[3], NL) + tail
        # the lines are reset once a draw is accepted
        for t in (1, 2, 4, 5):
            assert list(tr[t]["prompt"]) == S[t] + _prefix(budget) and tr[t]["fb"] == 0
        # the rejected drafts, each under the lines it saw
        red = r["redrawn"]
        assert [x["step"] for x in red] == [0, 0, 3]
        assert [list(x["prompt"]) for x in red] == [
            S[0] + _prefix(budget),
            S[0] + [T.think] + F.block(LINES[0][:1], NL) + tail,
            S[3] + _prefix(budget),
        ]
        assert [x["fb"] for x in red] == [0, len(LINES[0][0]) + (budget > 0), 0]
        n1, n2, n3 = (len(LINES[0][0]) + (budget > 0), tr[0]["fb"], tr[3]["fb"])
        assert r["fb"] == {
            "draws": 3,
            "tokens": n1 + n2 + n3,
            "lines": 3,
            "unrendered": 0,
            "rejections": 3,
            "retry_draws": 3,
            "retry_acc": 2,
            "rej_after_rej": 1,
            "repeats": 0,
        }
        assert r["rej"] == 3 and r["draws"] == 9 and grpo.n_rejected(r) == 3


@scripted
def test_max_rej_counts_retries_per_step() -> None:
    # two rejections at state 0: max_rej 1 ends the attempt there, max_rej 2 recovers
    one = _sample(16, max_rej=1)
    for r in one:
        assert r["status"] == "fail:max_rej" and r["fail_step"] == 0 and len(r["causes"]) == 2
        end = r["trace"][-1]
        assert end["rejected"] and list(end["prompt"])[-len(LINES[0][0]) - 1 :] == [
            *LINES[0][0],
            NL,
        ]
        assert (
            r["fb"]["rejections"] == 2 and r["fb"]["retry_draws"] == 1 and r["fb"]["retry_acc"] == 0
        )
        assert len(r["redrawn"]) == 1
    # per step, not per attempt: three rejections in all, never more than two at a state
    assert all(r["status"] == "done" for r in _sample(16, max_rej=2))


@scripted
def test_a_repeated_action_adds_no_line_and_counts_as_a_repeat() -> None:
    bad = {0: ["have n1 : S := h1 ;"] * 2}  # the second draw repeats the first
    script = _script(16, bad)
    s0 = bytes(_states()[0])
    script[(s0, 1)] = script[(s0, 0)]  # seeing the line, it repeats again: no new line ever
    res = think_generate(
        FBScripted(T, script),
        T,
        [PROMPT],
        "logical",
        budgets=16,
        max_action=64,
        max_steps=10,
        max_rej=3,
        batch=1,
        seed=3,
        feedback="context",
    )
    r = res[0]
    assert r["status"] == "fail:max_rej" and len(r["causes"]) == 4
    assert r["fb"]["lines"] == 1 and r["fb"]["repeats"] == 3 and r["fb"]["rej_after_rej"] == 3
    m = F.metrics(res)
    assert m["repeat_rate"] == 1.0 and m["repeat_share"] == 0.75 and m["retry_accept"] == 0.0


@scripted
def test_an_unrenderable_rejection_is_a_plain_resample() -> None:
    # the first draw at state 0 is malformed (no line), the second (still no line) the valid one
    script = _script(16, {0: ["have n1 : ( ¬ S ) := ⟨ ⊢ P , ⊢ Q ⟩ ;"]})
    res = think_generate(
        FBScripted(T, script, key="draws"),
        T,
        [PROMPT],
        "logical",
        budgets=16,
        max_action=64,
        max_steps=10,
        max_rej=2,
        batch=1,
        seed=3,
        feedback="context",
        keep_rejected=True,
    )
    r = res[0]
    assert r["status"] == "done" and len(r["causes"]) == 1 and r["causes"][0].startswith("S:")
    assert r["fb"]["unrendered"] == 1 and r["fb"]["lines"] == 0 and r["fb"]["draws"] == 0
    assert r["fb"]["retry_draws"] == 1 and r["fb"]["retry_acc"] == 1
    assert list(r["trace"][0]["prompt"]) == _states()[0] + [T.think]  # no line in the retry


def test_feedback_needs_a_guided_arm_and_the_per_step_think_format() -> None:
    m = FBScripted(T, {})
    with pytest.raises(ValueError, match="verifier's arms"):
        think_generate(m, T, [PROMPT], "plain", budgets=16, feedback="context")
    with pytest.raises(ValueError, match="per-step format"):
        think_generate(
            m, T, [PROMPT], "logical", budgets=16, feedback="context", think_format="root"
        )
    with pytest.raises(ValueError, match="per-step format"):
        old = D.lean_tok.LeanTokenizer("lean_staten")
        think_generate(tiny_model(old.vocab_size), old, [PROMPT], "logical", feedback="context")
    with pytest.raises(AssertionError):
        think_generate(m, T, [PROMPT], "logical", budgets=16, feedback="yes")


# --- the default path ---------------------------------------------------------------------------


def _tiny_guided(**kw: Any) -> list[dict[str, Any]]:
    b = torch.zeros(T.vocab_size)
    b[T.eos] = 3.0
    model = bias_logits(tiny_model(T.vocab_size, seed=3), b)
    prompts = [pilot_rec(r)["prompt"] for r in ("k12_37120", "k12_16472") for _ in range(4)]
    return think_generate(
        model,
        T,
        prompts,
        "logical",
        budgets=[0, 16],
        max_action=8,
        max_steps=4,
        batch=8,
        seed=5,
        keep_rejected=True,
        **kw,
    )


def _drop_fb(res: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for x in res:
        y = {k: v for k, v in x.items() if k != "fb"}
        for key in ("trace", "redrawn"):
            if y.get(key) is not None:
                y[key] = [{k: v for k, v in s.items() if k != "fb"} for s in y[key]]
        out.append(y)
    return out


def test_feedback_off_is_the_default_and_max_rej_0_is_the_stop_arm() -> None:
    base = _tiny_guided(max_rej=0)
    assert _tiny_guided(max_rej=0, feedback="none") == base
    fb = _tiny_guided(max_rej=0, feedback="context")
    assert all("fb" in x for x in fb) and sum(len(x["causes"]) for x in fb) > 0
    assert _drop_fb(fb) == base  # the first rejection ends the attempt: no line is ever shown


def test_grpo_feedback_none_is_the_stop_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """test_grpo_sched.py's verifier_stop digests (fa28f6e's) with --feedback none given: the same
    rollouts, rows, lines and weights; args.json records the option."""
    stop = CONFIGS["verifier_stop"]
    monkeypatch.setitem(CONFIGS, "fb_none", lambda iso: [*stop(iso), "--feedback", "none"])
    got = run_digest(tmp_path, monkeypatch, "fb_none")
    gold = SCHED_GOLDEN["verifier_stop"]
    assert {k: got[k] for k in ("rollouts", "rows", "lines", "weights")} == {
        k: gold[k] for k in ("rollouts", "rows", "lines", "weights")
    }
    assert got["args"] != gold["args"]


def test_grpo_feedback_at_max_rej_0_trains_the_stop_runs_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stop = CONFIGS["verifier_stop"]
    monkeypatch.setitem(CONFIGS, "fb_stop", lambda iso: [*stop(iso), "--feedback", "context"])
    got = run_digest(tmp_path, monkeypatch, "fb_stop")
    assert got["weights"] == SCHED_GOLDEN["verifier_stop"]["weights"]
    assert got["rows"] != SCHED_GOLDEN["verifier_stop"]["rows"]  # + row['feedback']


def test_the_option_is_left_out_unless_given() -> None:
    base = ["--init", "x", "--name", "g", "--out", "o", "--targets", "t"]
    assert "feedback" not in vars(grpo.parser().parse_args(base))
    assert grpo.parser().parse_args([*base, "--feedback", "context"]).feedback == "context"
    with pytest.raises(SystemExit):
        grpo.parser().parse_args([*base, "--feedback", "lean"])


# --- training: masking and the step penalty -----------------------------------------------------


def _fb_attempts() -> list[dict[str, Any]]:
    """Two scripted attempts with feedback: one solved after its retries (ok), one ended at state
    0 by the per-step cap (max_rej 1)."""
    ok = _sample(16, max_rej=2, n=1)[0]
    ok["ok"] = True
    bad = _sample(16, max_rej=1, n=1)[0]
    bad["ok"] = False
    return [ok, bad]


@scripted
def test_feedback_tokens_are_prompt_tokens_and_the_retry_thinks() -> None:
    att = _fb_attempts()
    src: list[tuple[int, str]] = []
    items, _ = grpo.group_items(att, T, 0.0, "fail", rej_penalty=P, src=src)
    assert items is not None
    ids = (T.think, T.end_think)
    n_line = len(F.block(LINES[0], NL)) + 1
    retry = items[0]  # attempt 0's accepted step 0: the retry after two lines
    p, g, m, _ = retry
    assert p[-n_line:] == F.block(LINES[0], NL) + [NL] and len(m) == len(g) and all(m)
    assert grpo.open_think(p, ids) and grpo.think_len(p, g, ids) == len(THINK) + 1
    # without the lines the same prompt is an ordinary one, and budget 0's closed block is none
    assert grpo.open_think(p[:-n_line], ids)
    assert not grpo.open_think([*p, T.end_think], ids)
    model = tiny_model(T.vocab_size, seed=4)
    bans = phase_bans(T, T.vocab_size, True)
    s = grpo._token_logp(model, [retry], 1.0, "cpu", entropy=True, think_ids=ids, bans=bans)
    scored = s.m[0].nonzero().flatten().tolist()
    assert scored == list(range(len(p) - 1, len(p) - 1 + len(g)))  # gen only: no feedback token
    th = s.th[0].nonzero().flatten().tolist()
    assert th == list(
        range(len(p) - 1, len(p) + len(THINK))
    )  # the retry's think block, </think> too
    # the think ban mass reads the scored think positions only (`policy_loss`'s m & th)
    assert s.ban_mass is not None
    want = float(s.ban_mass[0, len(p) - 1 : len(p) + len(THINK)].mean())
    assert grpo.ban_mass(model, [retry], 1.0, "cpu", think_ids=ids, bans=bans) == pytest.approx(
        want
    )
    # the PPO ratio's old log-probs: one per scored (generated) token of every item
    old = grpo.old_logps(model, items, 1.0, "cpu", think_ids=ids, bans=bans)
    assert [int(o.numel()) for o in old] == [sum(it[2]) for it in items]
    assert all(torch.isfinite(o).all() for o in old)


@scripted
def test_every_rejected_draft_costs_minus_p() -> None:
    att = _fb_attempts()
    src: list[tuple[int, str]] = []
    items, masked = grpo.group_items(att, T, 0.0, "fail", rej_penalty=P, src=src)
    assert items is not None and masked == [False, False]
    advs = [round(it[3], 6) for it in items]
    # attempt 0: six accepted steps at +0.5 (the retries too), its three rejected drafts at -P;
    # attempt 1: the draw that ended it at -0.5 - P, its one rejected draft at -P
    assert advs == [0.5] * 6 + [-P] * 3 + [round(-0.5 - P, 6), -P]
    assert src == [(0, "acc")] * 6 + [(0, "rej")] * 3 + [(1, "acc"), (1, "rej")]
    # 'diverge': a rejected draft is compared with the draw accepted at its state (the prompts
    # differ by the lines): the shared think block and `have n1 : ( ¬ S ) :=` are not scored
    acc0 = att[0]["trace"][0]
    for it, red in zip(items[6:8], att[0]["redrawn"][:2], strict=True):
        n = grpo._lcp(list(red["gen"]), list(acc0["gen"]))
        assert n > len(THINK) + 1 and grpo.same_state(acc0, red)
        assert not any(it[2][:n]) and all(it[2][n:])
    # no outcome signal (both solved): only the rejected drafts are trained, at -P
    both = [att[0], {**att[0]}]
    its, _ = grpo.group_items(both, T, 0.0, "fail", rej_penalty=P)
    assert its is not None and [round(x[3], 6) for x in its] == [-P] * 6
    # without the penalty a no-signal group is skipped, as before
    assert grpo.group_items(both, T, 0.0, "fail")[0] is None


def test_same_state_matches_prompts_without_feedback() -> None:
    a = {"state": b"\x01", "prompt": b"\x01\x02"}
    assert grpo.same_state(a, {"state": b"\x01", "prompt": b"\x01\x02"})
    assert not grpo.same_state(a, {"state": b"\x01", "prompt": b"\x01\x03"})  # today's rule
    assert grpo.same_state({**a, "fb": 0}, {"state": b"\x01", "prompt": b"\x01\x03", "fb": 4})


# --- logging ------------------------------------------------------------------------------------


@scripted
def test_metrics_of_known_attempts() -> None:
    res = _sample(16, max_rej=2) + _sample(16, max_rej=1)
    for x, ok in zip(res, (True, True, False, False), strict=True):
        x["ok"] = ok
    m = F.metrics(res, [16, 16, 0, 0])
    assert m["attempts"] == 4 and m["rej_per_attempt"] == (3 + 3 + 2 + 2) / 4
    assert m["retries_per_attempt"] == (3 + 3 + 1 + 1) / 4
    assert (m["retry_draws"], m["retry_acc"]) == (3 + 3 + 1 + 1, 2 + 2)
    assert m["retry_accept"] == 4 / 8 and m["repeats"] == 0 and m["repeat_rate"] == 0.0
    assert m["unrendered"] == 0.0 and m["solved_assisted"] == 1.0
    assert (
        m["by_budget"]["0"]["retry_accept"] == 0.0 and m["by_budget"]["16"]["retry_accept"] == 4 / 6
    )
    assert "retry acc 50% (4/8)" in F.line_text(m)


def _go_feedback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: list[str]) -> Any:
    go = setup_go(tmp_path, monkeypatch)
    rolls: list[tuple[int, dict[str, Any]]] = []  # (budget, attempt), in sampling order
    real = grpo.think_generate

    def spy(*a: Any, **kw: Any) -> Any:
        res = real(*a, **kw)
        rolls.extend((int(kw["budgets"][0]), x) for x in res)
        return res

    monkeypatch.setattr(grpo, "think_generate", spy)
    argv = ["--iters", "2", "--arm", "logical", "--max_rej", "2", "--rej_penalty", "0.1", *extra]
    hist, out = go(argv)
    return hist, out, rolls


def test_grpo_logs_the_feedback_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    hist, out, rolls = _go_feedback(tmp_path, monkeypatch, ["--feedback", "context", "--monitor"])
    lines = capsys.readouterr().out
    assert json.loads((out / "args.json").read_text())["feedback"] == "context"
    n_it = len(rolls) // len(hist)  # 2 prompts x 4 attempts per iteration
    for k, row in enumerate(hist):
        fb = row["feedback"]
        chunk = rolls[k * n_it : (k + 1) * n_it]
        flat = [x for _, x in chunk]
        want = F.metrics(flat, [b for b, _ in chunk])
        assert {kk: fb[kk] for kk in want} == want
        assert fb["max_rej"] == 2
        assert sum(v["draws"] for v in fb["retry_accept_by_k"].values()) == fb["retry_draws"]
        assert sum(v["acc"] for v in fb["retry_accept_by_k"].values()) == fb["retry_acc"]
        assert all(int(k2) <= 2 for k2 in fb["retry_accept_by_k"])  # never past --max_rej
        assert re.search(rf"^fb it {row['iter']}: rej/att ", lines, re.M)
        # every attempt's rejections at a state stay within the cap
        for x in flat:
            assert x["fb"]["rejections"] == len(x["causes"])
            assert x["status"] != "fail:max_rej" or x["fb"]["rejections"] >= 3
    assert sum(r["feedback"]["rej_per_attempt"] for r in hist) > 0, "the tiny model is rejected"


def test_grpo_without_feedback_logs_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hist, out, _ = _go_feedback(tmp_path, monkeypatch, [])
    assert all("feedback" not in r for r in hist)
    assert "feedback" not in json.loads((out / "args.json").read_text())


def test_grpo_feedback_refuses_the_plain_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    go = setup_go(tmp_path, monkeypatch)
    with pytest.raises(AssertionError, match="guided --arm"):
        go(["--iters", "1", "--feedback", "context"])


def test_eval_feedback_cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    iso = isolate(tmp_path, monkeypatch)
    ck = iso / "m.pt"
    save_ckpt(str(ck), tiny_model(T.vocab_size, seed=2), T.mode, {"think_format": "think"})
    real = ev.load_ckpt

    def biased(path: Any, device: str = "cpu") -> Any:
        m, tok, x = real(path, device)
        b = torch.zeros(m.cfg["vocab"])
        b[tok.eos] = 3.0
        return bias_logits(m, b), tok, x

    monkeypatch.setattr(ev, "load_ckpt", biased)
    monkeypatch.setattr(ev, "judge", _crc_judge)
    sets = iso / "set.jsonl"
    rows = [{"name": r, "prompt": pilot_rec(r)["prompt"]} for r in ("k12_37120", "k12_16472")]
    sets.write_text("".join(json.dumps(r) + "\n" for r in rows))
    summ = ev.run(
        ev.parser().parse_args(
            [
                "--ckpt",
                str(ck),
                "--sets",
                str(sets),
                "--out",
                str(iso / "ev"),
                "--k",
                "4",
                "--budgets",
                "16",
                "--arms",
                "logical,feedback",
                "--max_action",
                "8",
                "--max_steps",
                "4",
                "--max_rej",
                "2",
                "--device",
                "cpu",
                "--dump_traces",
            ]
        )
    )
    cell = summ["cells"]["feedback_B16"]
    assert cell["feedback"]["max_rej"] == 2 and cell["feedback"]["attempts"] == 8
    assert "feedback" not in summ["cells"]["logical_B16"]
    got = [json.loads(x) for x in (iso / "ev" / "rows_feedback_B16.jsonl").read_text().splitlines()]
    assert all(set(r["fb"]) == set(F.KEYS) and len(r["fb"]["rejections"]) == 4 for r in got)
    assert sum(sum(r["fb"]["rejections"]) for r in got) == cell["feedback"]["rej_per_attempt"] * 8
    lg = [json.loads(x) for x in (iso / "ev" / "rows_logical_B16.jsonl").read_text().splitlines()]
    assert all("fb" not in r for r in lg)
