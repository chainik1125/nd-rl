"""State formats (`stateview`): flat vs goalstack rendering, the process switch, the dataset
wrapper, train -> checkpoint -> load, the sampler's prompts."""

from __future__ import annotations

import collections
import json
import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from code.thinking import ckpt as C
from code.thinking import stateview as SV
from code.thinking import train
from code.thinking.dan import trace_dsl
from code.thinking.data import heldout_records, plain_pairs, trace_pairs
from code.thinking.sample import think_generate
from code.thinking.tok import D, EffortTokenizer, ThinkTokenizer

from ._pipeline_helpers import (
    HELDOUT,
    ScriptedModel,
    isolate,
    k12_head,
    needs_k12,
    pilot,
    script_from_pairs,
    tiny_model,
)

SE = D.state_env
T = ThinkTokenizer()
OLD = D.lean_tok.LeanTokenizer("lean_staten")
GOAL, NL, ACT = "⊢", "<nl>", "<act>"


@pytest.fixture(autouse=True)
def _clean_format() -> Iterator[None]:
    """Each test starts and ends with no state format set (flat, the class unpatched)."""
    SV.reset_state_format()
    yield
    SV.reset_state_format()


def _actions(rec: dict[str, Any]) -> list[list[str]]:
    return SE.split_actions(SE.canonicalise(rec["prompt"], rec["proof"]))


def _replay(rec: dict[str, Any]) -> Iterator[tuple[Any, list[str]]]:
    """(environment before the action, action) for each step of a K12 proof."""
    env = SE.Env(rec["prompt"], canon=True)
    for a in _actions(rec):
        yield env, a
        ok, why = env.apply(a)
        assert ok, why
    assert env.done


def expected_goals(rec: dict[str, Any]) -> list[list[Any]]:
    """Per step, the open frames' goals computed from the action text alone (not from
    `Env.frames`): the theorem's conclusion; `have F := ( fun ...` opens a box proving F's
    consequent (or False for a negation); `Or.elim` opens a branch proving F; `exact` closes the
    innermost box (an Or.elim's first branch becomes its second, same goal)."""
    _, concl, _ = SE.prompt_parts(rec["prompt"])
    stack: list[tuple[str, Any]] = [("top", concl)]
    out = []
    for a in _actions(rec):
        out.append([g for _, g in stack])
        if a[0] == "exact":
            kind, g = stack[-1]
            if kind == "or1":
                stack[-1] = ("or2", g)
            elif kind != "top":
                stack.pop()
            continue
        f, used = SE.parse_ftoks(a, 3)  # have nX : F := term
        term = a[3 + used + 1]
        if term == "(":
            stack.append(("neg", SE.BOT) if f[0] == "not" else ("imp", f[2]))
        elif term == "Or.elim":
            stack.append(("or1", f))
    return out


def parse_goalstack(toks: list[str]) -> tuple[list[list[list[str]]], list[list[str]]]:
    """goalstack tokens -> (per frame its hypothesis lines, the frames' goals in order)."""
    assert toks[0] == "<st>" and toks[-1] == ACT
    frames: list[list[list[str]]] = [[]]
    goals: list[list[str]] = []
    line: list[str] = []
    for t in toks[1:]:
        line.append(t)
        if t in (NL, ACT):
            if line[0] == GOAL:
                goals.append(line[1:-1])
                if t == NL:
                    frames.append([])
            else:
                assert t == NL, "a hypothesis line ends in <nl>"
                frames[-1].append(line)
            line = []
    assert len(frames) == len(goals)
    return frames, goals


# ------------------------------------------------------------------------------------- rendering
@needs_k12
def test_single_frame_states_are_identical_to_flat() -> None:
    n = 0
    for rec in k12_head(300):
        for env, _ in _replay(rec):
            if len(env.frames) == 1:
                assert SV.render(env, "goalstack") == SV.render(env, "flat") == env.state_tokens()
                n += 1
    assert n > 300  # every initial state at least


@needs_k12
def test_goals_in_order_at_every_nested_depth() -> None:
    """Replay K12 proofs: every state's goalstack shows each open frame's hypotheses then its goal,
    outermost first; dropping the enclosing goal lines gives the flat state."""
    seen: collections.Counter[tuple[str, ...]] = collections.Counter()
    for rec in k12_head(3000):
        for (env, _), goals in zip(_replay(rec), expected_goals(rec), strict=True):
            flat, gs = SV.render(env, "flat"), SV.render(env, "goalstack")
            frames, shown = parse_goalstack(gs)
            assert shown == [SE.ftoks(g) for g in goals]
            assert frames == [list(map(list, fr.htoks)) for fr in env.frames]
            assert all(frames[1:]), "a box always shows its binder"
            assert gs.count(GOAL) == len(env.frames) and flat.count(GOAL) == 1
            inner = [t for fr in frames for line in fr for t in line]
            assert flat == ["<st>", *inner, GOAL, *shown[-1], ACT]
            extra = sum(2 + len(g) for g in shown[:-1])
            assert len(gs) == len(flat) + extra
            seen[tuple(fr.kind for fr in env.frames[1:])] += 1
    # an implication box inside an Or.elim branch, Or.elim inside an implication box, ...
    for pat in [
        ("or2", "imp"),
        ("imp", "or1"),
        ("imp", "or2"),
        ("imp", "imp"),
        ("or2", "neg"),
        ("imp", "neg"),
        ("neg", "or1"),
        ("neg", "or2"),
    ]:
        assert seen[pat] > 0, (pat, seen)


@needs_k12
def test_imp_box_inside_an_or_elim_branch_example() -> None:
    """K12 #84 at its first `top > or2 > imp` state, written out."""
    (rec,) = [r for i, r in enumerate(k12_head(85)) if i == 84]
    env = next(e for e, _ in _replay(rec) if [f.kind for f in e.frames] == ["top", "or2", "imp"])
    hyps = (
        "h1 : P <nl> h2 : ( ( ( ¬ R ) → P ) ∨ ( ( R ∨ Q ) ∧ R ) ) <nl> n1 : P <nl> "
        "n2 : ( ( ( ¬ R ) → P ) ∨ ( ( R ∨ Q ) ∧ R ) ) <nl>"
    )
    assert " ".join(SV.render(env, "flat")) == (
        f"<st> {hyps} n4 : ( ( R ∨ Q ) ∧ R ) <nl> n6 : ( ¬ R ) <nl> ⊢ P <act>"
    )
    assert " ".join(SV.render(env, "goalstack")) == (
        f"<st> {hyps} ⊢ ( ( ¬ R ) → P ) <nl> n4 : ( ( R ∨ Q ) ∧ R ) <nl> ⊢ ( ( ¬ R ) → P ) <nl> "
        "n6 : ( ¬ R ) <nl> ⊢ P <act>"
    )


@needs_k12
def test_token_ids_stay_in_the_vocabularies() -> None:
    toks = (T, OLD, EffortTokenizer())
    for rec in k12_head(500):
        for env, _ in _replay(rec):
            gs = SV.render(env, "goalstack")
            for tok in toks:
                ids = tok.encode_toks(gs)
                assert len(ids) == len(gs) and max(ids) < tok.vocab_size <= 256


@needs_k12
def test_pair_lengths_against_flat() -> None:
    """goalstack adds `⊢ G <nl>` per enclosing frame; on 3,000 K12 proofs no pair nears 1,024."""
    d_state, flat_pair, gs_pair = [], [], []
    for rec in k12_head(3000):
        with SV.state_format("flat"):
            fp = plain_pairs(rec, T, "empty")
        with SV.state_format("goalstack"):
            gp = plain_pairs(rec, T, "empty")
        assert [q for _, q, _ in fp] == [q for _, q, _ in gp]
        for (p0, q0, _), (p1, q1, _) in zip(fp, gp, strict=True):
            d_state.append(len(p1) - len(p0))
            flat_pair.append(len(p0) + len(q0))
            gs_pair.append(len(p1) + len(q1))
    assert min(d_state) == 0 and max(gs_pair) <= 1024
    assert max(gs_pair) >= max(flat_pair) and sum(d > 0 for d in d_state) > len(d_state) // 10


# --------------------------------------------------------------------------- the process switch
def test_one_format_per_process() -> None:
    assert SV.current() == "flat" and not SV.is_set()
    assert SE.Env.state_tokens is SV._FLAT
    SV.set_state_format("goalstack")
    SV.set_state_format("goalstack")  # idempotent
    assert SE.Env.state_tokens is not SV._FLAT
    with pytest.raises(SV.StateFormatError):
        SV.set_state_format("flat")
    with pytest.raises(ValueError):
        SV.check_format("stack")
    SV.reset_state_format()
    SV.set_state_format("flat")
    with pytest.raises(SV.StateFormatError):
        SV.set_state_format("goalstack")
    assert SE.Env.state_tokens is SV._FLAT


def test_dataset_format_defaults_to_flat() -> None:
    assert SV.data_format({"config": {}}) == "flat" and SV.ckpt_format({}) == "flat"
    assert SV.data_format({"state_format": "goalstack", "config": {}}) == "goalstack"
    assert SV.data_format({"config": {"state_format": "goalstack"}}) == "goalstack"


def test_patched_env_reaches_the_fork_and_the_trace_builder() -> None:
    """Under goalstack, `Env.state_tokens` (what `decompose`, `data.trace_pairs` and the fork's
    samplers call) renders goalstack; the pilot traces' nested states show their goals."""
    nested = 0
    with SV.state_format("goalstack"):
        for rec in pilot():
            pairs = trace_pairs(rec, T)
            env = SE.Env(rec["prompt"], canon=True, base=0, assign=False)
            for (p, _, _), st in zip(pairs, rec["steps"], strict=True):
                assert env.state_tokens() == SV.render(env, "goalstack")
                assert p == bytes(T.encode_toks(SV.render(env, "goalstack")))
                nested += len(env.frames) > 1
                ok, why = env.apply(trace_dsl().tokens(st["action"]))
                assert ok, why
    assert nested > 0


# ------------------------------------------------------------------------------- dataset -> model
def _k12_file(tmp: Path, n: int) -> Path:
    path = tmp / f"k12_{n}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in k12_head(n)))
    return path


@needs_k12
def test_wrapper_train_checkpoint_load_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp = isolate(tmp_path, monkeypatch)
    plain = _k12_file(tmp, 200)
    held = tmp / "held.jsonl"
    held.write_text("".join(HELDOUT.read_text().splitlines(keepends=True)[:40]))
    pairs = tmp / "pairs_g.pkl"
    argv = ["--plain", str(plain), "--no-trace-mode", "none", "--exclude", str(held)]
    assert SV.main(["data", "--state-format", "goalstack", "--", *argv, "--out", str(pairs)]) == 0
    with open(pairs, "rb") as f:
        data = pickle.load(f)
    rep = json.loads((tmp / "pairs_g.pkl.report.json").read_text())
    assert data["state_format"] == "goalstack" == rep["config"]["state_format"]
    assert rep["plain_used"] == 200 and rep["lengths"]["pairs_over_max_len"] == 0
    states = [p for r in data["records"] for p, _, _ in r]
    assert max(p.count(T.stoi[GOAL]) for p in states) >= 3
    flat_states = []  # the same records rendered flat: identical where one frame is open
    with SV.state_format("flat"):
        for r in [json.loads(x) for x in plain.read_text().splitlines()]:
            flat_states += [p for p, _, _ in plain_pairs(r, T, "none")]
    assert len(flat_states) == len(states)
    assert all(
        (a == b) == (b.count(T.stoi[GOAL]) == 1) for a, b in zip(flat_states, states, strict=True)
    )

    SV.reset_state_format()  # a new process: train sets the format from the pickle
    seen: list[Any] = []

    def spy(*a: Any, **k: Any) -> Any:
        out = heldout_records(*a, **k)
        seen.append(out)
        return out

    monkeypatch.setattr(train, "heldout_records", spy)
    out = tmp / "stage1"
    args = ["--data", str(pairs), "--heldout", str(held), "--out", str(out), "--best_steps", "3"]
    args += ["--best_dims", "2,32,4,64", "--tok_budget", "2048", "--device", "cpu"]
    assert train.main(args) == 0
    (held_recs,) = seen
    assert max(p.count(T.stoi[GOAL]) for r in held_recs for p, _, _ in r) >= 2
    assert SV.current() == "goalstack"

    SV.reset_state_format()  # a new process: load_ckpt sets it from the checkpoint
    model, tok, extra = C.load_ckpt(out / "model.pt")
    assert extra["state_format"] == "goalstack" == model.state_format == SV.current()
    assert SE.Env.state_tokens is not SV._FLAT
    flat_ck = tmp / "flat.pt"
    with pytest.raises(SV.StateFormatError):
        C.save_ckpt(flat_ck, tiny_model(T.vocab_size), T.mode, extra={"state_format": "flat"})
    C.save_ckpt(tmp / "again.pt", model, tok.mode, extra={"x": 1})  # inherits goalstack
    _, _, e2 = C.load_ckpt(tmp / "again.pt")
    assert e2["state_format"] == "goalstack"

    SV.reset_state_format()
    C.save_ckpt(flat_ck, tiny_model(T.vocab_size), T.mode, extra={"x": 1})
    _, _, e3 = C.load_ckpt(flat_ck)
    assert "state_format" not in e3 and SV.current() == "flat"  # flat checkpoints unchanged
    with pytest.raises(SV.StateFormatError):  # one format per process
        C.load_ckpt(out / "model.pt")
    m, _, _ = C.load_ckpt(out / "model.pt", set_format=False)
    with pytest.raises(SV.StateFormatError):  # nor sampled in another
        think_generate(m, T, [json.loads(plain.read_text().splitlines()[0])["prompt"]])


# ----------------------------------------------------------------------------------- the sampler
@needs_k12
def test_sampler_prompts_use_the_format(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model scripted on goalstack pairs finishes nested K12 proofs only if the sampler shows it
    goalstack states; scripted on flat pairs it stops at the first nested state."""
    monkeypatch.setattr(D.state_sample, "NAME_BASE_MAX", 0)
    recs = [r for r in k12_head(60) if max(len(e.frames) for e, _ in _replay(r)) == 3][:4]
    assert len(recs) == 4
    with SV.state_format("flat"):
        flat = {r["prompt"]: plain_pairs(r, T, "empty") for r in recs}
    SV.set_state_format("goalstack")
    for r in recs:
        gs = plain_pairs(r, T, "empty")
        prompts = [r["prompt"]] * 2
        res = think_generate(ScriptedModel(T, script_from_pairs(gs)), T, prompts, budgets=16)
        for x in res:
            assert x["status"] == "done", x
            assert [bytes(s["state"]) for s in x["trace"]] == [p for p, _, _ in gs]
            assert [bytes(s["prompt"]) for s in x["trace"]] == [
                p + bytes([T.think]) for p, _, _ in gs
            ]
        first_nested = next(i for i, (p, _, _) in enumerate(gs) if p.count(T.stoi[GOAL]) > 1)
        (x,) = think_generate(
            ScriptedModel(T, script_from_pairs(flat[r["prompt"]])), T, prompts[:1], budgets=16
        )
        assert x["status"] != "done" and x["fail_step"] == first_nested
