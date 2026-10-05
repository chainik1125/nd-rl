"""Regression tests for trace_dsl (CPU; needs Dan's fork at ND_TAKEHOME).

code/.venv/bin/python -m pytest code/experiments/current/thinking_traces_20261005/test_trace_dsl.py
"""

import copy
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
trace_dsl = pytest.importorskip("trace_dsl")

PILOT = os.path.join(HERE, "pilot_traces.jsonl")
C = "( ( P > ( ( ~ P ) & R ) ) & ( ~ S ) )"


def pilot():
    with open(PILOT) as f:
        return [json.loads(line) for line in f if line.strip()]


def first():
    return next(r for r in pilot() if r["id"] == "k12_37120")


def test_vocabulary_is_lean_staten_plus_three():
    assert trace_dsl.TOK.vocab_size == 111
    assert set(trace_dsl.NEW_TOKENS) == {"<think>", "</think>", "<no>"}
    assert not set(trace_dsl.NEW_TOKENS) & set(trace_dsl.TOK.itos)
    for alias in trace_dsl.ALIASES.values():
        assert alias in trace_dsl.TOK.stoi


@pytest.mark.parametrize("rec", pilot(), ids=lambda r: r["id"])
def test_pilot_trace_valid_strict(rec):
    out = trace_dsl.validate(rec, lean=False, strict=True)
    assert out["ok"], out["error"]


@pytest.mark.skipif(not os.path.exists(os.path.expanduser("~/.elan/bin/lean")), reason="no Lean")
def test_pilot_lean_accepts_first_trace():
    out = trace_dsl.validate(first(), lean=True)
    assert out["ok"] and out["lean"], out["error"]


BAD = {
    "no_on_working_candidate": f"have n5 : {C} := <| n2 , n1 |> ;\n<no>",
    "failing_candidate_unrejected": f"have n5 : {C} := <| n1 , n2 |> ;",
    "false_look": "n1 : S",
    "ill_typed_reduction": f"|- {C} := <| n1 , n2 |>",
    "no_on_provable_goal": "|- ( ~ S ) <no>",
    "out_of_vocabulary": "|- ( ~ S ) because",
    "reject_without_candidate": "<no>",
    "non_canonical_candidate_name": f"have n9 : {C} := <| n2 , n1 |> ;",
    "action_not_last_survivor": f"have n5 : {C} := <| n2 , n1 |> ;\nhave n5 : {C} := n2 ;\n<no>\n"
    f"have n5 : ( ~ S ) := n1 ;",
}


@pytest.mark.parametrize("name", sorted(BAD))
def test_false_claims_rejected(name):
    rec = copy.deepcopy(first())
    rec["steps"][3]["think"] = BAD[name]
    out = trace_dsl.validate(rec, lean=False)
    assert not out["ok"], name


def test_unflagged_dead_goal_is_soft_then_strict_error():
    rec = copy.deepcopy(next(r for r in pilot() if r["id"] == "tb_58b532c9"))
    rec["steps"][1]["think"] = rec["steps"][1]["think"].replace("n4 : P |- Q <no>", "n4 : P |- Q")
    assert trace_dsl.validate(rec, lean=False)["ok"]
    assert not trace_dsl.validate(rec, lean=False, strict=True)["ok"]


def test_prompt_example_step():
    """The worked example in production_prompt.md checks against the real environment."""
    prompt = "THM ( P > Q ) , ( R > Q ) , ( P v R ) SEQ Q PRF"
    env = trace_dsl.Env(prompt, canon=True, base=0, assign=False)
    for a in (
        "have n1 : ( P > Q ) := h1 ;",
        "have n2 : ( R > Q ) := h2 ;",
        "have n3 : ( P v R ) := h3 ;",
    ):
        assert env.apply(trace_dsl.tokens(a))[0]
    think = trace_dsl.tokens(
        "<think>\n|- Q := n1 |- P\n|- P <no>\n"
        "|- Q := Or.elim n3 ( fun ( n5 : P ) => |- Q ) ( fun ( n6 : R ) => |- Q )\n"
        "n5 : P |- Q := n1 n5\nn6 : R |- Q := n2 n6\n</think>"
    )
    action = trace_dsl.tokens("have n4 : Q := Or.elim n3 ( fun ( n5 : P ) => by")
    assert trace_dsl.check_step(env, think, action, {}) == []


def test_validator_does_not_write_into_the_fork():
    trace_dsl.validate(first(), lean=False)
    assert sys.dont_write_bytecode
    for sub in ("", "nd_verify"):
        assert not os.path.exists(os.path.join(trace_dsl.TAKEHOME, sub, "__pycache__"))


def _rec(rid):
    return copy.deepcopy(next(r for r in pilot() if r["id"] == rid))


def test_dead_box_candidate_needs_its_no():
    """Peirce, step 1: the box ( P > Q ) passes the environment but leaves Q unprovable."""
    rec = _rec("tb_58b532c9")
    think = rec["steps"][1]["think"]
    assert "n4 : P |- Q <no>\n<no>\n" in think
    rec["steps"][1]["think"] = think.replace("n4 : P |- Q <no>\n<no>\n", "n4 : P |- Q <no>\n", 1)
    out = trace_dsl.validate(rec, lean=False)
    assert not out["ok"] and "dead" in out["error"]


def test_no_goal_is_false_when_premises_contradict():
    """K12 53241 has inconsistent premises: every goal is provable, so `<no>` on a goal is false."""
    rec = _rec("k12_53241")
    rec["steps"][2]["think"] = "|- S <no>\n" + rec["steps"][2]["think"]
    assert not trace_dsl.validate(rec, lean=False)["ok"]


def test_bare_have_is_rejected_not_a_crash():
    rec = _rec("k12_37120")
    rec["steps"][3]["think"] = "have\n" + rec["steps"][3]["think"]
    assert not trace_dsl.validate(rec, lean=False)["ok"]
    rec = _rec("k12_37120")
    rec["steps"][0]["action"] = "have"
    assert not trace_dsl.validate(rec, lean=False)["ok"]


def test_vacuous_reduction_is_soft():
    rec = _rec("k12_37120")
    rec["steps"][3]["think"] = f"|- {C} := |- {C}\n" + rec["steps"][3]["think"]
    out = trace_dsl.validate(rec, lean=False)
    assert out["ok"] and any("vacuous" in x for x in out["soft"])
    assert not trace_dsl.validate(rec, lean=False, strict=True)["ok"]


def _mutants(line, rng):
    """Corruptions of one think line: toggle <no>, swap an atom, a name or a connective."""
    t = line.split()
    out = []
    if t and t[-1] == "<no>":
        out.append(" ".join(t[:-1]) if len(t) > 1 else None)
    elif "|-" in t:
        out.append(line + " <no>")
    for pool in (("P", "Q", "R", "S"), ("&", "v", ">")):
        idx = [i for i, x in enumerate(t) if x in pool]
        if idx:
            i = rng.choice(idx)
            u = list(t)
            u[i] = rng.choice([x for x in pool if x != t[i]])
            out.append(" ".join(u))
    names = [i for i, x in enumerate(t) if trace_dsl.is_name(x)]
    if names:
        i = rng.choice(names)
        u = list(t)
        u[i] = rng.choice([f"n{k}" for k in range(1, 9) if f"n{k}" != t[i]])
        out.append(" ".join(u))
    return out


def test_mutation_sensitivity_strict():
    """Pilot (2026-10-05): 305 / 337 mutants rejected; the 32 survivors were inspected and are true
    claims (fresh-binder renames, <no> on reductions with a dead hole, one failing candidate
    turned into another).  Guard against the validator getting weaker."""
    import random

    rng = random.Random(0)
    total = rejected = 0
    for rec in pilot():
        for si, st in enumerate(rec["steps"]):
            lines = st["think"].split("\n") if st["think"] else []
            for li, line in enumerate(lines):
                for new in _mutants(line, rng):
                    m = copy.deepcopy(rec)
                    m["steps"][si]["think"] = "\n".join(
                        lines[:li] + ([new] if new is not None else []) + lines[li + 1 :]
                    )
                    total += 1
                    rejected += not trace_dsl.validate(m, lean=False, strict=True)["ok"]
    assert total >= 300
    assert rejected / total >= 0.88, (rejected, total)
