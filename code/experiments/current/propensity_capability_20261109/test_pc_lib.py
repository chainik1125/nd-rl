"""Tests for the propensity/capability read (CPU only, no GPU).

    python test_pc_lib.py              # all tests; the model tests load stage1_full_s3.pt on the CPU (~2-4 min)
    PC_FAST=1 python test_pc_lib.py    # skip the model tests
    python -m pytest test_pc_lib.py    # the same under pytest

Fast tests: applicability on hand-made sequents over P Q R S (parser, not strings); every direction's opening is
the canonical action Dan's `decompose` gives and is legal in Dan's Env at every name base, on hand-made and on all 346
frozen theorems; the direction classifiers agree with the verifier (and with Dan's `patterns` predicates) on
constructed proofs for every direction, including attempts that open a box and abandon the direction; the result
rows meet SPEC.md's schema.

Model tests (CPU, Dan's full-corpus base): with no forced prefix the runner's sampling is `env_generate` byte for
byte; every sampled forced attempt starts with its direction's opening, in the state the env builds; the exact
first-move probabilities (the E arm) match sampled first-move frequencies within Monte Carlo error.
"""

# ruff: noqa: E501, SIM115, B905, B007, SIM300, I001
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

from pc_lib import (  # first: it puts Dan's code on the path
    BASES,
    DIRS,
    FORCED,
    KD,
    Attempt,
    applicable,
    attempt_actions,
    classify,
    first_move_rows,
    forced_prefix,
    opening,
    premise_actions,
    shift_action,
    theorem,
)

from nd_verify.verify import verify  # type: ignore[import-not-found]  # noqa: E402
from patterns import derived_ore, parse, prune_lines, reductio  # type: ignore[import-not-found]  # noqa: E402
from state_env import Env, decompose  # type: ignore[import-not-found]  # noqa: E402

SETS = Path(os.environ.get("PC_SETS", "/scratch/cowifornia/nd-rl/propensity_capability/sets"))
CKPT = KD / "bases/stage1_full_s3.pt"
FAST = os.environ.get("PC_FAST") == "1"


def t(s: str) -> list[str]:
    return s.split()


def acts_of(prompt: str, nd: str) -> list[list[str]]:
    steps, _, _ = decompose(prompt, nd, canon=True)
    return [list(a) for _, a, _ in steps]


# ----------------------------------------------------------------------------- constructed proofs
# name: (prompt, ND proof, verifier verdict, first, opens, through)
PROOFS = {
    "reductio": ("THM ( ( ~ P ) > Q ) , ( ~ Q ) SEQ P PRF",
                 "N1 ( ( ~ P ) > Q ) : PR ; N2 ( ~ Q ) : PR ; N3 | ( ~ P ) : AS ; N4 | Q : IMPE N1 N3 ; N5 | F : NEGE N4 N2 ; "
                 "N6 ( ~ ( ~ P ) ) : NEGI N3 N5 ; N7 P : DN N6 ; QED", True, "reductio", {"reductio"}, {"reductio"}),
    "negation": ("THM ( P > Q ) , ( ~ Q ) SEQ ( ~ P ) PRF",
                 "N1 ( P > Q ) : PR ; N2 ( ~ Q ) : PR ; N3 | P : AS ; N4 | Q : IMPE N1 N3 ; N5 | F : NEGE N4 N2 ; "
                 "N6 ( ~ P ) : NEGI N3 N5 ; QED", True, "negation", {"negation"}, {"negation"}),
    "conditional": ("THM ( P > Q ) , ( Q > R ) SEQ ( P > R ) PRF",
                    "N1 ( P > Q ) : PR ; N2 ( Q > R ) : PR ; N3 | P : AS ; N4 | Q : IMPE N1 N3 ; N5 | R : IMPE N2 N4 ; "
                    "N6 ( P > R ) : IMPI N3 N5 ; QED", True, "conditional", {"conditional"}, {"conditional"}),
    "cases": ("THM ( P v Q ) , ( P > R ) , ( Q > R ) SEQ R PRF",
              "N1 ( P v Q ) : PR ; N2 ( P > R ) : PR ; N3 ( Q > R ) : PR ; N4 | P : AS ; N5 | R : IMPE N2 N4 ; N6 | Q : AS ; "
              "N7 | R : IMPE N3 N6 ; N8 R : ORE N1 N4 N5 N6 N7 ; QED", True, "cases", {"cases"}, {"cases"}),
    "direct": ("THM ( P & Q ) SEQ ( Q & P ) PRF",
               "N1 ( P & Q ) : PR ; N2 Q : ANDE2 N1 ; N3 P : ANDE1 N1 ; N4 ( Q & P ) : ANDI N2 N3 ; QED",
               True, "direct", {"direct"}, {"direct"}),
    # a reductio on the goal in focus inside a conditional box: both directions, the conditional first
    "reductio_in_conditional": ("THM ( ( ~ P ) > Q ) , ( ~ Q ) SEQ ( R > P ) PRF",
                                "N1 ( ( ~ P ) > Q ) : PR ; N2 ( ~ Q ) : PR ; N3 | R : AS ; N4 | | ( ~ P ) : AS ; N5 | | Q : IMPE N1 N4 ; "
                                "N6 | | F : NEGE N5 N2 ; N7 | ( ~ ( ~ P ) ) : NEGI N4 N6 ; N8 | P : DN N7 ; N9 ( R > P ) : IMPI N3 N8 ; QED",
                                True, "conditional", {"conditional", "reductio"}, {"conditional", "reductio"}),
    # a reductio on a lemma (P, the goal is P v R): reductio anywhere, but the first action is not the goal's opening
    "reductio_lemma": ("THM ( ( ~ P ) > Q ) , ( ~ Q ) SEQ ( P v R ) PRF",
                       "N1 ( ( ~ P ) > Q ) : PR ; N2 ( ~ Q ) : PR ; N3 | ( ~ P ) : AS ; N4 | Q : IMPE N1 N3 ; N5 | F : NEGE N4 N2 ; "
                       "N6 ( ~ ( ~ P ) ) : NEGI N3 N5 ; N7 P : DN N6 ; N8 ( P v R ) : ORI1 N7 ; QED",
                       True, "other", {"reductio"}, {"reductio"}),
    # a negated goal attacked by reductio (assume ¬¬¬P's binder ¬¬P): reductio, not negation; the inner ¬P box is a lemma
    "reductio_on_negated_goal": ("THM ( P > Q ) , ( ~ Q ) SEQ ( ~ P ) PRF",
                                 "N1 ( P > Q ) : PR ; N2 ( ~ Q ) : PR ; N3 | ( ~ ( ~ P ) ) : AS ; N4 | | P : AS ; N5 | | Q : IMPE N1 N4 ; "
                                 "N6 | | F : NEGE N5 N2 ; N7 | ( ~ P ) : NEGI N4 N6 ; N8 | F : NEGE N7 N3 ; "
                                 "N9 ( ~ ( ~ ( ~ P ) ) ) : NEGI N3 N8 ; N10 ( ~ P ) : DN N9 ; QED",
                                 True, "reductio", {"reductio"}, {"reductio"}),
    # goal ¬¬P by negation introduction: the same box shape as a reductio on P, but it is the goal's own introduction
    "negation_of_double_negation": ("THM P SEQ ( ~ ( ~ P ) ) PRF",
                                    "N1 P : PR ; N2 | ( ~ P ) : AS ; N3 | F : NEGE N1 N2 ; N4 ( ~ ( ~ P ) ) : NEGI N2 N3 ; QED",
                                    True, "negation", {"negation"}, {"negation"}),
    # opens a reductio box, closes it, then proves the goal directly: takes the direction, does not go through it
    "abandon_after_negi": ("THM ( P & Q ) SEQ P PRF",
                           "N1 ( P & Q ) : PR ; N2 | ( ~ P ) : AS ; N3 | P : ANDE1 N1 ; N4 | F : NEGE N3 N2 ; "
                           "N5 ( ~ ( ~ P ) ) : NEGI N2 N4 ; N6 P : ANDE1 N1 ; QED", True, "reductio", {"reductio"}, set()),
    # ... even with the DN written: it is not on the conclusion's dependency path
    "abandon_after_dn": ("THM ( P & Q ) SEQ P PRF",
                         "N1 ( P & Q ) : PR ; N2 | ( ~ P ) : AS ; N3 | P : ANDE1 N1 ; N4 | F : NEGE N3 N2 ; "
                         "N5 ( ~ ( ~ P ) ) : NEGI N2 N4 ; N6 P : DN N5 ; N7 P : ANDE1 N1 ; QED", True, "reductio", {"reductio"}, set()),
    # a case split on a derived disjunction is no direction
    "derived_ore": ("THM ( ( P v Q ) & R ) , ( P > S ) , ( Q > S ) SEQ S PRF",
                    "N1 ( ( P v Q ) & R ) : PR ; N2 ( P > S ) : PR ; N3 ( Q > S ) : PR ; N4 ( P v Q ) : ANDE1 N1 ; N5 | P : AS ; "
                    "N6 | S : IMPE N2 N5 ; N7 | Q : AS ; N8 | S : IMPE N3 N7 ; N9 S : ORE N4 N5 N6 N7 N8 ; QED",
                    True, "direct", {"direct"}, {"direct"}),
    # a premise case split toward a lemma: cases anywhere, the first action is not the goal's opening
    "cases_lemma": ("THM ( P v Q ) , ( P > R ) , ( Q > R ) SEQ ( R & ( P v Q ) ) PRF",
                    "N1 ( P v Q ) : PR ; N2 ( P > R ) : PR ; N3 ( Q > R ) : PR ; N4 | P : AS ; N5 | R : IMPE N2 N4 ; N6 | Q : AS ; "
                    "N7 | R : IMPE N3 N6 ; N8 R : ORE N1 N4 N5 N6 N7 ; N9 ( R & ( P v Q ) ) : ANDI N8 N1 ; QED",
                    True, "other", {"cases"}, {"cases"}),
    # an implication box for a lemma is plumbing, not the conditional direction
    "implication_lemma": ("THM ( P > Q ) SEQ ( ( P > Q ) & ( P > Q ) ) PRF",
                          "N1 ( P > Q ) : PR ; N2 | P : AS ; N3 | Q : IMPE N1 N2 ; N4 ( P > Q ) : IMPI N2 N3 ; "
                          "N5 ( ( P > Q ) & ( P > Q ) ) : ANDI N4 N4 ; QED", True, "other", {"direct"}, {"direct"}),
    # DN of a premise is not a reductio
    "dn_of_premise": ("THM ( ~ ( ~ P ) ) SEQ P PRF", "N1 ( ~ ( ~ P ) ) : PR ; N2 P : DN N1 ; QED",
                      True, "direct", {"direct"}, {"direct"}),
    # broken inside the move: the verifier rejects; the opening is still taken, the direction is not gone through
    "reductio_bad_line": ("THM ( ( ~ P ) > Q ) , ( ~ Q ) SEQ P PRF",
                          "N1 ( ( ~ P ) > Q ) : PR ; N2 ( ~ Q ) : PR ; N3 | ( ~ P ) : AS ; N4 | R : IMPE N1 N3 ; N5 | F : NEGE N4 N2 ; "
                          "N6 ( ~ ( ~ P ) ) : NEGI N3 N5 ; N7 P : DN N6 ; QED", False, "reductio", {"reductio"}, set()),
    "negation_bad_line": ("THM ( P > Q ) , ( ~ Q ) SEQ ( ~ P ) PRF",
                          "N1 ( P > Q ) : PR ; N2 ( ~ Q ) : PR ; N3 | P : AS ; N4 | R : IMPE N1 N3 ; N5 | F : NEGE N4 N2 ; "
                          "N6 ( ~ P ) : NEGI N3 N5 ; QED", False, "negation", {"negation"}, set()),
    "conditional_bad_line": ("THM ( P > Q ) , ( Q > R ) SEQ ( P > R ) PRF",
                             "N1 ( P > Q ) : PR ; N2 ( Q > R ) : PR ; N3 | P : AS ; N4 | S : IMPE N1 N3 ; N5 | R : IMPE N2 N4 ; "
                             "N6 ( P > R ) : IMPI N3 N5 ; QED", False, "conditional", {"conditional"}, set()),
    "cases_bad_line": ("THM ( P v Q ) , ( P > R ) , ( Q > R ) SEQ R PRF",
                       "N1 ( P v Q ) : PR ; N2 ( P > R ) : PR ; N3 ( Q > R ) : PR ; N4 | P : AS ; N5 | R : IMPE N3 N4 ; N6 | Q : AS ; "
                       "N7 | R : IMPE N3 N6 ; N8 R : ORE N1 N4 N5 N6 N7 ; QED", False, "cases", {"cases"}, set()),
}  # fmt: skip


def test_applicable() -> None:
    """From the parsed formulas: nesting that string tests get wrong."""
    cases = [
        ("THM P SEQ Q PRF", dict(negation=False, conditional=False, cases=False)),
        (
            "THM ( P > Q ) SEQ ( ~ ( Q & P ) ) PRF",
            dict(negation=True, conditional=False, cases=False),
        ),
        (
            "THM ( P v Q ) SEQ ( ( ~ P ) > Q ) PRF",
            dict(negation=False, conditional=True, cases=True),
        ),  # ¬ inside →
        (
            "THM ( ~ ( P v Q ) ) SEQ ( ~ P ) PRF",
            dict(negation=True, conditional=False, cases=False),
        ),  # ∨ under ¬
        (
            "THM ( ( P v Q ) & R ) SEQ ( P v Q ) PRF",
            dict(negation=False, conditional=False, cases=False),
        ),  # ∨ under ∧
        (
            "THM ( ( P v Q ) > R ) , ( S v P ) SEQ ( ~ ( ~ R ) ) PRF",
            dict(negation=True, conditional=False, cases=True),
        ),
        (
            "THM SEQ ( ( P > Q ) v ( ~ ( P > Q ) ) ) PRF",
            dict(negation=False, conditional=False, cases=False),
        ),
        ("THM ( ( P > Q ) > P ) SEQ P PRF", dict(negation=False, conditional=False, cases=False)),
        (
            "THM P , ( Q v ( R > S ) ) SEQ ( ( P v Q ) > ( ~ S ) ) PRF",
            dict(negation=False, conditional=True, cases=True),
        ),
    ]
    for prompt, want in cases:
        got = applicable(prompt)
        assert got["reductio"] and got["direct"], prompt
        assert {k: got[k] for k in want} == want, (prompt, got)
        for d in FORCED:
            assert (opening(prompt, d) is not None) == got[d], (prompt, d)


def test_openings_are_dans_canonical_actions() -> None:
    """Each direction's opening is exactly the action Dan's `decompose` (canonical names) gives for a proof that
    starts with that move."""
    for name, d in (("reductio", "reductio"), ("negation", "negation"), ("conditional", "conditional"), ("cases", "cases"),
                    ("reductio_on_negated_goal", "reductio"), ("negation_of_double_negation", "negation")):  # fmt: skip
        prompt, nd = PROOFS[name][:2]
        acts = acts_of(prompt, nd)
        m = len(theorem(prompt)[0])
        assert acts[:m] == premise_actions(prompt), name
        assert acts[m] == opening(prompt, d), (name, acts[m], opening(prompt, d))
        assert forced_prefix(prompt, d) == acts[: m + 1]


def _legal_everywhere(prompt: str, d: str) -> None:
    for base in (0, 1, 11, 32):
        e = Env(prompt, canon=True, base=base, assign=True)
        for a in forced_prefix(prompt, d):
            sa = shift_action(a, base)
            ok, why = e.apply(sa)
            assert ok, (prompt, d, base, why)
            assert list(e._eff) == sa, (
                prompt,
                d,
                base,
                e._eff,
                sa,
            )  # the env keeps the action's names
        assert len(e.frames) == 2 and not e.done, (prompt, d)
        c = classify(prompt, base, [shift_action(a, base) for a in forced_prefix(prompt, d)])
        assert c["first"] == d and d in c["opens"] and not c["through"], (prompt, d, c)


def test_openings_legal_handmade() -> None:
    for prompt, *_ in PROOFS.values():
        for d in FORCED:
            if applicable(prompt)[d]:
                _legal_everywhere(prompt, d)


def test_openings_legal_on_frozen_theorems() -> None:
    """Every applicable direction's forced prefix is legal in Dan's Env, at four name bases, on all 346 frozen
    theorems, and the frozen applicability is the parser's."""
    fn = SETS / "theorems.jsonl"
    if not fn.exists():
        return
    thms = [json.loads(x) for x in open(fn)]
    assert len(thms) == 346
    n = 0
    for th in thms:
        assert th["applicable"] == applicable(th["prompt"]), th["name"]
        for d in FORCED:
            if th["applicable"][d]:
                _legal_everywhere(th["prompt"], d)
                n += 1
    assert n == 587, n  # 106 + 33 on the strict set, 240 + 44 + 84 + 80 on the others


def test_classifiers_agree_with_verifier() -> None:
    """Constructed proofs for every direction: the verifier's verdict, then the classifier's first / opens / through
    at name bases 0 and 9; `through` with the verifier equals `through` under Lean acceptance when the proof is valid,
    and is empty for the broken direction when it is not; Dan's pruned-proof predicates agree where they apply."""
    for name, (prompt, nd, valid, first, opens, through) in PROOFS.items():
        prem, goal = theorem(prompt)
        ok, why = verify(prem, goal, nd.split())
        assert ok == valid, (name, why)
        acts = acts_of(prompt, nd)
        for base in (0, 9):
            sa = [shift_action(a, base) for a in acts]
            c = classify(prompt, base, sa)  # the verifier decides validity
            assert c["done"] and c["first"] == first, (name, base, c)
            assert set(c["opens"]) == opens and set(c["through"]) == through, (name, base, c)
            if valid:
                assert classify(prompt, base, sa, accepted=True) == c, name
            else:
                assert classify(prompt, base, sa, accepted=False) == c, name
        if valid:  # Dan's predicates are structural: they apply to valid proofs
            lines = prune_lines(parse(nd))
            assert reductio(lines) == ("reductio" in through), name
            has_premise_ore = any(x["rule"] == "ORE" for x in lines) and not derived_ore(lines)
            assert has_premise_ore == ("cases" in through), name


def test_abandoned_box() -> None:
    """Opened and never finished (the attempt ends inside the box, or the env refuses the next action): the
    direction is taken, not gone through, and the attempt is not done."""
    for name in ("reductio", "negation", "conditional", "cases"):
        prompt, nd, *_ = PROOFS[name]
        d = name
        m = len(theorem(prompt)[0])
        acts = acts_of(prompt, nd)
        for cut in (m + 1, m + 2):
            c = classify(prompt, 0, acts[:cut])
            assert not c["done"] and c["first"] == d and c["opens"] == [d] and c["through"] == [], (
                name,
                cut,
                c,
            )
        # a refused action right after the opening (exact of a premise: not the box's last statement)
        c = classify(prompt, 0, acts[: m + 1] + [t("exact n1")])
        assert c["fail"] and c["opens"] == [d] and c["n_actions"] == m + 1, (name, c)


def test_invalid_premise_line() -> None:
    """A restated premise with the wrong formula (the env does not check it): the first action is `invalid`, and
    no direction counts as taken."""
    prompt = PROOFS["reductio"][0]
    bad = [t("have n1 : Q := h1 ;"), t("have n2 : ( ¬ Q ) := h2 ;")] + [opening(prompt, "reductio")]
    c = classify(prompt, 0, bad)
    assert c["first"] == "invalid" and c["opens"] == ["direct"], c
    # an attempt that restates no premise and opens the box at once
    c = classify(prompt, 0, [t("have n1 : ( ¬ ( ¬ P ) ) := ( fun ( n2 : ( ¬ P ) ) => by")])
    assert c["n_pr"] == 0 and c["first"] == "reductio" and c["opens"] == ["reductio"], c


def test_forced_prefix_is_first_action() -> None:
    """A forced attempt, replayed as the runner replays it (prefix shifted to the base, then the model's ended actions):
    its first proof action is the forced opening whatever the model wrote after it."""
    for name in ("reductio", "negation", "conditional", "cases"):
        prompt, nd, *_ = PROOFS[name]
        acts = acts_of(prompt, nd)
        m = len(theorem(prompt)[0])
        for base in (0, 5, 32):
            cont = [shift_action(a, base) for a in acts[m + 1 :]]
            att = Attempt(0, base, cont + [t("have")], [True] * len(cont) + [False], "truncated")
            full = attempt_actions(forced_prefix(prompt, name), att)
            assert len(full) == len(acts)  # the truncated last action is not applied
            c = classify(prompt, base, full)
            assert c["first"] == name and name in c["through"], (name, base, c)


def test_first_move_rows_states() -> None:
    """The scorer's rows: each premise action from its own state, each opening from the state after every premise."""

    class Stoi(dict):
        def __missing__(self, x):
            self[x] = len(self)
            return self[x]

    class Tok:
        eos, with_history, stoi = -1, False, Stoi()

        def encode_toks(self, toks):
            return [self.stoi[x] for x in toks]

    tok = Tok()
    prompt = PROOFS["cases"][0]
    ds = [d for d in FORCED if applicable(prompt)[d]]
    rows, npos, labels = first_move_rows(tok, prompt, 7, ds)
    assert labels == ["pr1", "pr2", "pr3", "reductio", "cases"]
    e = Env(prompt, canon=True, base=7, assign=True)
    for a in premise_actions(prompt):
        e.apply(shift_action(a, 7))
    st = tok.encode_toks(e.state_tokens())
    assert rows[3][0] == st and rows[4][0] == st
    assert (
        rows[4][1][:-1] == tok.encode_toks(shift_action(opening(prompt, "cases"), 7))
        and rows[4][1][-1] == -1
    )
    op = shift_action(opening(prompt, "cases"), 7)
    assert [op[i] for i in npos[4]] == ["n11", "n12"] and npos[0] == [
        1
    ]  # the have's name and the binder (base 7 + 3 + 1, + 2)


def test_result_rows_schema() -> None:
    """Synthetic raw attempts through the runner's row builder: every row meets SPEC.md's schema and the counts."""
    import pc_run

    prompt, nd = PROOFS["cases"][:2]
    acts = acts_of(prompt, nd)
    th = {"name": "x", "set": "transfer_other", "n_lines": 8, "schema": None, "source": "gen", "n_prem": 3,
          "prompt": prompt, "applicable": applicable(prompt), "dir_class": {d: None for d in DIRS} | {"reductio": "idle"}}  # fmt: skip
    ck = {"ckpt": "full_s3", "dose": "full", "seed": 3, "stage": "base", "x_dan": 0.0}
    U = [{"name": "x", "d": None, "draw": 0, "base": 0, "raw": [" ".join(a) for a in acts], "ended": [True] * len(acts),
          "end": "done", "nd": nd, "ok": True, "tokens": 50},
         {"name": "x", "d": None, "draw": 1, "base": 4, "raw": [" ".join(a) for a in [shift_action(x, 4) for x in acts[:4]]] + ["exact n1"],
          "ended": [True] * 5, "end": "syntax:exact not last", "nd": "LEANPARSE env", "ok": False, "tokens": 30}]  # fmt: skip
    m = 3
    F = []
    for d in ("reductio", "cases"):
        for i in range(2):
            raw = [] if d == "reductio" else [" ".join(a) for a in acts[m + 1 :]]
            F.append({"name": "x", "d": d, "draw": i, "base": 0, "raw": raw, "ended": [True] * len(raw),
                      "end": "step_cap" if d == "reductio" else "done", "nd": nd, "ok": d == "cases", "tokens": 7})  # fmt: skip
    E = {
        "logp": {
            "x": {"name_free": {"reductio": [-3.0] * len(BASES), "cases": [-0.1] * len(BASES)}}
        }
    }
    rows, qc = pc_run.result_rows(ck, [th], U, F, E, 2, 2)
    for r in rows:
        pc_run.check_row(r)
    by = {(r["arm"], r["direction"]): r for r in rows}
    assert len(rows) == 1 + 5 + 2 + 2
    assert (
        by[("U", "any")]["solved"] == 1
        and by[("U", "any")]["refused"] == 1
        and by[("U", "any")]["tokens"] == 80
    )
    assert (
        by[("U", "cases")]["used"] == 2
        and by[("U", "cases")]["first"] == 2
        and by[("U", "cases")]["solved"] == 1
    )
    assert by[("U", "cases")]["solved_first"] == 1 and by[("U", "reductio")]["used"] == 0
    assert by[("F", "cases")]["solved"] == 2 and by[("F", "reductio")]["capped"] == 2
    assert (
        abs(by[("E", "cases")]["logp_open"] + 0.1) < 1e-9
        and by[("E", "reductio")]["dir_class"] == "idle"
    )
    assert (
        qc["F_first_not_forced"] == 0
        and qc["replay_done_vs_env"] == 0
        and qc["F_opens_not_forced"] == 0
    )
    assert by[("U", "negation")]["applicable"] is False


# ----------------------------------------------------------------------------- model tests (CPU)
_MODEL: list = []


def model():
    if not _MODEL:
        import torch
        from model import load_ckpt  # type: ignore[import-not-found]

        torch.set_num_threads(8)
        m, tok, _ = load_ckpt(str(CKPT), "cpu")
        _MODEL.append((m, tok))
    return _MODEL[0]


def frozen(n_strict: int, n_other: int, need=None) -> list[dict]:
    thms = [json.loads(x) for x in open(SETS / "theorems.jsonl")]
    s = [x for x in thms if x["set"] == "strict106"][:n_strict]
    o = [x for x in thms if x["set"] == "transfer_other" and (need is None or need(x))][:n_other]
    return s + o


def test_unforced_identity() -> None:
    """With no forced prefix, the runner's U sampling is Dan's `env_generate` byte for byte (same seed, same batch),
    with the same end counts, prompt tokens and waves (the factor read's check 1, here on the CPU)."""
    if FAST or not CKPT.exists():
        return
    import pc_run
    from state_sample import env_generate  # type: ignore[import-not-found]

    m, tok = model()
    thms = frozen(2, 3)
    for batch, seed, k in ((8, 7, 3), (16, 2500, 4)):
        items = pc_run.u_items(thms, k)
        st1: dict = {}
        st2: dict = {}
        a = env_generate(m, tok, [x["prompt"] for x in items], greedy=False, temperature=0.8, max_action=256,
                         max_steps=48, batch=batch, seed=seed, stats=st1, gate=False)  # fmt: skip
        atts, _ = pc_run.sample(m, tok, items, seed, batch, st2, lean=False)
        assert a == [x.nd for x in atts], (batch, seed)
        assert (
            dict(st1["env_end"]) == dict(st2["env_end"])
            and st1["prompt_tokens"] == st2["prompt_tokens"]
        )
        assert st1["env_waves"] == st2["env_waves"]
        print(
            f"  identity batch {batch} seed {seed}: {len(a)} attempts identical; ends {dict(st1['env_end'])}",
            flush=True,
        )


def test_forced_attempts_start_with_opening() -> None:
    """Sampled forced attempts: the first state the model sees is the env's state after the prefix at the attempt's
    base, and the classifier finds the forced opening as the first proof action of every attempt."""
    if FAST or not CKPT.exists():
        return
    import pc_run

    m, tok = model()
    from state_sample import prompt_ids  # type: ignore[import-not-found]

    thms = frozen(
        1,
        6,
        need=lambda x: (
            x["applicable"]["negation"]
            or x["applicable"]["cases"]
            or x["applicable"]["conditional"]
        ),
    )
    items = pc_run.f_items(thms, 3)
    assert {it["d"] for it in items} == set(FORCED), {it["d"] for it in items}
    atts = pc_run.run_env(m, tok, items, 0.8, 256, 48, 64, 11, keep_states=True, gate=False)
    for it, a in zip(items, atts):
        e = Env(it["prompt"], canon=True, base=a.base, assign=True)
        for x in it["prefix"]:
            assert e.apply(shift_action(x, a.base))[0]
        assert a.states[0] == prompt_ids(tok, e), it["name"]
        c = classify(it["prompt"], a.base, attempt_actions(it["prefix"], a))
        assert c["first"] == it["d"] and it["d"] in c["opens"], (it["name"], it["d"], c)
    print(
        f"  {len(items)} forced attempts over {len(thms)} theorems: every one opens with its direction",
        flush=True,
    )


def test_first_move_probabilities() -> None:
    """The E arm's exact first-move probability (name-free, averaged over the 33 name bases) against the sampled
    frequency of the same first proof action from the same state, the base drawn as the sampler draws it."""
    if FAST or not CKPT.exists():
        return
    import pc_run

    m, tok = model()
    thms = frozen(0, 30, need=lambda x: sum(x["applicable"][d] for d in FORCED) >= 2)
    E = pc_run.e_arm(m, tok, thms, 2**26)
    cand = []
    for th in thms:
        for d in FORCED:
            if th["applicable"][d]:
                p = math.exp(pc_run.logmeanexp(E["logp"][th["name"]]["name_free"][d]))
                cand.append((p, th, d))
    picks: list = []
    for target in (
        0.1,
        0.3,
        0.6,
    ):  # prefer a direction not picked yet, then the probability nearest the target
        ok = [c for c in cand if 0.03 < c[0] < 0.97 and c not in picks]
        best = min(
            ok, key=lambda c: (c[2] in {x[2] for x in picks}, abs(c[0] - target)), default=None
        )
        if best is not None:
            picks.append(best)
    assert picks, "no first move with probability between 3 % and 97 % in the first 30 theorems"
    n = int(os.environ.get("PC_NFREQ", "1500"))
    for i, (p, th, d) in enumerate(picks):
        pre = premise_actions(th["prompt"])
        items = [{"name": th["name"], "prompt": th["prompt"], "prefix": pre}] * n
        atts = pc_run.run_env(
            m, tok, items, 0.8, 256, 48, min(n, 500), 4242 + i, stop_after=1, gate=False
        )
        hits = sum(
            classify(th["prompt"], a.base, attempt_actions(pre, a))["first"] == d for a in atts
        )
        f = hits / n
        se = math.sqrt(p * (1 - p) / n)
        z = (f - p) / se
        full = math.exp(pc_run.logmeanexp(E["logp"][th["name"]]["full"][d]))
        print(f"  {th['name']} {d}: exact {100 * p:.2f} % (all tokens {100 * full:.2f} %), sampled {100 * f:.2f} % "
              f"of {n}, z {z:+.2f}", flush=True)  # fmt: skip
        assert abs(z) < 4, (th["name"], d, p, f, z)


if __name__ == "__main__":
    random.seed(0)
    n = 0
    for k, v in list(globals().items()):
        if k.startswith("test_") and callable(v):
            v()
            n += 1
            print("ok", k, flush=True)
    print(f"{n} tests passed")
