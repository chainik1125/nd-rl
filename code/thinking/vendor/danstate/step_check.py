#!/usr/bin/env python3
"""step_check.py -- a per-step check, sound with respect to Lean, for one `lean_seq` action in the proof-state
environment (run `guided-tts`, proposal 24).

    check(env, act) -> None | reason        None: pass (Lean may accept);  reason: Lean is certain to reject the step
    canonical(act)  -> bool                 every formula in the action is written in `lean_tok.ftoks`'s form
    lean_source(env, act, tok) -> str       the step as a standalone Lean theorem (the agreement test)

`env` is the `state_env.Env` *before* the action and `act` an action the environment accepts structurally.  The
environment already checks everything about box openings (`( fun ( x : A ) => by`, `Or.elim j ( fun …`), `exact` and
names in scope; what it does not check is whether a `have`'s term has the declared type ("no symbolic type check").
This module checks exactly that, for the two kinds of `have` whose term is complete on one line:

  * premise lines  `have n : F := h<k> ;`  -- F must be h<k>'s type up to `¬A ≡ A → False`;
  * atomic terms   (reiteration, `.1` / `.2`, `.elim` by declared head incl. `Not.elim`, application, `⟨a, b⟩`,
    `Or.inl` / `Or.inr`, `Classical.byContradiction (fun hh => a hh)`).

The rules are `lean_prefilter._term`'s (run `lean-prefilter`: 0 false rejects in 1,310,119 whole texts), applied in
the environment's scope instead of a re-parsed text.  A formula not in the fully parenthesised canonical form is not
modelled (Lean's precedence parse could differ from the ND parser's): such a step passes, and the caller stops
checking the rest of the attempt (`canonical`), because later steps cite hypotheses whose literal type the
environment may have read differently from Lean.

Use: only after `step_check_validate.py` shows 0 false rejects against Lean on >= 50,000 steps (brief).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lean_tok import ftoks, ParseFail
from state_env import parse_ftoks, is_name
import lean_prefilter as LP
from nd2lean import lf


def _formula_spans(act):
    """(start, end) of every formula the action writes: the declared type, and a box binder's type."""
    out = []
    if len(act) < 4 or act[0] != 'have' or act[2] != ':':
        return out
    f, used = parse_ftoks(act, 3)
    out.append((3, 3 + used))
    i = 3 + used + 1                              # after ':='
    j = None
    if i < len(act) and act[i] == '(':
        j = i + 5
    elif i < len(act) and act[i] == 'Or.elim':
        j = i + 7
    if j is not None and j < len(act):
        _, u2 = parse_ftoks(act, j)
        out.append((j, j + u2))
    return out


def canonical(act):
    try:
        for a, b in _formula_spans(act):
            f, _ = parse_ftoks(act, a)
            if list(act[a:b]) != ftoks(f):
                return False
        return True
    except (ParseFail, IndexError):
        return False


def scope(env):
    sc = {}
    for fr in env.frames:                         # inner frames override (canonical naming never shadows anyway)
        sc.update(fr.names)
    return sc


def check(env, act):
    """None = pass, else the reason Lean must reject `act` in `env`'s state."""
    try:
        if not act or act[0] != 'have':
            return None
        f, used = parse_ftoks(act, 3)
        if list(act[3:3 + used]) != ftoks(f):
            return None
        i = 3 + used
        if act[i] != ':=':
            return None
        t = act[i + 1]
        if t in ('(', 'Or.elim'):
            return None                           # box openings: fully checked by the environment
        if act[-1] != ';':
            return None
        p = LP._P(list(act[i + 1:]))
        LP._term(p, scope(env), f, env.prem)
        if p.peek() != ';' or p.i != len(act) - i - 2:
            return None
        return None
    except LP._Reject as r:
        return str(r)
    except (LP._Pass, ParseFail, IndexError, KeyError, RecursionError):
        return None


def lean_source(env, act, tok):
    """`theorem t (P Q R S : Prop) (h1 : ..) .. (n3 : A) .. : F := by have z : F := <term> ; exact z` -- the step's
    `have` in a context with exactly the environment's hypotheses, typed as the state shows them."""
    f, used = parse_ftoks(act, 3)
    term = tok.text(list(act[3 + used + 1:-1]))
    hyps = ''.join(f' ({n} : {lf(g)})' for n, g in scope(env).items())
    prem = ''.join(f' (h{j + 1} : {lf(g)})' for j, g in enumerate(env.prem))
    return f'theorem t (P Q R S : Prop){prem}{hyps} : {lf(f)} := by have z : {lf(f)} := {term} ; exact z'
