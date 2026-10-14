"""A per-step check for the proof-state environment that is sound with respect to Lean.

    check_step(env, action) -> None | reason

``env`` is the ``state_env.Env`` before the action, and ``action`` one the environment accepts.
None means Lean may accept the step; a reason code means Lean is certain to reject it, and so the
finished proof. The environment checks everything about box openings, ``exact`` and the names in
scope; it does not check that a ``have``'s term has the declared type. This module checks exactly
that, for the ``have``\\ s whose term is complete on one line: premise lines ``:= h<k>`` and the
atomic terms (a name, ``.1`` / ``.2``, ``.elim``, an application, ``⟨ a , b ⟩``, ``Or.inl`` /
``Or.inr``, ``Classical.byContradiction ( fun hh => a hh )``).

Why it is sound. In the ``lean_seq`` fragment every hypothesis has a declared type, the only free
symbols are the four ``Prop`` variables, and the only definition Lean unfolds is ``¬ A := A →
False``, so two formulas are definitionally equal if and only if they are equal once every ``¬ A``
is rewritten to ``A → False`` (``norm``). The rules, and what Lean accepts beyond the ND rules:

- ``¬ A`` and ``A → False`` are interchangeable everywhere (declared types, arguments, results);
- ``n .elim`` resolves on the declared head of ``n``'s type (generalised field notation):
  ``False.elim`` (any type), ``Not.elim`` (``A → C`` for ``n : ¬ A``), ``And.elim`` and ``Or.elim``
  (their eliminator types); on a declared ``A → B`` or an atom Lean reports an error;
- ``n .1`` / ``n .2`` only on a declared ``∧`` (otherwise "invalid projection");
- ``⟨ a , b ⟩`` only against an expected ``∧``; ``f a`` only for an ``f`` of type ``→`` or ``¬``.

Every formula the environment accepts is in ``lean_formula_tokens``' fully parenthesised form (the
token parser is a bijection), so Lean reads the declared types as the environment does.

How the soundness was established: in the take-home fork (``step_check.py`` and
``lean_prefilter.py`` at 4dcd0e4f, run ``guided-tts``), 314,541 distinct (state, step) pairs from
model samples, each step a standalone Lean 4.34 theorem whose binders are the state's hypotheses
(as ``lean_source`` builds it), gave 0 false rejects and 0 misses: Lean accepted 187,036 and
rejected 127,505, exactly the checker's verdicts (``artifacts/gt/validate_s{0,1,2}.json``). This
port returns the fork's verdict on every one of those pairs
(``tests/fixtures/proof_state/revalidate_guided_tts.py``) and on the fixture's 10,724 pairs
(``tests/test_step_check.py``), which that test also re-checks against Lean when it is installed.

Precondition: ``action`` is one the environment accepts. Outside it the check still never rejects
a step Lean accepts, but it passes every term that is not one of the shapes above.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence

from code.shared_code.lean_seq import (
    FALSE,
    BinOp,
    Bot,
    Formula,
    LeanParseError,
    Not,
    lean_formula_tokens,
    parse_lean_formula,
    text,
)
from code.shared_code.state_env import Env


@functools.lru_cache(maxsize=1 << 16)
def norm(f: Formula) -> Formula:
    """``f`` with every ``¬ A`` rewritten to ``A → False``: Lean's definitional equality here."""
    if isinstance(f, Not):
        return BinOp("imp", norm(f.sub), FALSE)
    if isinstance(f, BinOp):
        return BinOp(f.op, norm(f.left), norm(f.right))
    return f


_BY_CONTRADICTION = ["Classical.byContradiction", "(", "fun", "hh", "=>"]


def _is_name(t: str) -> bool:
    return len(t) > 1 and t[0] == "n" and t[1:].isdigit()


def _shape(term: Sequence[str]) -> tuple[str, list[str]] | None:
    """An atomic term -> (its kind, the names it cites), or None if it is not one."""
    t = list(term)
    names = [x for x in t if _is_name(x)]
    if len(t) == 1 and names == t:
        return "reiterate", names
    if len(t) == 2 and _is_name(t[0]) and t[1] in (".1", ".2", ".elim"):
        return t[1], names
    if len(t) == 2 and names == t:
        return "apply", names
    if len(t) == 5 and t[0::2] == ["⟨", ",", "⟩"] and names == [t[1], t[3]]:
        return "and-intro", names
    if len(t) == 2 and t[0] in ("Or.inl", "Or.inr") and names == t[1:]:
        return t[0], names
    if len(t) == 8 and t[:5] == _BY_CONTRADICTION and t[6:] == ["hh", ")"] and names == t[5:6]:
        return "by-contradiction", names
    return None


def _elim_ok(declared: Formula, want: Formula) -> bool:
    """Whether ``n .elim : want`` for ``n`` declared ``declared``. Field notation resolves on the
    declared head; an implication or an atom has no ``.elim``."""
    w = norm(want)
    if isinstance(declared, Bot):  # False.elim : False → c
        return True
    if isinstance(declared, Not):  # Not.elim : ¬ a → a → c
        return isinstance(w, BinOp) and w.op == "imp" and w.left == norm(declared.sub)
    if not (isinstance(declared, BinOp) and declared.op != "imp"):
        return False
    if not (isinstance(w, BinOp) and w.op == "imp"):
        return False
    a, b = norm(declared.left), norm(declared.right)
    if declared.op == "and":  # And.elim : a ∧ b → (a → b → c) → c
        return w.left == BinOp("imp", a, BinOp("imp", b, w.right))
    rest = w.right  # Or.elim : a ∨ b → (a → c) → (b → c) → c
    if not (isinstance(rest, BinOp) and rest.op == "imp"):
        return False
    return w.left == BinOp("imp", a, rest.right) and rest.left == BinOp("imp", b, rest.right)


def check_term(
    premises: Sequence[Formula],
    scope: Mapping[str, Formula],
    declared: Formula,
    term: Sequence[str],
) -> str | None:
    """The check behind ``check_step``: the term of ``have n : declared := <term> ;`` (``term``
    without the ``;``), given the theorem's premises and the hypotheses in scope by name.

    Returns None (Lean may accept) or a reason: ``unknown-premise``, ``premise-type``, ``unbound``,
    ``reiterate``, ``projection``, ``and-elim``, ``elim``, ``function-expected``, ``app-arg``,
    ``app-result``, ``anon-ctor``, ``and-intro``, ``or-intro-target``, ``or-intro``,
    ``by-contradiction``.
    """
    goal = norm(declared)
    if len(term) == 1 and term[0][:1] == "h" and term[0][1:].isdigit():  # a premise line
        k = int(term[0][1:])
        if not 1 <= k <= len(premises):
            return "unknown-premise"
        return None if norm(premises[k - 1]) == goal else "premise-type"
    shape = _shape(term)
    if shape is None:
        return None  # not modelled: Lean decides
    kind, names = shape
    if any(n not in scope for n in names):
        return "unbound"
    a = scope[names[0]]
    if kind == "reiterate":
        return None if norm(a) == goal else "reiterate"
    if kind in (".1", ".2"):
        if not isinstance(a, BinOp) or a.op != "and":
            return "projection"
        return None if norm(a.left if kind == ".1" else a.right) == goal else "and-elim"
    if kind == ".elim":
        return None if _elim_ok(a, declared) else "elim"
    if kind == "apply":
        fn = norm(a)
        if not isinstance(fn, BinOp) or fn.op != "imp":
            return "function-expected"
        if norm(scope[names[1]]) != fn.left:
            return "app-arg"
        return None if fn.right == goal else "app-result"
    if kind == "and-intro":
        if not isinstance(goal, BinOp) or goal.op != "and":
            return "anon-ctor"
        ok = norm(a) == goal.left and norm(scope[names[1]]) == goal.right
        return None if ok else "and-intro"
    if kind in ("Or.inl", "Or.inr"):
        if not isinstance(goal, BinOp) or goal.op != "or":
            return "or-intro-target"
        return None if norm(a) == (goal.left if kind == "Or.inl" else goal.right) else "or-intro"
    # by-contradiction: a : (declared → False) → False
    want = BinOp("imp", BinOp("imp", goal, FALSE), FALSE)
    return None if norm(a) == want else "by-contradiction"


def _have(action: Sequence[str]) -> tuple[Formula, list[str]] | None:
    """``have n : F := <term> ;`` -> (F, term), or None for any other action."""
    if len(action) < 4 or action[0] != "have":
        return None
    try:
        declared, i = parse_lean_formula(action, 3)
    except LeanParseError:
        return None
    if i >= len(action) or action[i] != ":=" or action[-1] != ";":
        return None
    return declared, list(action[i + 1 : -1])


def check_step(env: Env, action: Sequence[str]) -> str | None:
    """None if Lean may accept ``action`` in ``env``'s state, else the reason it must reject it.

    Box openings, ``exact`` and anything else that is not a one-line ``have`` pass: the
    environment checks them completely. See the module docstring for the precondition.
    """
    have = _have(action)
    if have is None:
        return None
    declared, term = have
    if not term or term[0] in ("(", "Or.elim"):
        return None
    try:
        return check_term(env.premises, env.scope(), declared, term)
    except RecursionError:  # a formula nested hundreds deep: not modelled
        return None


def formula_text(f: Formula) -> str:
    """A formula as Lean source, fully parenthesised."""
    return " ".join(lean_formula_tokens(f))


def lean_source(env: Env, action: Sequence[str]) -> str:
    """A one-line ``have`` action as a standalone Lean theorem whose binders are exactly the
    state's hypotheses, typed as the state shows them: Lean accepts it if and only if it accepts
    the step. For checking the checker (``lean_check.check_theorems``)."""
    have = _have(action)
    if have is None:
        raise ValueError("not a one-line have action")
    declared, term = have
    binders = [f"( h{j} : {formula_text(p)} )" for j, p in enumerate(env.premises, start=1)]
    binders += [f"( {n} : {formula_text(f)} )" for n, f in env.scope().items()]
    goal = formula_text(declared)
    head = " ".join(["theorem t ( P Q R S : Prop )", *binders])
    return f"{head} : {goal} := by have z : {goal} := {text(term)} ; exact z"
