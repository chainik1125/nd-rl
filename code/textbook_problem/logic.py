"""Formula utilities for the textbook-problem workstream: parsing, printing, structure, and two
validity checks (classical by truth table, intuitionistic by Dyckhoff's contraction-free G4ip).

Formulas are the fork verifier's tuples: ('atom', 'P'), ('bot',), ('not', A),
('and'|'or'|'imp', A, B).
"""

import itertools
import os
import sys
from functools import cache

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ws3/fork_pin"))
from nd_verify.verify import BOT, parse_formula  # noqa: E402

BIN = {"and": "&", "or": "v", "imp": ">"}
PRETTY = {"and": "∧", "or": "∨", "imp": "→"}


def parse_prompt(prompt: str):
    """'THM A , B SEQ C PRF' -> ([A, B], C)."""
    toks = prompt.split()
    assert toks[0] == "THM" and toks[-1] == "PRF", prompt
    i, prem = 1, []
    if toks[i] != "SEQ":
        while True:
            f, i = parse_formula(toks, i)
            prem.append(f)
            if toks[i] == ",":
                i += 1
                continue
            break
    assert toks[i] == "SEQ", prompt
    goal, i = parse_formula(toks, i + 1)
    assert toks[i] == "PRF", prompt
    return prem, goal


def to_tokens(f) -> str:
    if f[0] == "atom":
        return f[1]
    if f == BOT:
        return "F"
    if f[0] == "not":
        return f"( ~ {to_tokens(f[1])} )"
    return f"( {to_tokens(f[1])} {BIN[f[0]]} {to_tokens(f[2])} )"


def pretty(f, top=True) -> str:
    if f[0] == "atom":
        return f[1]
    if f == BOT:
        return "⊥"
    if f[0] == "not":
        return "¬" + pretty(f[1], False)
    s = f"{pretty(f[1], False)} {PRETTY[f[0]]} {pretty(f[2], False)}"
    return s if top else f"({s})"


def pretty_seq(prem, goal) -> str:
    return (", ".join(pretty(p) for p in prem) + " ⊢ " if prem else "⊢ ") + pretty(goal)


# ---------------------------------------------------------------- structure


def subformulas(f):
    yield f
    for c in f[1:]:
        if isinstance(c, tuple):
            yield from subformulas(c)


def atoms(f) -> set:
    return {g[1] for g in subformulas(f) if g[0] == "atom"}


def size(f) -> int:
    """Number of connectives (¬, ∧, ∨, →); atoms and ⊥ count 0."""
    return sum(1 for g in subformulas(f) if g[0] in ("not", "and", "or", "imp"))


def depth(f) -> int:
    kids = [c for c in f[1:] if isinstance(c, tuple)]
    return 0 if not kids else 1 + max(depth(c) for c in kids)


def count(f, op) -> int:
    return sum(1 for g in subformulas(f) if g[0] == op)


def max_neg_stack(f) -> int:
    best = 0
    for g in subformulas(f):
        n = 0
        while g[0] == "not":
            n, g = n + 1, g[1]
        best = max(best, n)
    return best


def is_iff(f) -> bool:
    """(A → B) ∧ (B → A): how the tokenizer spells a biconditional."""
    return (
        f[0] == "and"
        and f[1][0] == "imp"
        and f[2][0] == "imp"
        and f[1][1] == f[2][2]
        and f[1][2] == f[2][1]
    )


def count_iff(f) -> int:
    return sum(1 for g in subformulas(f) if is_iff(g))


# ---------------------------------------------------------------- semantics


def ev(f, v) -> bool:
    t = f[0]
    if t == "atom":
        return v[f[1]]
    if t == "bot":
        return False
    if t == "not":
        return not ev(f[1], v)
    a, b = ev(f[1], v), ev(f[2], v)
    return a and b if t == "and" else a or b if t == "or" else (not a) or b


def valuations(names):
    names = sorted(names)
    for bits in itertools.product([False, True], repeat=len(names)):
        yield dict(zip(names, bits, strict=True))


def classically_valid(prem, goal) -> bool:
    names = set().union(atoms(goal), *[atoms(p) for p in prem])
    return all(ev(goal, v) for v in valuations(names) if all(ev(p, v) for p in prem))


def satisfiable(fs) -> bool:
    names = set().union(set(), *[atoms(f) for f in fs])
    return any(all(ev(f, v) for f in fs) for v in valuations(names))


def needed_premises(prem, goal):
    """Indices of premises in one minimal subset that still entails the goal classically
    (greedy drop from the full set; the first premise that can go, goes)."""
    keep = list(range(len(prem)))
    for i in list(keep):
        trial = [j for j in keep if j != i]
        if classically_valid([prem[j] for j in trial], goal):
            keep = trial
    return keep


# ---------------------------------------------------------------- intuitionistic validity (G4ip)


def _neg_to_imp(f):
    if f[0] == "not":
        return ("imp", _neg_to_imp(f[1]), BOT)
    if f[0] in ("and", "or", "imp"):
        return (f[0], _neg_to_imp(f[1]), _neg_to_imp(f[2]))
    return f


@cache
def _g4ip(gamma: frozenset, c) -> bool:
    if BOT in gamma or c in gamma:
        return True
    # invertible left rules
    for a in gamma:
        rest = gamma - {a}
        t = a[0]
        if t == "and":
            return _g4ip(rest | {a[1], a[2]}, c)
        if t == "or":
            return _g4ip(rest | {a[1]}, c) and _g4ip(rest | {a[2]}, c)
        if t == "imp":
            x, y = a[1], a[2]
            if x == BOT:
                return _g4ip(rest, c)
            if x[0] == "atom" and x in gamma:
                return _g4ip(rest | {y}, c)
            if x[0] == "and":
                return _g4ip(rest | {("imp", x[1], ("imp", x[2], y))}, c)
            if x[0] == "or":
                return _g4ip(rest | {("imp", x[1], y), ("imp", x[2], y)}, c)
    # invertible right rules
    if c[0] == "and":
        return _g4ip(gamma, c[1]) and _g4ip(gamma, c[2])
    if c[0] == "imp":
        return _g4ip(gamma | {c[1]}, c[2])
    # non-invertible choices
    if c[0] == "or" and (_g4ip(gamma, c[1]) or _g4ip(gamma, c[2])):
        return True
    for a in gamma:
        if a[0] == "imp" and a[1][0] == "imp":
            (x, y), d = a[1][1:], a[2]
            rest = gamma - {a}
            if _g4ip(rest | {("imp", y, d)}, ("imp", x, y)) and _g4ip(rest | {d}, c):
                return True
    return False


def intuitionistically_valid(prem, goal) -> bool:
    return _g4ip(frozenset(_neg_to_imp(p) for p in prem), _neg_to_imp(goal))
