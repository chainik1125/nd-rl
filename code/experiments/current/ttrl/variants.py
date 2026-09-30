"""Statement-only variants of a propositional sequent, for test-time RL (AlphaProof's TTRL).

A variant is a new sequent built from the target's premises and goal alone: no reference proof,
no model. Every variant is checked valid by truth table (classical logic over P, Q, R, S has 16
valuations, and the fork's ND is classical, so valid <=> provable). Kinds:

  dec_impi   goal A > B          ->  Γ, A |- B
  dec_negi   goal ~A             ->  Γ, A |- F
  dec_raa    any goal G != F     ->  Γ, ~G |- F
  dec_andi   goal A & B          ->  Γ |- A   and   Γ |- B
  dec_ori    goal A v B          ->  Γ |- A   or    Γ |- B   (only the valid ones)
  dec_ande   premise A & B       ->  replace it by A, B
  dec_ore    premise A v B       ->  replace it by A (case 1), by B (case 2)
  spec       premise A > B       ->  replace it by B, or by ~A      (stronger premise)
             premise ~(A & B)    ->  replace it by ~A, or by ~B     (stronger premise)
  cut_left   ψ a subformula of Γ or the goal, Γ |= ψ   ->  Γ |- ψ
  cut_right  same ψ                                     ->  Γ, ψ |- G

Dropped: the target itself, anything whose goal is already a premise, duplicates (premises as a
set). `depth` > 1 applies the same moves to the variants.
"""

import itertools

ATOMS = "PQRS"
BINARY = "&v>"


def parse(text):
    """'( ( P > Q ) & ( ~ R ) )' -> ('&', ('>', 'P', 'Q'), ('~', 'R')). Atoms and F are strings."""
    toks = text.split()
    pos = 0

    def go():
        nonlocal pos
        t = toks[pos]
        pos += 1
        if t != "(":
            assert t in ATOMS or t == "F", t
            return t
        if toks[pos] == "~":
            pos += 1
            a = go()
            assert toks[pos] == ")"
            pos += 1
            return ("~", a)
        a = go()
        op = toks[pos]
        assert op in BINARY, op
        pos += 1
        b = go()
        assert toks[pos] == ")"
        pos += 1
        return (op, a, b)

    f = go()
    assert pos == len(toks), text
    return f


def show(f):
    if isinstance(f, str):
        return f
    if f[0] == "~":
        return f"( ~ {show(f[1])} )"
    return f"( {show(f[1])} {f[0]} {show(f[2])} )"


def parse_thm(thm):
    """'A , B |- G' (a pool record's `thm`) -> ([A, B], G)."""
    lhs, rhs = thm.split("|-")
    prems = [parse(p) for p in lhs.split(" , ")] if lhs.strip() else []
    return prems, parse(rhs.strip())


def prompt(prems, goal):
    """The pool's prompt format: 'THM A , B SEQ G PRF'."""
    return f"THM {' , '.join(show(p) for p in prems)} SEQ {show(goal)} PRF".replace(
        "THM  SEQ", "THM SEQ"
    )


def value(f, v):
    if isinstance(f, str):
        return False if f == "F" else v[f]
    if f[0] == "~":
        return not value(f[1], v)
    a, b = value(f[1], v), value(f[2], v)
    return a and b if f[0] == "&" else a or b if f[0] == "v" else (not a) or b


VALUATIONS = [
    dict(zip(ATOMS, bits, strict=True)) for bits in itertools.product((False, True), repeat=4)
]


def entails(prems, goal):
    return all(value(goal, v) for v in VALUATIONS if all(value(p, v) for p in prems))


def subformulas(f):
    yield f
    if not isinstance(f, str):
        for g in f[1:]:
            yield from subformulas(g)


def key(prems, goal):
    return (frozenset(prems), goal)


def _replace(prems, i, new):
    return prems[:i] + [x for x in new if x not in prems] + prems[i + 1 :]


def moves(prems, goal):
    """One step of every kind: yields (kind, premises, goal). Not yet validity-filtered."""
    if not isinstance(goal, str):
        op = goal[0]
        if op == ">":
            yield "dec_impi", prems + [goal[1]], goal[2]
        if op == "~":
            yield "dec_negi", prems + [goal[1]], "F"
        if op == "&":
            yield "dec_andi", prems, goal[1]
            yield "dec_andi", prems, goal[2]
        if op == "v":
            yield "dec_ori", prems, goal[1]
            yield "dec_ori", prems, goal[2]
    if goal != "F":
        yield "dec_raa", prems + [("~", goal)], "F"
    for i, p in enumerate(prems):
        if isinstance(p, str):
            continue
        if p[0] == "&":
            yield "dec_ande", _replace(prems, i, [p[1], p[2]]), goal
        if p[0] == "v":
            yield "dec_ore", _replace(prems, i, [p[1]]), goal
            yield "dec_ore", _replace(prems, i, [p[2]]), goal
        if p[0] == ">":
            yield "spec", _replace(prems, i, [p[2]]), goal
            yield "spec", _replace(prems, i, [("~", p[1])]), goal
        if p[0] == "~" and not isinstance(p[1], str) and p[1][0] == "&":
            yield "spec", _replace(prems, i, [("~", p[1][1])]), goal
            yield "spec", _replace(prems, i, [("~", p[1][2])]), goal
    seen = set()
    for f in itertools.chain(*(subformulas(x) for x in prems + [goal])):
        if f in seen or f in prems or f == goal:
            continue
        seen.add(f)
        if entails(prems, f):
            yield "cut_left", prems, f
            yield "cut_right", prems + [f], goal


def variants(prems, goal, depth=1):
    """Valid, non-trivial, deduplicated variants: dicts with kind (a/b for depth 2), prems, goal."""
    done = {key(prems, goal)}
    out, frontier = [], [("", prems, goal)]
    for _ in range(depth):
        nxt = []
        for path, ps, g in frontier:
            for kind, vp, vg in moves(ps, g):
                k = key(vp, vg)
                if k in done or vg in vp or not entails(vp, vg):
                    continue
                done.add(k)
                v = {"kind": f"{path}/{kind}" if path else kind, "prems": vp, "goal": vg}
                out.append(v)
                nxt.append((v["kind"], vp, vg))
        frontier = nxt
    return out
