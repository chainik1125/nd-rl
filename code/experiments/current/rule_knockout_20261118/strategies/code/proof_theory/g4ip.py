# ruff: noqa
# Scratch analysis script written by a research agent on 2026-11-18; not reviewed.
"""G4ip (Dyckhoff 1992 LJT) decision procedure for intuitionistic and minimal propositional logic.  Scratch.

Formulas are the tuples of flags.py.  ¬A is read as A -> ⊥.  Minimal logic = the same calculus with ⊥ treated as an
ordinary atom (no ⊥-left rule).  prove(prem, concl, minimal=False, no_nested=False, budget=...) -> True/False, or None
when the step budget runs out.  no_nested=True disables the one non-invertible left rule, L→→ (using a hypothesis
(C -> D) -> B, which needs a proof of the implication C -> D): a True answer then means "provable without ever proving
an implication/negation in order to feed it to a hypothesis".
"""

import sys

sys.setrecursionlimit(100000)
BOT = ("bot",)


class Budget(Exception):
    pass


def norm(f):
    t = f[0]
    if t == "not":
        return ("imp", norm(f[1]), BOT)
    if t in ("and", "or", "imp"):
        return (t, norm(f[1]), norm(f[2]))
    return f


def prove(prem, concl, minimal=False, no_nested=False, budget=200000):
    memo = {}
    steps = [0]
    atomic = (lambda f: f[0] in ("atom", "bot")) if minimal else (lambda f: f[0] == "atom")

    def go(G, C):
        steps[0] += 1
        if steps[0] > budget:
            raise Budget()
        G = set(G)
        # saturate invertible left rules
        changed = True
        while changed:
            changed = False
            if C in G or (not minimal and BOT in G):
                return True
            for f in list(G):
                t = f[0]
                if t == "and":
                    G.discard(f)
                    G.add(f[1])
                    G.add(f[2])
                    changed = True
                elif t == "imp":
                    a, b = f[1], f[2]
                    if a in G:
                        G.discard(f)
                        G.add(b)
                        changed = True
                    elif a[0] == "and":
                        G.discard(f)
                        G.add(("imp", a[1], ("imp", a[2], b)))
                        changed = True
                    elif a[0] == "or":
                        G.discard(f)
                        G.add(("imp", a[1], b))
                        G.add(("imp", a[2], b))
                        changed = True
                    elif a == BOT and not minimal:
                        G.discard(f)
                        changed = True
        key = (frozenset(G), C)
        if key in memo:
            return memo[key]
        memo[key] = False  # loop guard (the calculus terminates; this is only a cheap cut)
        res = step(G, C)
        memo[key] = res
        return res

    def step(G, C):
        t = C[0]
        if t == "imp":
            return go(G | {C[1]}, C[2])
        if t == "and":
            return go(G, C[1]) and go(G, C[2])
        for f in G:
            if f[0] == "or":
                rest = G - {f}
                return go(rest | {f[1]}, C) and go(rest | {f[2]}, C)
        if t == "or":
            if go(G, C[1]) or go(G, C[2]):
                return True
        if not no_nested:
            for f in G:
                if f[0] == "imp" and f[1][0] == "imp":
                    (_, (_, c, d), b) = f
                    rest = G - {f}
                    if go(rest | {("imp", d, b)}, ("imp", c, d)) and go(rest | {b}, C):
                        return True
        return False

    try:
        return go({norm(p) for p in prem}, norm(concl))
    except Budget:
        return None


if __name__ == "__main__":
    A, B, C = ("atom", "A"), ("atom", "B"), ("atom", "C")
    imp = lambda a, b: ("imp", a, b)
    neg = lambda a: ("not", a)
    tests = [
        ("K", [], imp(A, imp(B, A)), True, True),
        ("S", [], imp(imp(A, imp(B, C)), imp(imp(A, B), imp(A, C))), True, True),
        ("exfalso", [], imp(BOT, A), True, False),
        ("DS", [("or", A, B), neg(A)], B, True, False),
        ("MT", [imp(A, B), neg(B)], neg(A), True, True),
        ("DNE", [], imp(neg(neg(A)), A), False, False),
        ("EM", [], ("or", A, neg(A)), False, False),
        ("Peirce", [], imp(imp(imp(A, B), A), A), False, False),
        ("nnEM", [], neg(neg(("or", A, neg(A)))), True, True),
        ("contrapos", [], imp(imp(A, B), imp(neg(B), neg(A))), True, True),
        ("tripleneg", [], imp(neg(neg(neg(A))), neg(A)), True, True),
        ("deMorgan-i", [], imp(neg(("or", A, B)), ("and", neg(A), neg(B))), True, True),
        ("deMorgan-c", [], imp(neg(("and", A, B)), ("or", neg(A), neg(B))), False, False),
        ("nnDNE", [], neg(neg(imp(neg(neg(A)), A))), True, False),
    ]
    for name, pr, c, wi, wm in tests:
        gi, gm = prove(pr, c), prove(pr, c, minimal=True)
        print(
            f"{name:12s} int {gi} (want {wi})  min {gm} (want {wm})  no-nested {prove(pr, c, no_nested=True)}",
            "OK" if (gi, gm) == (wi, wm) else "MISMATCH",
        )
