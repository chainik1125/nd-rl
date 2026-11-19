# ruff: noqa
# Scratch analysis script written by a research agent on 2026-11-18; not reviewed.
"""Independence checks by brute-force 3-element logical matrices.
A matrix = (imp table 3x3, designated set D, constant f).  It 'validates' a schema if every valuation gives a
designated value, and must be closed under modus ponens (a in D, a->b in D  =>  b in D).
For each axiom X in a basis, find a matrix validating the others (and MP) but refuting X  => X is independent."""

import itertools, sys

V = range(3)


def mk(ax):  # ax: function (imp, f, vals...) -> value ; arity
    return ax


I_ = lambda i, f, a: i[a][a]
B_ = lambda i, f, a, b, c: i[i[b][c]][i[i[a][b]][i[a][c]]]
C_ = lambda i, f, a, b, c: i[i[a][i[b][c]]][i[b][i[a][c]]]
K_ = lambda i, f, a, b: i[a][i[b][a]]
W_ = lambda i, f, a, b: i[i[a][i[a][b]]][i[a][b]]
S_ = lambda i, f, a, b, c: i[i[a][i[b][c]]][i[i[a][b]][i[a][c]]]
DNE = lambda i, f, a: i[i[i[a][f]][f]][a]
EFQ = lambda i, f, a: i[f][a]
PEIRCE = lambda i, f, a, b: i[i[i[a][b]][a]][a]
AR = {
    "I": (I_, 1),
    "B": (B_, 3),
    "C": (C_, 3),
    "K": (K_, 2),
    "W": (W_, 2),
    "S": (S_, 3),
    "DNE": (DNE, 1),
    "EFQ": (EFQ, 1),
    "PEIRCE": (PEIRCE, 2),
}


def valid(name, i, f, D):
    fn, n = AR[name]
    return all(fn(i, f, *v) in D for v in itertools.product(V, repeat=n))


def mp_ok(i, D):
    return all(not (a in D and i[a][b] in D) or b in D for a in V for b in V)


def find(keep, drop):
    for D in [{2}, {1, 2}]:
        for flat in itertools.product(V, repeat=9):
            i = [flat[0:3], flat[3:6], flat[6:9]]
            if not mp_ok(i, D):
                continue
            for f in V:
                if all(valid(k, i, f, D) for k in keep) and not valid(drop, i, f, D):
                    return i, f, sorted(D)
    return None


def report(basis):
    print("basis", basis)
    for x in basis:
        r = find([b for b in basis if b != x], x)
        print(
            f"  {x:7s} independent of the rest: ",
            "YES, 3-element matrix imp=%s f=%s designated=%s" % r
            if r
            else "no 3-element matrix found",
        )


report(["B", "C", "I", "K", "W"])
report(["B", "C", "I", "K", "W", "DNE"])
report(["B", "C", "I", "K", "W", "PEIRCE"])
report(["B", "C", "I", "K", "W", "DNE", "EFQ"])
report(["K", "S"])
# named matrices
L3 = [[2, 2, 2], [1, 2, 2], [0, 1, 2]]  # Lukasiewicz 3-valued, values 0,1/2,1 -> 0,1,2
G3 = [[2, 2, 2], [0, 2, 2], [0, 1, 2]]  # Goedel/Heyting 3-chain


def rm3(a, b):
    a -= 1
    b -= 1
    return (max(-a, b) if a <= b else min(-a, b)) + 1


RM3 = [
    [rm3(a, b) for b in V] for a in V
]  # Sugihara RM3, values -1,0,1 -> 0,1,2 ; designated {0,1} -> {1,2}
for nm, i, f, D in [("L3", L3, 0, {2}), ("G3", G3, 0, {2}), ("RM3 (f=0)", RM3, 1, {1, 2})]:
    print(nm, "MP ok" if mp_ok(i, D) else "MP FAILS", {k: valid(k, i, f, D) for k in AR})
