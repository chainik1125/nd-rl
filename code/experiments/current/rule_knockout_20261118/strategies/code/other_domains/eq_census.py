# ruff: noqa
# Scratch analysis script written by a research agent on 2026-11-18; not reviewed.
"""Feasibility check for the equational domain: one binary operation, axioms A (assoc), C (comm), I (idempotent).
Take random identities s = t that hold in semilattices (A+C+I: s and t have the same variable set) and ask, per
axiom X, whether the identity NEEDS X: i.e. some finite magma satisfying the other two axioms refutes it.
Countermodels: all magmas of size 2 and 3 (sound; 'not refuted' is then checked against exact normal forms)."""

import itertools, random
from collections import Counter


def magmas(n):
    for flat in itertools.product(range(n), repeat=n * n):
        yield [flat[k * n : (k + 1) * n] for k in range(n)]


def props(m):
    n = len(m)
    R = range(n)
    return (
        all(m[m[x][y]][z] == m[x][m[y][z]] for x in R for y in R for z in R),
        all(m[x][y] == m[y][x] for x in R for y in R),
        all(m[x][x] == x for x in R),
    )


models = {"A": [], "C": [], "I": []}  # models[X] = magmas satisfying the other two and violating X
for n in (2, 3):
    for m in magmas(n):
        a, c, i = props(m)
        if c and i and not a:
            models["A"].append(m)
        if a and i and not c:
            models["C"].append(m)
        if a and c and not i:
            models["I"].append(m)
print({k: len(v) for k, v in models.items()}, "countermodel magmas of size <= 3")


def ev(t, m, v):
    return v[t] if isinstance(t, int) else m[ev(t[0], m, v)][ev(t[1], m, v)]


def rand_term(rng, leaves, nv):
    if leaves == 1:
        return rng.randrange(nv)
    k = rng.randrange(1, leaves)
    return (rand_term(rng, k, nv), rand_term(rng, leaves - k, nv))


def vars_(t):
    return {t} if isinstance(t, int) else vars_(t[0]) | vars_(t[1])


def refuted(s, t, ms, nv):
    for m in ms:
        n = len(m)
        if any(ev(s, m, v) != ev(t, m, v) for v in itertools.product(range(n), repeat=nv)):
            return True
    return False


# exact normal forms for the three two-axiom theories
def flat(t):
    return [t] if isinstance(t, int) else flat(t[0]) + flat(t[1])


def nf_AC(t):
    return tuple(sorted(flat(t)))  # commutative semigroups: multiset of leaves


def nf_CI(t):  # commutative idempotent magmas: sort children, x*x -> x
    if isinstance(t, int):
        return t
    l, r = nf_CI(t[0]), nf_CI(t[1])
    if l == r:
        return l
    return tuple(sorted((l, r), key=repr))


rng = random.Random(0)
cnt = Counter()
agree = Counter()
N = 20000
for _ in range(N):
    nv = 3
    while True:
        s = rand_term(rng, rng.randint(2, 5), nv)
        t = rand_term(rng, rng.randint(1, 5), nv)
        if s != t and vars_(s) == vars_(t):
            break
    need = "".join(x for x in "ACI" if refuted(s, t, models[x], nv))
    cnt[need or "none (holds in all small models of every two-axiom subset)"] += 1
    # exactness of the small-model test where a normal form is available
    agree["I: model test == normal form"] += ("I" in need) == (nf_AC(s) != nf_AC(t))
    agree["A: model test == normal form"] += ("A" in need) == (nf_CI(s) != nf_CI(t))
print(N, "random semilattice identities, by set of axioms needed:", dict(sorted(cnt.items())))
print({k: f"{v}/{N}" for k, v in agree.items()})
