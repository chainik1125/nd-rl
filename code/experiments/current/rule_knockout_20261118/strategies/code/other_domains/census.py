# ruff: noqa
# Scratch analysis script written by a research agent on 2026-11-18; not reviewed.
"""Census of small purely implicational formulas by which structural rules their proofs need.
Formula: atom (int) or (A, B) for A -> B.
Proof search over beta-normal eta-long lambda terms.  A proof is summarised by its usage vector: how many times each
hypothesis in scope is used, capped at 2.  A lambda-bound hypothesis must satisfy the logic's discipline:
  BCI  (linear)        used exactly once      BCK  (affine)          used at most once
  BCIW (relevant, R->) used at least once     BCKW (intuitionistic)  any
Depth-bounded (term depth D), so 'not found' is exact only where normal proofs are size-bounded (BCI, BCK) or where a
separate complete check agrees (intuitionistic: loop-checked search; relevant: 3-element matrix refutation)."""

import itertools, sys, functools
from collections import Counter

sys.setrecursionlimit(10000)


def split(f):
    args = []
    while isinstance(f, tuple):
        args.append(f[0])
        f = f[1]
    return args, f


OK = {
    "BCI": lambda c: c == 1,
    "BCK": lambda c: c <= 1,
    "BCIW": lambda c: c >= 1,
    "BCKW": lambda c: True,
}


def prove(goal, logic, D):
    ok = OK[logic]
    need_bound = logic in ("BCI", "BCK")  # usage of every hypothesis <= 1: prune early

    @functools.lru_cache(maxsize=None)
    def go(ctx, goal, d):
        """ctx: tuple of formulas. returns frozenset of usage vectors (tuples, capped at 2) over ctx"""
        args, p = split(goal)
        n0 = len(ctx)
        full = ctx + tuple(args)
        res = set()
        if d == 0:
            return frozenset()
        for hi, h in enumerate(full):
            hargs, hp = split(h)
            if hp != p:
                continue
            base = [0] * len(full)
            base[hi] = 1
            partial = {tuple(base)}
            for a in hargs:
                sub = go(full, a, d - 1)
                if not sub:
                    partial = set()
                    break
                nxt = set()
                for u in partial:
                    for v in sub:
                        w = tuple(min(2, x + y) for x, y in zip(u, v))
                        if need_bound and max(w) > 1:
                            continue
                        nxt.add(w)
                partial = nxt
                if not partial:
                    break
            res |= partial
        out = set()
        for u in res:
            if all(ok(c) for c in u[n0:]):
                out.add(u[:n0])
        return frozenset(out)

    return bool(go((), goal, D))


# complete intuitionistic check for the implicational fragment: set contexts + loop check
def int_valid(goal):
    def go(ctx, goal, seen):
        args, p = split(goal)
        ctx = ctx | frozenset(args)
        key = (ctx, p)
        if key in seen:
            return False
        seen = seen | {key}
        for h in ctx:
            hargs, hp = split(h)
            if hp == p and all(go(ctx, a, seen) for a in hargs):
                return True
        return False

    return go(frozenset(), goal, frozenset())


def classical(f, natoms):
    def ev(f, v):
        return v[f] if not isinstance(f, tuple) else (not ev(f[0], v)) or ev(f[1], v)

    return all(ev(f, v) for v in itertools.product([False, True], repeat=natoms))


# matrices validating B, C, I, W + MP (no K): any formula they refute is not a theorem of R-> (BCIW)
def rm3(a, b):
    a -= 1
    b -= 1
    return (max(-a, b) if a <= b else min(-a, b)) + 1


RM3 = [[rm3(a, b) for b in range(3)] for a in range(3)]
KFREE = [(RM3, {1, 2}), ([(1, 1, 1), (0, 1, 0), (0, 1, 2)], {1, 2})]


def refuted_relevant(f, natoms):
    def ev(f, v, i):
        return v[f] if not isinstance(f, tuple) else i[ev(f[0], v, i)][ev(f[1], v, i)]

    return any(
        ev(f, v, i) not in Dd for i, Dd in KFREE for v in itertools.product(range(3), repeat=natoms)
    )


def shapes(n):
    if n == 1:
        yield None
        return
    for k in range(1, n):
        for l in shapes(k):
            for r in shapes(n - k):
                yield (l, r)


def fill(shape, it):
    if shape is None:
        return next(it)
    l = fill(shape[0], it)
    r = fill(shape[1], it)
    return (l, r)


def labelings(n, maxatoms):
    """restricted growth strings: atom labellings up to renaming"""

    def rec(pref, mx):
        if len(pref) == n:
            yield tuple(pref)
            return
        for a in range(min(mx + 1, maxatoms - 1) + 1):
            yield from rec(pref + [a], max(mx, a))

    yield from rec([0], 0)


def show(f, top=True):
    if not isinstance(f, tuple):
        return "PQRSTU"[f]
    s = f"{show(f[0], False)} → {show(f[1], True)}"
    return s if top else f"({s})"


if __name__ == "__main__":
    NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    D = 6
    tot = Counter()
    ex = {}
    rows = []
    for n in range(2, NMAX + 1):
        cnt = Counter()
        for sh in shapes(n):
            for lab in labelings(n, 4):
                f = fill(sh, iter(lab))
                na = max(lab) + 1
                if not classical(f, na):
                    cnt["not a tautology"] += 1
                    continue
                if not int_valid(f):
                    cnt["classical only (needs Peirce/DNE)"] += 1
                    ex.setdefault("classical only (needs Peirce/DNE)", f)
                    continue
                assert prove(f, "BCKW", D) or True
                lin = prove(f, "BCI", D)
                aff = prove(f, "BCK", D)
                rel = prove(f, "BCIW", D)
                assert not lin or (aff and rel)
                if lin:
                    c = "BCI: needs neither"
                elif aff and rel:
                    c = "provable with K alone and with W alone (no unique need)"
                elif aff:
                    c = (
                        "needs K only (BCK, not relevant)"
                        if refuted_relevant(f, na)
                        else "needs K only, UNCERTIFIED"
                    )
                elif rel:
                    c = "needs W only (relevant, not BCK)"
                else:
                    c = (
                        "needs K and W"
                        if refuted_relevant(f, na)
                        else "needs W; K need UNCERTIFIED (no relevant proof at depth %d, no matrix refutation)"
                        % D
                    )
                cnt[c] += 1
                ex.setdefault(c, f)
        rows.append((n, cnt))
        tot += cnt
        print(n, dict(cnt), flush=True)
    print("TOTAL", dict(tot))
    for c, f in ex.items():
        print(f"{c:60s} e.g. {show(f)}")
