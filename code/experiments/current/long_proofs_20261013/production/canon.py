"""Exact theorem identity up to atom renaming and premise order: the lexicographically least renamed string over all
premise orders consistent with the skeleton sort (permutations tried within groups of equal skeleton, capped)."""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import certify as C  # noqa: E402


def skel(f):
    return ' '.join('a' if t in C.LETTERS else t for t in C.fstr(f).split())


def rename(order, concl):
    m, toks = {}, []
    for t in (' , '.join(C.fstr(p) for p in order) + ' |- ' + C.fstr(concl)).split():
        toks.append(m.setdefault(t, f'a{len(m)}') if t in C.LETTERS else t)
    return ' '.join(toks)


def canon_key(prem, concl, cap=5040):
    groups = [list(g) for _, g in itertools.groupby(sorted(prem, key=skel), key=skel)]
    n = 1
    for g in groups:
        n *= len(list(itertools.permutations(g))) if len(g) <= 7 else 10 ** 9
    if n > cap:
        return C.theorem_key(prem, concl)   # too many orders: fall back to the approximate key
    return min(rename([p for part in combo for p in part], concl)
               for combo in itertools.product(*[list(itertools.permutations(g)) for g in groups]))
