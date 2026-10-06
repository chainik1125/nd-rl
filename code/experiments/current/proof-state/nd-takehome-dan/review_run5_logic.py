#!/usr/bin/env python3
"""Reviewer's independent logic checks on the run-5 pools (no minlen involved).
  classical validity  : truth table over the 4 atoms.
  intuitionistic      : Dyckhoff's contraction-free calculus G4ip (decision procedure) — a theorem provable in G4ip has a
                        DN-free natural-deduction proof (DN is the only classical rule in spec.md), so a reductio target must be
                        G4ip-UNprovable to require DN at all.
"""
import json, sys, os, itertools, functools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from review_run5_recount import parse_formula, thm_of_prompt, rd

sys.setrecursionlimit(10000)
ATOMS = ('P', 'Q', 'R', 'S')


def parse_thm(thm):
    prem, concl = thm.split('|-')
    prems = [p.strip() for p in prem.split(',') if p.strip()]
    ps = [parse_formula(p.split(), 0)[0] for p in prems]
    c = parse_formula(concl.split(), 0)[0]
    return ps, c


def ev(f, v):
    if f == 'F':
        return False
    if isinstance(f, str):
        return v[f]
    if f[0] == '~':
        return not ev(f[1], v)
    a, b = ev(f[1], v), ev(f[2], v)
    return {'&': a and b, 'v': a or b, '>': (not a) or b}[f[0]]


def classical_valid(ps, c):
    for bits in itertools.product([False, True], repeat=4):
        v = dict(zip(ATOMS, bits))
        if all(ev(p, v) for p in ps) and not ev(c, v):
            return False
    return True


# ---- G4ip.  Formulas: atoms (str), 'F', ('~',A) -> ('>',A,'F'), ('&',A,B), ('v',A,B), ('>',A,B)
def imp_form(f):
    if isinstance(f, str):
        return f
    if f[0] == '~':
        return ('>', imp_form(f[1]), 'F')
    return (f[0], imp_form(f[1]), imp_form(f[2]))


@functools.lru_cache(maxsize=None)
def g4ip(gamma, goal):
    """gamma: frozenset of formulas; goal: formula. True iff Γ ⇒ goal is provable in G4ip (intuitionistic)."""
    if 'F' in gamma or goal in gamma and isinstance(goal, str):
        return True
    # invertible left rules first
    for a in gamma:
        if isinstance(a, tuple):
            rest = gamma - {a}
            if a[0] == '&':
                return g4ip(rest | {a[1], a[2]}, goal)
            if a[0] == 'v':
                return g4ip(rest | {a[1]}, goal) and g4ip(rest | {a[2]}, goal)
            if a[0] == '>':
                ante, cons = a[1], a[2]
                if ante == 'F':
                    return g4ip(rest, goal)                  # F > C is useless
                if isinstance(ante, str) and ante in gamma:
                    return g4ip(rest | {cons}, goal)         # L→ atom
                if isinstance(ante, tuple):
                    if ante[0] == '&':
                        return g4ip(rest | {('>', ante[1], ('>', ante[2], cons))}, goal)
                    if ante[0] == 'v':
                        return g4ip(rest | {('>', ante[1], cons), ('>', ante[2], cons)}, goal)
    # invertible right rules
    if isinstance(goal, tuple):
        if goal[0] == '&':
            return g4ip(gamma, goal[1]) and g4ip(gamma, goal[2])
        if goal[0] == '>':
            return g4ip(gamma | {goal[1]}, goal[2])
    # non-invertible: R∨, and L→→
    if isinstance(goal, tuple) and goal[0] == 'v':
        if g4ip(gamma, goal[1]) or g4ip(gamma, goal[2]):
            return True
    for a in gamma:
        if isinstance(a, tuple) and a[0] == '>' and isinstance(a[1], tuple) and a[1][0] == '>':
            rest = gamma - {a}
            b, c = a[1], a[2]     # (A>B)>C
            if g4ip(rest | {('>', b[2], c)}, b) and g4ip(rest | {c}, goal):
                return True
    return False


def intuitionistic(ps, c):
    g4ip.cache_clear()
    return g4ip(frozenset(imp_form(p) for p in ps), imp_form(c))


def selftest():
    cases = [('( P > Q ) , P |- Q', True, True), ('( ~ ( ~ P ) ) |- P', True, False), ('|- ( P v ( ~ P ) )', True, False),
             ('|- ( ( ( P > Q ) > P ) > P )', True, False), ('( P > Q ) , ( ~ Q ) |- ( ~ P )', True, True), ('P |- ( ~ ( ~ P ) )', True, True),
             ('( ~ ( P & Q ) ) |- ( ( ~ P ) v ( ~ Q ) )', True, False), ('( ( ~ P ) v ( ~ Q ) ) |- ( ~ ( P & Q ) )', True, True),
             ('( ~ ( P v Q ) ) |- ( ( ~ P ) & ( ~ Q ) )', True, True), ('( P v Q ) , ( ~ P ) |- Q', True, True), ('P |- Q', False, False),
             ('( P > Q ) |- ( ( ~ P ) v Q )', True, False), ('|- ( ( P > Q ) v ( Q > P ) )', True, False), ('F |- P', True, True),
             ('( ~ ( ~ ( P v ( ~ P ) ) ) ) |- ( P v ( ~ P ) )', True, False), ('|- ( ~ ( ~ ( P v ( ~ P ) ) ) )', True, True)]
    bad = 0
    for thm, cv, iv in cases:
        ps, c = parse_thm(thm)
        got = (classical_valid(ps, c), intuitionistic(ps, c))
        print('ok  ' if got == (cv, iv) else 'FAIL', thm, got)
        bad += got != (cv, iv)
    print('LOGIC SELFTEST', 'PASS' if not bad else 'FAIL')
    return bad == 0


if __name__ == '__main__':
    assert selftest()
    out = {}
    for fn in ['data/p2/targets_reductio_req.jsonl', 'data/p2/transfer_reductio_req.jsonl', 'data/p2/targets_derived_ore_req.jsonl', 'data/p2/transfer_derived_ore_req.jsonl']:
        n = cv = iv = 0; names_i = []
        for r in rd(fn):
            ps, c = parse_thm(thm_of_prompt(r['prompt'])); n += 1
            cv += classical_valid(ps, c)
            i = intuitionistic(ps, c); iv += i
            if i:
                names_i.append(r['name'])
        out[fn] = {'n': n, 'classically_valid': cv, 'intuitionistically_provable': iv, 'intuitionistic_names': names_i[:20]}
        print(f'{fn:40s} n {n} classically valid {cv} intuitionistically provable (G4ip) {iv}', flush=True)
    json.dump(out, open('review_out/logic.json', 'w'), indent=1)
