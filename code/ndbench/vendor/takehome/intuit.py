#!/usr/bin/env python3
"""Decision procedure for intuitionistic propositional logic (Dyckhoff's contraction-free calculus G4ip),
used ONLY to label theorems as classical-only (valid classically — a verifier-checked proof exists — but not
intuitionistically provable, so every proof must use DN).

  python intuit.py --selftest
  python intuit.py --in data/p2/targets_reductio.jsonl --out data/p2/targets_reductio_intuit.jsonl

Formulas are the verifier's tuples: ('atom',a) ('bot',) ('not',A) ('and',A,B) ('or',A,B) ('imp',A,B).
( ~ A ) is treated as ( A > F ).
"""
import argparse, json, sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nd_verify.verify import parse_formula

BOT = ('bot',)
sys.setrecursionlimit(10000)


def elim_not(f):
    if f[0] == 'not':
        return ('imp', elim_not(f[1]), BOT)
    if f[0] in ('and', 'or', 'imp'):
        return (f[0], elim_not(f[1]), elim_not(f[2]))
    return f


def prove(gamma, goal, memo=None):
    """G4ip: gamma is a tuple (multiset) of formulas, goal a formula. Returns True iff gamma => goal in LJ."""
    if memo is None:
        memo = {}
    key = (tuple(sorted(gamma)), goal)
    if key in memo:
        return memo[key]
    memo[key] = False   # guard (cycles cannot occur in G4ip, but be safe)
    res = _prove(gamma, goal, memo)
    memo[key] = res
    return res


def _prove(gamma, goal, memo):
    gl = list(gamma)
    # axioms
    if goal in gl and goal[0] == 'atom':
        return True
    if BOT in gl:
        return True
    if goal == BOT and any(g == BOT for g in gl):
        return True
    # right rules (invertible)
    if goal[0] == 'and':
        return prove(gamma, goal[1], memo) and prove(gamma, goal[2], memo)
    if goal[0] == 'imp':
        return prove(tuple(gl + [goal[1]]), goal[2], memo)
    # left rules on non-implications (invertible)
    for i, g in enumerate(gl):
        rest = gl[:i] + gl[i + 1:]
        if g[0] == 'and':
            return prove(tuple(rest + [g[1], g[2]]), goal, memo)
        if g[0] == 'or':
            return prove(tuple(rest + [g[1]]), goal, memo) and prove(tuple(rest + [g[2]]), goal, memo)
        if g[0] == 'imp':
            A = g[1]
            if A[0] == 'and':      # (A&B)>C  =>  A>(B>C)
                return prove(tuple(rest + [('imp', A[1], ('imp', A[2], g[2]))]), goal, memo)
            if A[0] == 'or':       # (AvB)>C  =>  A>C, B>C
                return prove(tuple(rest + [('imp', A[1], g[2]), ('imp', A[2], g[2])]), goal, memo)
            if A == BOT:           # F>C is trivially true: drop
                return prove(tuple(rest), goal, memo)
    # atom-implication left rule (invertible when the atom is present)
    for i, g in enumerate(gl):
        if g[0] == 'imp' and g[1][0] == 'atom' and g[1] in gl:
            rest = gl[:i] + gl[i + 1:]
            return prove(tuple(rest + [g[2]]), goal, memo)
    # non-invertible choices
    if goal[0] == 'or':
        if prove(gamma, goal[1], memo) or prove(gamma, goal[2], memo):
            return True
    for i, g in enumerate(gl):
        if g[0] == 'imp' and g[1][0] == 'imp':
            # (A>B)>C: from gamma, A, B>C prove B ; and from gamma, C prove goal
            A, B, C = g[1][1], g[1][2], g[2]
            rest = gl[:i] + gl[i + 1:]
            if prove(tuple(rest + [A, ('imp', B, C)]), B, memo) and prove(tuple(rest + [C]), goal, memo):
                return True
    return False


def parse_thm(prompt):
    toks = prompt.split()
    i = 1
    prem = []
    if toks[i] != 'SEQ':
        while True:
            f, i = parse_formula(toks, i)
            prem.append(f)
            if toks[i] == ',':
                i += 1; continue
            break
    concl, i = parse_formula(toks, i + 1)
    return prem, concl


def intuit_provable(prompt):
    prem, concl = parse_thm(prompt)
    return prove(tuple(elim_not(p) for p in prem), elim_not(concl))


def selftest():
    cases = [('THM ( P > Q ) , P SEQ Q PRF', True), ('THM ( ~ ( ~ P ) ) SEQ P PRF', False), ('THM P SEQ ( ~ ( ~ P ) ) PRF', True),
             ('THM SEQ ( P v ( ~ P ) ) PRF', False), ('THM ( ( P > Q ) > P ) SEQ P PRF', False), ('THM ( P > Q ) SEQ ( ( ~ Q ) > ( ~ P ) ) PRF', True),
             ('THM ( ( ~ Q ) > ( ~ P ) ) SEQ ( P > Q ) PRF', False), ('THM ( ~ ( P & Q ) ) SEQ ( ( ~ P ) v ( ~ Q ) ) PRF', False),
             ('THM ( ~ ( P v Q ) ) SEQ ( ( ~ P ) & ( ~ Q ) ) PRF', True), ('THM ( P v Q ) , ( ~ P ) SEQ Q PRF', True),
             ('THM P , ( ~ P ) SEQ Q PRF', True), ('THM ( ~ ( ~ ( ~ P ) ) ) SEQ ( ~ P ) PRF', True), ('THM ( P > Q ) , ( ~ Q ) SEQ ( ~ P ) PRF', True),
             ('THM SEQ ( ( ( P > Q ) > P ) > P ) PRF', False), ('THM ( ~ ( P > Q ) ) SEQ ( P & ( ~ Q ) ) PRF', False), ('THM ( P & ( ~ Q ) ) SEQ ( ~ ( P > Q ) ) PRF', True),
             ('THM ( P & ( Q v R ) ) SEQ ( ( P & Q ) v ( P & R ) ) PRF', True), ('THM ( ~ ( ~ ( P v Q ) ) ) SEQ ( P v Q ) PRF', False)]
    bad = 0
    for prompt, exp in cases:
        got = intuit_provable(prompt)
        bad += got != exp
        print('ok  ' if got == exp else 'FAIL', exp, got, prompt)
    print('INTUIT SELFTEST', 'PASS' if not bad else 'FAIL')
    # validation-36: compare with the classical_only flag
    vs = [json.loads(l) for l in open('targets/validation_36.jsonl')]
    dis = [(v['name'], v['classical_only'], not intuit_provable(v['prompt'])) for v in vs if v['classical_only'] != (not intuit_provable(v['prompt']))]
    print('validation-36 disagreements with classical_only flag:', dis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--in', dest='inp'); ap.add_argument('--out')
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    n = c = 0
    with open(a.out, 'w') as fo:
        for l in open(a.inp):
            r = json.loads(l)
            ip = intuit_provable(r['prompt'])
            n += 1; c += not ip
            fo.write(json.dumps({'name': r['name'], 'thm': r['thm'], 'intuit_provable': ip, 'classical_only': not ip}) + '\n')
    print(f'{a.inp}: {n} theorems, classical-only {c} ({c/n:.1%})')


if __name__ == '__main__':
    main()
