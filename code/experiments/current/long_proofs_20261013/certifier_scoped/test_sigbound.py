"""Tests for the signature bound.

1. kill table (brute force): for every perturbed formula X and every rule instance over a pool of formulas that
   contain X in many positions, if the instance's soundness inequality fails for some valuation (atoms + z), its
   signature must be in the perturbation's kill set.
   Everything is in the nu-system (negations unfolded, ~A := (A > F)), where NEGE / NEGI are IMPE / IMPI.
2. real proofs: for every proof given (written proofs, minlen minimal proofs), every requirement's kill set is hit by
   a line of that proof, every forced hypothesis appears as an AS line, and lb <= body length. Proof validity is
   not re-checked here (no nd_verify, per Robbie's rule); run lean_check.py on the same files for that.

    python test_sigbound.py [--proofs FILE.jsonl ...]
"""
import argparse
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sigbound as SB  # noqa: E402

V = SB.C.V          # only its parser is used
BOT = ('bot',)
P, Q, R = ('atom', 'P'), ('atom', 'Q'), ('atom', 'R')


def N(a):
    return ('imp', a, BOT)


def pool_with(X):
    base = [P, Q, R, BOT, X, N(X), N(N(X))]
    if X[0] != 'atom':
        base += [X[1]] + ([X[2]] if X[0] != 'not' else [])
    out = set(base)
    for a, b in itertools.product(base, repeat=2):
        for op in ('and', 'or', 'imp'):
            out.add((op, a, b))
    out |= {N(f) for f in list(out)}
    return sorted(out, key=SB.C.fstr)


def instances(pool):
    """(signature or None, list of (lhs-formulas, rhs) checks) for every rule instance whose formulas are in pool.
    Each check is a function I -> bool over the bitmask world."""
    for C in pool:
        k = C[0]
        if k == 'and':
            A, B = C[1], C[2]
            yield ('ANDI', C), lambda I, m, A=A, B=B, C=C: (I(A) & I(B)) & ~I(C) & m == 0
            yield ('ANDE1', C), lambda I, m, A=A, C=C: I(C) & ~I(A) & m == 0
            yield ('ANDE2', C), lambda I, m, B=B, C=C: I(C) & ~I(B) & m == 0
        elif k == 'imp':
            A, B = C[1], C[2]
            yield ('IMPE', C), lambda I, m, A=A, B=B, C=C: I(C) & I(A) & ~I(B) & m == 0
            yield ('IMPI', C), lambda I, m, A=A, B=B, C=C: ((m & ~I(A)) | I(B)) & ~I(C) & m == 0
            if B == BOT and A[0] == 'imp' and A[2] == BOT:
                yield ('DN', C), lambda I, m, G=A[1], C=C: I(C) & ~I(G) & m == 0
        elif k == 'or':
            A, B = C[1], C[2]
            yield ('ORI1', C), lambda I, m, A=A, C=C: I(A) & ~I(C) & m == 0
            yield ('ORI2', C), lambda I, m, B=B, C=C: I(B) & ~I(C) & m == 0
            yield ('ORE', C), lambda I, m, A=A, B=B, C=C: I(C) & ~(I(A) | I(B)) & m == 0
    # BOTE: I(F) = 0 always (F is never perturbed); R, PR, AS have no condition


def test_kill_table():
    names = ['P', 'Q', 'R']
    m, full = SB._masks(names)
    Xs = [P, BOT]
    for a, b in itertools.product([P, Q, N(P), (('and', P, Q)), ('imp', Q, P)], repeat=2):
        Xs += [('and', a, b), ('or', a, b), ('imp', a, b)]
    Xs += [N(P), N(N(P)), N(N(N(P))), N(('and', P, Q)), N(N(('or', P, Q))), N(BOT), N(N(BOT)), ('imp', P, P),
           ('and', P, P), ('or', P, P), ('or', N(P), N(N(P))), ('imp', BOT, BOT), ('imp', N(P), P)]
    n_checked = 0
    for X in Xs:
        for kind, pert, kills in SB.perturbations(X):
            cache = {}

            def I(f, cache=cache, X=X, pert=pert):
                return SB._eval(f, m, full, X, pert, cache)
            for sig, ok in instances(pool_with(X)):
                n_checked += 1
                if not ok(I, full) and sig not in kills:
                    raise AssertionError(f'unsound instance {sig} under {kind} of {SB.C.fstr(X)} not in kills')
    print(f'kill table: {n_checked} rule instances checked, all unsound ones are in their kill sets')


def proof_signatures(text):
    """-> (set of signatures, set of AS formulas, n_body) of a verified proof text."""
    prompt, body = text.split(' PRF ', 1)
    lines = V.parse_proof_tokens(body.split())
    f = {ln['idx']: SB.nu(ln['formula']) for ln in lines}
    alias = {'NEGE': 'IMPE', 'NEGI': 'IMPI'}
    sigs, hyps = set(), set()
    for ln in lines:
        r, refs, G = alias.get(ln['rule'], ln['rule']), ln['refs'], f[ln['idx']]
        if r in ('ANDI', 'IMPI', 'ORI1', 'ORI2'):
            sigs.add((r, G))
        elif r == 'IMPE' and ln['rule'] == 'NEGE':
            sigs.add((r, f[refs[1]]))           # NEGE A ~A: the negation is the major premise
        elif r in ('ANDE1', 'ANDE2', 'IMPE', 'ORE', 'DN'):
            sigs.add((r, f[refs[0]]))
        elif r == 'AS':
            hyps.add(G)
    return sigs, hyps, sum(1 for ln in lines if ln['rule'] != 'PR')


def check_proof(text):
    prompt = text.split(' PRF ', 1)[0] + ' PRF'
    prem, concl = SB.parse_thm(prompt)
    sigs, hyps, n_body = proof_signatures(text)
    reqs = SB.requirements(prem, concl)
    for X, kind, kills in reqs:
        if not (set(kills) & sigs):
            return f'requirement {kind} {SB.C.fstr(X)} not hit'
        if len(kills) == 1:
            rule, Y = kills[0]
            need = [Y[1], Y[2]] if rule == 'ORE' else [Y[1]] if rule == 'IMPI' else []
            for h in need:
                if h not in hyps:
                    return f'forced hypothesis {SB.C.fstr(h)} missing'
    b = SB.sig_bound(prem, concl)
    if b['lb'] > n_body:
        return f"lb {b['lb']} > body {n_body}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--proofs', nargs='*', default=[])
    a = ap.parse_args()
    test_kill_table()
    n = bad = 0
    for p in a.proofs:
        for line in open(p):
            r = json.loads(line)
            texts = []
            if 'text' in r:
                texts.append(r['text'])
            if r.get('min_proof'):
                texts.append(r['prompt'] + ' ' + r['min_proof'])
            for t in texts:
                n += 1
                err = check_proof(t)
                if err:
                    bad += 1
                    print('FAIL', err, t[:300])
    print(f'real proofs: {n} checked, {bad} failures')


if __name__ == '__main__':
    main()
