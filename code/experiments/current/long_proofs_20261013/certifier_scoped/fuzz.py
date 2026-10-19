"""Adversarial fuzz: random valid sequents (negation, F, double negation, disjunction, shared subformulas,
inconsistent premises, (A v A), tautological premises, conclusion inside a premise), exact minimum from
certify_fast (its minimal proofs re-checked in Lean), check sig_bound <= exact.

    python fuzz.py OUT.jsonl --n 3000 --seed 0 --procs 3 [--floor 16 --secs 5]
"""
import argparse
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'certifier_exact'))
import sigbound as SB  # noqa: E402

BOT = ('bot',)


def rand_formula(rng, atoms, pool, depth):
    if pool and rng.random() < 0.3:
        return rng.choice(pool)                         # shared subformula
    if depth == 0 or rng.random() < 0.25:
        return BOT if rng.random() < 0.06 else ('atom', rng.choice(atoms))
    x = rng.random()
    if x < 0.22:
        f = ('not', rand_formula(rng, atoms, pool, depth - 1))
    elif x < 0.27:
        f = ('not', ('not', rand_formula(rng, atoms, pool, depth - 1)))
    else:
        op = rng.choice(['and', 'or', 'imp', 'imp'])
        a = rand_formula(rng, atoms, pool, depth - 1)
        b = a if rng.random() < 0.07 else rand_formula(rng, atoms, pool, depth - 1)   # (A v A), (A > A)
        f = (op, a, b)
    pool.append(f)
    return f


def gen(rng):
    atoms = rng.sample(SB.C.LETTERS, rng.randint(2, 4))
    pool = []
    prem = [rand_formula(rng, atoms, pool, rng.randint(1, 3)) for _ in range(rng.randint(1, 5))]
    mode = rng.random()
    if mode < 0.15:                                       # inconsistent premises: ex falso
        a = ('atom', rng.choice(atoms))
        prem += [a, ('not', a)] if rng.random() < 0.5 else [('imp', a, BOT), a]
    if mode > 0.85 and pool:                              # conclusion inside a premise
        concl = rng.choice(pool)
    else:
        concl = rand_formula(rng, atoms, pool, rng.randint(1, 3))
    prem = list(dict.fromkeys(prem))
    return prem, concl


def _work(args):
    prompt, floor, secs, mem = args
    from certify_fast import certify_fast
    prem, concl = SB.parse_thm(prompt)
    b = SB.sig_bound(prem, concl)
    r = certify_fast(prompt, len(prem), floor=floor, secs=secs, mem_cap_mb=mem, want_proof=True)
    return {'prompt': prompt, **b, **{k: r.get(k) for k in ('min_body', 'cert_ge', 'timeout', 'min_proof')}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--procs', type=int, default=3)
    ap.add_argument('--floor', type=int, default=16)
    ap.add_argument('--secs', type=float, default=5.0)
    ap.add_argument('--mem', type=float, default=256)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    prompts, seen = [], set()
    while len(prompts) < a.n:
        prem, concl = gen(rng)
        if concl in prem or not SB.C.truth_table(prem, concl)[1]:
            continue
        p = 'THM ' + ' , '.join(SB.C.fstr(f) for f in prem) + ' SEQ ' + SB.C.fstr(concl) + ' PRF'
        if p not in seen:
            seen.add(p)
            prompts.append(p)
    viol = n_exact = 0
    ratios = []
    with ProcessPoolExecutor(a.procs) as ex:
        rs = list(ex.map(_work, [(p, a.floor, a.secs, a.mem) for p in prompts], chunksize=8))
    from lean_check import lean_ok
    found = [r for r in rs if r['min_proof']]
    for r, (ok, msg) in zip(found, lean_ok([r['prompt'] + ' ' + r['min_proof'] for r in found])):
        r['lean_ok'] = ok
        if not ok:                      # an exact label without a Lean-valid witness is not used
            print('LEAN REJECTS minimal proof', msg[:100], r['prompt'], flush=True)
            r['min_body'] = None
    with open(a.out, 'w') as out:
        for r in rs:
            out.write(json.dumps(r) + '\n')
            if r['min_body'] is not None:
                n_exact += 1
                ratios.append(r['lb'] / r['min_body'])
                if r['lb'] > r['min_body']:
                    viol += 1
                    print('VIOLATION', r, flush=True)
    ratios.sort()
    print(f'{len(prompts)} sequents, {n_exact} exact (Lean-checked witness), violations {viol}; lb/exact mean '
          f'{sum(ratios)/len(ratios):.3f} median {ratios[len(ratios)//2]:.3f} min {ratios[0]:.3f}')


if __name__ == '__main__':
    main()
