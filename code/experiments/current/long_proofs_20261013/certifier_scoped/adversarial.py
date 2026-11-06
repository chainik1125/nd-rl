"""Adversarial test: plant shortcuts in structured theorems with long proofs, then check sig_bound <= exact minimum.

Mutations of (premises, conclusion), each keeping the sequent valid:
  lemma    add a proper subformula of a premise as a new premise (a hard-won fact for free)
  dneg     add ~~X for a subformula X
  incons   add an atom and its negation (ex falso)
  contra   replace an implication premise (A > B) by (~B > ~A) or (~A v B)
  guard    add a guard atom (the antecedent of an implication subformula) as a premise
  dup      conjoin a subformula of one premise onto another premise
  concl    add (X > G) and X for a subformula X
  none     unchanged (exact check of the generator's own theorems)
Exact minimum: certify_fast at floor = written + 1 (a proof is found unless timeout); the minimal proof is
re-checked in Lean.

    python adversarial.py OUT.jsonl IN.jsonl [...] --n 2000 --seed 0 --procs 2 --secs 10
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
MUTS = ['lemma', 'dneg', 'incons', 'contra', 'guard', 'dup', 'concl']


def subs(f, acc):
    acc.append(f)
    if f[0] == 'not':
        subs(f[1], acc)
    elif f[0] in ('and', 'or', 'imp'):
        subs(f[1], acc)
        subs(f[2], acc)
    return acc


def mutate(rng, prem, concl, kind):
    prem = list(prem)
    allsub = [s for p in prem for s in subs(p, [])[1:]]
    if kind == 'none':
        pass
    elif kind == 'lemma':
        cands = [s for s in allsub if s[0] != 'atom'] or allsub
        prem.append(rng.choice(cands))
    elif kind == 'dneg':
        prem.append(('not', ('not', rng.choice(allsub))))
    elif kind == 'incons':
        a = rng.choice([s for s in allsub if s[0] == 'atom'] or [('atom', 'P')])
        prem += [a, ('not', a)]
    elif kind == 'contra':
        idx = [i for i, p in enumerate(prem) if p[0] == 'imp']
        if not idx:
            return None
        i = rng.choice(idx)
        A, B = prem[i][1], prem[i][2]
        prem[i] = ('imp', ('not', B), ('not', A)) if rng.random() < 0.5 else ('or', ('not', A), B)
    elif kind == 'guard':
        cands = [s[1] for s in allsub if s[0] == 'imp']
        if not cands:
            return None
        prem.append(rng.choice(cands))
    elif kind == 'dup':
        if len(prem) < 2:
            return None
        i, j = rng.sample(range(len(prem)), 2)
        prem[j] = ('and', prem[j], rng.choice(subs(prem[i], [])))
    elif kind == 'concl':
        X = rng.choice(allsub)
        prem += [('imp', X, concl), X]
    prem = list(dict.fromkeys(prem))
    if concl in prem or not SB.C.truth_table(prem, concl)[1]:
        return None
    return prem


def _work(args):
    prompt, floor, secs, mem, kind, src = args
    from certify_fast import certify_fast
    prem, concl = SB.parse_thm(prompt)
    b = SB.sig_bound(prem, concl)
    r = certify_fast(prompt, len(prem), floor=floor, secs=secs, mem_cap_mb=mem, want_proof=True)
    return {'prompt': prompt, 'kind': kind, 'src': src, 'floor': floor, **b,
            **{k: r.get(k) for k in ('min_body', 'cert_ge', 'timeout', 'min_proof')}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--procs', type=int, default=2)
    ap.add_argument('--secs', type=float, default=10.0)
    ap.add_argument('--mem', type=float, default=512)
    ap.add_argument('--max_body', type=int, default=22)
    ap.add_argument('--kinds', default=','.join(MUTS), help=f'comma list from {MUTS} + none')
    a = ap.parse_args()
    rng = random.Random(a.seed)
    kinds = a.kinds.split(',')
    base = []
    for p in a.paths:
        for line in open(p):
            r = json.loads(line)
            if 'text' in r and r.get('n_body', 99) <= a.max_body:
                base.append((r['text'].split(' PRF ', 1)[0] + ' PRF', r['n_body'], os.path.basename(p)))
    jobs, seen = [], set()
    tries = 0
    while len(jobs) < a.n and tries < 50 * a.n:
        tries += 1
        prompt, nb, src = rng.choice(base)
        prem, concl = SB.parse_thm(prompt)
        kind = rng.choice(kinds)
        newp = mutate(rng, prem, concl, kind)
        if newp is None or len(set().union(*[SB.C.atoms(f, set()) for f in newp + [concl]])) > 10:
            continue
        q = 'THM ' + ' , '.join(SB.C.fstr(f) for f in newp) + ' SEQ ' + SB.C.fstr(concl) + ' PRF'
        if q in seen:
            continue
        seen.add(q)
        # the original written proof still works for every mutation but 'contra' (which may need more lines)
        jobs.append((q, nb + 1 if kind != 'contra' else nb + 8, a.secs, a.mem, kind, src))
    with ProcessPoolExecutor(a.procs) as ex:
        rs = list(ex.map(_work, jobs, chunksize=4))
    from lean_check import lean_ok
    found = [r for r in rs if r['min_proof']]
    for r, (ok, msg) in zip(found, lean_ok([r['prompt'] + ' ' + r['min_proof'] for r in found])):
        r['lean_ok'] = ok
        if not ok:
            print('LEAN REJECTS minimal proof', msg[:100], r['prompt'], flush=True)
            r['min_body'] = None
    viol = 0
    by = {}
    with open(a.out, 'w') as out:
        for r in rs:
            out.write(json.dumps(r) + '\n')
            st = by.setdefault(r['kind'], [0, 0, 0, []])
            st[0] += 1
            if r['min_body'] is None:
                st[1] += 1
                continue
            st[3].append(r['lb'] / r['min_body'])
            if r['lb'] > r['min_body']:
                viol += 1
                st[2] += 1
                print('VIOLATION', r['lb'], r['min_body'], r['prompt'], flush=True)
    print(f'{len(rs)} mutated theorems, violations {viol}')
    for k, (n, nolab, v, rat) in sorted(by.items()):
        rat.sort()
        print(f'  {k:7s} n={n:5d} unlabelled(timeout)={nolab:4d} violations={v} lb/exact mean='
              f'{sum(rat)/max(1,len(rat)):.3f} min={rat[0] if rat else 0:.3f}')


if __name__ == '__main__':
    main()
