"""Enumerate tiny theorems (few distinct subformulas) and ladder them through minlen bounds to find
ones with large exact minimal length. Writes jsonl of survivors at each rung.
    python search_small.py --atoms P Q --max_sub 7 --max_prem 2 --rungs 8 12 15 --secs 2 10 60 --out out/small.jsonl
"""
import argparse, itertools, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import certify
from certify import truth_table, fstr
from minlen import minlen, subformulas

def pool(atoms, max_conn):
    byc = {0: [('atom', a) for a in atoms]}
    for c in range(1, max_conn + 1):
        fs = []
        fs += [('not', f) for f in byc[c - 1] if f[0] != 'not']
        for i in range(c):
            for a in byc[i]:
                for b in byc[c - 1 - i]:
                    fs += [('and', a, b), ('or', a, b), ('imp', a, b)]
        byc[c] = fs
    return [f for c in byc for f in byc[c]]

ap = argparse.ArgumentParser()
ap.add_argument('--atoms', nargs='+', default=['P', 'Q'])
ap.add_argument('--max_conn', type=int, default=2)
ap.add_argument('--max_sub', type=int, default=7)
ap.add_argument('--max_prem', type=int, default=2)
ap.add_argument('--rungs', type=int, nargs='+', default=[8, 12, 15])
ap.add_argument('--secs', type=float, nargs='+', default=[2, 10, 60])
ap.add_argument('--out', required=True)
a = ap.parse_args()
P = pool(a.atoms, a.max_conn)
print(f'pool {len(P)} formulas', file=sys.stderr)
cands = []
for k in range(0, a.max_prem + 1):
    for prem in itertools.combinations(P, k):
        for concl in P:
            if concl in prem:
                continue
            sub = set()
            for f in list(prem) + [concl]:
                subformulas(f, sub)
            if len(sub) > a.max_sub:
                continue
            sat, valid = truth_table(list(prem), concl)
            if not (sat and valid):
                continue
            prompt = 'THM ' + ' , '.join(fstr(p) for p in prem) + (' ' if prem else '') + 'SEQ ' + fstr(concl) + ' PRF'
            cands.append({'prompt': prompt, 'n_prem': k, 'n_sub': len(sub)})
print(f'{len(cands)} valid consistent candidates', file=sys.stderr)
alive = cands
with open(a.out, 'w') as fo:
    for rung, secs in zip(a.rungs, a.secs):
        t0 = time.time(); nxt = []
        for c in alive:
            r = minlen(c['prompt'], bound=c['n_prem'] + rung, time_limit=secs)
            if r['min_lines_ub'] is not None:
                c['min_body'] = r['min_lines_ub'] - c['n_prem']; c['rung'] = rung
                fo.write(json.dumps(c) + '\n'); fo.flush()
            elif r['timeout']:
                c['timeout_at'] = rung; fo.write(json.dumps(c) + '\n'); fo.flush()
            else:
                nxt.append(c)
        print(f'rung {rung}: {len(alive)} tested, {len(nxt)} survive (no proof with body <= {rung}), {time.time()-t0:.0f}s', file=sys.stderr, flush=True)
        alive = nxt
    for c in alive:
        c['cert_ge_body'] = a.rungs[-1] + 1; fo.write(json.dumps(c) + '\n')
print('survivors:', file=sys.stderr)
for c in alive:
    print('  ', c['prompt'], file=sys.stderr)
