#!/usr/bin/env python3
"""Reviewer's independent held-out recount for run noise-floor.
Re-verifies every recorded proof with nd_verify (not trusting the 'solved' flag),
joins to data/p2/heldout.jsonl by name for my own slice predicates.
"""
import json, os, sys, collections, math
sys.path.insert(0, '/home/dan/review/noise-floor')
from nd_verify import verify_text

ROOT = '/home/dan/review/noise-floor'
pool = {}
for l in open(f'{ROOT}/data/p2/heldout.jsonl'):
    r = json.loads(l)
    pool[r['name']] = r

def slices(r):
    """my own predicates, from the pool record only"""
    out = ['overall', f"len{r['n_lines']}"]
    if r['pat']['depth3']:
        out.append('depth3')
    if r['n_lines'] == 6 and not (r['pat']['depth3'] or r['pat']['reductio'] or r['pat']['derived_ore']):
        out.append('len6_nopat')
    return out

def cell(pi, si):
    f = f'{ROOT}/artifacts/nf/heldout_p{pi}_s{si}.jsonl'
    if not os.path.exists(f):
        return None
    num = collections.Counter(); den = collections.Counter()
    seen = set(); mismatch = 0; nprf = 0
    for l in open(f):
        d = json.loads(l)
        nm = d['name']; seen.add(nm)
        r = pool[nm]
        # re-verify: solved iff at least one recorded proof passes nd_verify on my own call
        ok_any = False
        for p in d.get('proofs') or []:
            nprf += 1
            ok, reason, nl = verify_text(r['prompt'] + ' ' + p)
            if ok:
                ok_any = True
        if bool(d['solved']) != ok_any:
            mismatch += 1
        for s in slices(r):
            den[s] += 1
            if ok_any:
                num[s] += 1
    assert seen == set(pool), (len(seen), len(pool))
    return {'num': dict(num), 'den': dict(den), 'mismatch': mismatch, 'n_proofs': nprf}

if __name__ == '__main__':
    out = {}
    for pi in (1, 2, 3, 4):
        for si in range(13):
            c = cell(pi, si)
            if c is None:
                print('MISSING', pi, si); continue
            out[f'p{pi}_s{si}'] = c
            print(f"p{pi}_s{si} overall {c['num']['overall']}/{c['den']['overall']} "
                  f"d3 {c['num'].get('depth3',0)}/{c['den'].get('depth3',0)} "
                  f"l6 {c['num'].get('len6',0)}/{c['den'].get('len6',0)} "
                  f"l6np {c['num'].get('len6_nopat',0)}/{c['den'].get('len6_nopat',0)} "
                  f"judge-mismatch {c['mismatch']}", flush=True)
    json.dump(out, open(f'{ROOT}/rv/heldout_recount.json', 'w'), indent=1)
