#!/usr/bin/env python3
"""Paired theorem-bootstrap CI (B=1000) for differences in within-theorem finished-only AUROC."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pool_lib as PL
from evaluate_selection import auroc
PAIRS = [('verifier2_ens', 'loglik_mean_masked'), ('verifier2_ll', 'loglik_mean_masked'), ('verifier2_ens', 'verifier_ens'),
         ('verifier2_rall_s0', 'loglik_mean_masked'), ('verifier2_rall_s1', 'loglik_mean_masked'), ('verifier2_rall_s2', 'loglik_mean_masked')]
out = {}
sp = json.load(open(os.path.join(PL.SPRINT, 'results', 'splits.json')))
for ev, sets in (('dev (orig pools)', [('dev1108', sp['dev']['dev1108']), ('rr600', sp['dev']['rr600'])]),
                 ('holdout250', [('holdout250', None)]), ('textbook72', [('textbook72', None)])):
    per = {}
    for s, names in sets:
        pool = {r['name']: r for r in PL.load_pool(s)}; lab = PL.load_labels(s)
        S = {sc: PL.load_scores(s, sc) for p in PAIRS for sc in p}
        for n in (names or list(pool)):
            d = [x['done'] for x in pool[n]['attempts']]
            y = [v for v, f in zip(lab[n], d) if f]
            row = {sc: auroc([v for v, f in zip(S[sc][n], d) if f], y) for sc in S}
            if row[PAIRS[0][1]] is not None:
                per[(s, n)] = row
    keys = list(per); rng = np.random.default_rng(0); bi = rng.integers(0, len(keys), (1000, len(keys)))
    out[ev] = {'n_theorems': len(keys)}
    for a, b in PAIRS:
        d = np.array([per[k][a] - per[k][b] for k in keys]); bs = d[bi].mean(1)
        out[ev][f'{a} - {b}'] = dict(diff=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
json.dump(out, open(os.path.join(PL.SPRINT, 'results', 'phase2_within_ci.json'), 'w'), indent=1)
for ev, d in out.items():
    print(ev, d['n_theorems'])
    for k, v in d.items():
        if k != 'n_theorems':
            print(f"  {k:45s} {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")
