#!/usr/bin/env python3
"""verifier_ens = mean of the three seeds' logits (pre-registered 02:08: mean of 3 seeds).  Reads score files only.
  python ensemble_scores.py [--members verifier_s0 verifier_s1 verifier_s2] [--name verifier_ens]"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pool_lib as PL
ap = argparse.ArgumentParser()
ap.add_argument('--members', nargs='+', default=['verifier_s0', 'verifier_s1', 'verifier_s2'])
ap.add_argument('--name', default='verifier_ens')
ap.add_argument('--sets', nargs='+', default=list(PL.SET_FILES))
a = ap.parse_args()
for s in a.sets:
    M = [PL.load_scores(s, m) for m in a.members]
    names = [r['name'] for r in PL.load_pool(s)]
    PL.write_jsonl(os.path.join(PL.SCORES, f'{PL.pool_stem(s)}__{a.name}.jsonl'),
                   [dict(name=n, scores=[float(x) for x in np.mean([m[n] for m in M], 0)]) for n in names])
    print('wrote', s, a.name)
