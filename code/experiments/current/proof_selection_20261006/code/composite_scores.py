#!/usr/bin/env python3
"""completion_<scorer>: finished attempts first, ties broken by <scorer> (done*1e4 + score; -inf -> -1e3).
Reads pool rows (done) and score files only.   python composite_scores.py verifier_ens verifier_hist_ens ..."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pool_lib as PL
for sc in sys.argv[1:]:
    for s in PL.SET_FILES:
        S = PL.load_scores(s, sc)
        rows = PL.load_pool(s)
        f = lambda v: v if math.isfinite(v) else -1e3
        PL.write_jsonl(os.path.join(PL.SCORES, f'{PL.pool_stem(s)}__completion_{sc}.jsonl'),
                       [dict(name=r['name'], scores=[(1e4 if a['done'] else 0.0) + f(S[r['name']][a['i']])
                                                      for a in r['attempts']]) for r in rows])
    print('wrote completion_' + sc)
