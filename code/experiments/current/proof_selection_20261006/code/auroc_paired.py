#!/usr/bin/env python3
"""Paired theorem-bootstrap of AUROC differences (finished attempts only): scorer A - scorer B, pooled and within-theorem
(theorems with both classes among finished attempts).  Usage: auroc_paired.py <pool.jsonl> A B [B=2000]"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stopping import load_pool, RUNS
from auroc import auroc
pool, A, Bn = sys.argv[1], sys.argv[2], sys.argv[3]
split = sys.argv[4] if len(sys.argv) > 4 else None
names = None
if split:
    sp = json.load(open(os.path.join(RUNS, "pools_v2", "splits.json")))[split]; names = set(n for v in sp.values() for n in v)
stem, T, _ = load_pool(pool, os.path.join(RUNS, "scores"), None, names, [A, Bn])
def stats(ix):
    sa = np.concatenate([T[i]['S'][A][T[i]['d']] for i in ix]); sb = np.concatenate([T[i]['S'][Bn][T[i]['d']] for i in ix])
    y = np.concatenate([T[i]['v'][T[i]['d']] for i in ix])
    w = [(auroc(T[i]['S'][A][T[i]['d']], T[i]['v'][T[i]['d']]), auroc(T[i]['S'][Bn][T[i]['d']], T[i]['v'][T[i]['d']])) for i in ix]
    w = np.array([x for x in w if not np.isnan(x[0])])
    return auroc(sa, y) - auroc(sb, y), (w[:, 0] - w[:, 1]).mean(), len(w)
rng = np.random.default_rng(0)
d0 = stats(np.arange(len(T)))
bs = np.array([stats(rng.integers(0, len(T), len(T)))[:2] for _ in range(2000)])
print(json.dumps(dict(pool=stem, A=A, B=Bn, pooled_fin_diff=d0[0], pooled_ci=list(np.quantile(bs[:, 0], [.025, .975])),
                      within_fin_diff=d0[1], within_ci=list(np.quantile(bs[:, 1], [.025, .975])), n_within=d0[2])))
