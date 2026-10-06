#!/usr/bin/env python3
"""Scorer diagnostics (Worker B task 4): AUROC of each scorer for "valid".

For each pool and scorer: pooled AUROC over all attempts and over finished attempts only; within-theorem AUROC (mean
over theorems that have both a valid and an invalid candidate, all / finished-only); the same by per-theorem difficulty
bucket (pass rate 0-0.05, 0.05-0.25, 0.25-0.5, 0.5-0.9, 0.9-1).  Pooled AUROC mixes between-theorem difficulty with
within-theorem recognition; the within-theorem number is what selection among a theorem's own attempts can use, the
pooled number is what a global stopping threshold sees.  95 % intervals: theorem-level bootstrap (B=300).

Usage: auroc.py --pool <pool.jsonl> [--split dev] [--scores_dir ...] --out results/auroc_<stem>.json
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stopping import load_pool, RUNS

BUCKETS = [(0, 0.05), (0.05, 0.25), (0.25, 0.5), (0.5, 0.9), (0.9, 1.0001)]


def rankavg(x):
    o = np.argsort(x, kind='mergesort')
    xs = x[o]
    r = np.empty(len(x))
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def auroc(s, y):
    y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    s = np.where(np.isneginf(s), -1e300, np.where(np.isposinf(s), 1e300, s))
    r = rankavg(np.asarray(s, float))
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def metrics(T, name, ix):
    s_all = np.concatenate([T[i]['S'][name] for i in ix]); y_all = np.concatenate([T[i]['v'] for i in ix])
    d_all = np.concatenate([T[i]['d'] for i in ix])
    w_all, w_fin = [], []
    for i in ix:
        t = T[i]
        a = auroc(t['S'][name], t['v'])
        if not np.isnan(a):
            w_all.append(a)
        f = t['d']
        if f.sum() >= 2:
            a = auroc(t['S'][name][f], t['v'][f])
            if not np.isnan(a):
                w_fin.append(a)
    return dict(pooled_all=auroc(s_all, y_all), pooled_finished=auroc(s_all[d_all], y_all[d_all]),
                within_all=float(np.mean(w_all)) if w_all else np.nan, n_within_all=len(w_all),
                within_finished=float(np.mean(w_fin)) if w_fin else np.nan, n_within_finished=len(w_fin),
                n_theorems=len(ix), n_attempts=int(len(y_all)), n_finished=int(d_all.sum()),
                valid_rate=float(y_all.mean()) if len(y_all) else np.nan,
                p_valid_given_finished=float(y_all[d_all].mean()) if d_all.sum() else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', required=True)
    ap.add_argument('--split', default=None)
    ap.add_argument('--scores_dir', default=os.path.join(RUNS, 'scores'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--B', type=int, default=300)
    a = ap.parse_args()
    names = None
    if a.split:
        sp = json.load(open(os.path.join(RUNS, 'pools_v2', 'splits.json')))[a.split]
        names = set(n for v in sp.values() for n in v)
    stem, T, scs = load_pool(a.pool, a.scores_dir, None, names, None)
    scs = [s for s in scs if s not in ('oracle', 'random')]
    pr = np.array([t['v'].mean() for t in T])
    rng = np.random.default_rng(0)
    res = dict(pool=stem, split=a.split, scorers={})
    for s in scs:
        allix = np.arange(len(T))
        m = metrics(T, s, allix)
        bs = {k: [] for k in ('pooled_all', 'pooled_finished', 'within_all', 'within_finished')}
        for _ in range(a.B):
            mb = metrics(T, s, rng.integers(0, len(T), len(T)))
            for k in bs:
                bs[k].append(mb[k])
        for k in bs:
            x = np.array(bs[k]); x = x[~np.isnan(x)]
            m[k + '_ci'] = [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))] if len(x) else None
        m['buckets'] = {}
        for lo, hi in BUCKETS:
            ix = np.where((pr >= lo) & (pr < hi))[0]
            if len(ix):
                m['buckets'][f'{lo}-{min(hi, 1)}'] = metrics(T, s, ix)
        res['scorers'][s] = m
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1, default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
    print(f'{stem} split={a.split} theorems={len(T)}')
    print(f'{"scorer":22s} {"pooled":>7s} {"pooled_fin":>10s} {"within":>7s} {"within_fin":>10s}')
    for s, m in res['scorers'].items():
        print(f'{s:22s} {m["pooled_all"]:7.3f} {m["pooled_finished"]:10.3f} {m["within_all"]:7.3f} {m["within_finished"]:10.3f}')


if __name__ == '__main__':
    main()
