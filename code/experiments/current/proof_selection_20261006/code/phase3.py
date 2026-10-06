#!/usr/bin/env python3
"""Phase 3 (pre-registered in logs/worker_B.md): generator x judge matrix on the two test sets.

Pools: <set>__s{g}_r16__k64 for g in 0,1,2 (s0 = phase-1 pool).  Judge j score files: loglik_mean_masked (j = s0 on
the s0 pool), loglik_mean_masked_s{j} otherwise.  Per (set, generator, judge):
  auroc_within   within-theorem finished-only AUROC of the masked mean loglik (theorems with both classes among
                 finished attempts in that generator's pool; same theorem set for every judge of a generator)
  sel8 / sel64   fixed-k selection accuracy by completion_loglik (done ? loglik : -inf), k = 8 (R = 200 shared random
                 8-subsets per theorem, ties -> first in subset) and k = 64 (the whole pool)
Columns per generator: s0, s1, s2, siblings (mean of the two non-generator judges' loglik), xseed (mean of all three).
Self-judging penalty: paired theorem bootstrap (B = 2000, theorems resampled jointly across generators since all pools
share the same theorems) of (judge j - self) per generator, and the two-way fit cell = mu + gen + judge + delta*self.
Secondary: count-stop (N_max 16, c in {1, 2, 6}, R = 50 orderings) with self vs siblings selection; generator-seed
variance at k = 8 for random, completion, completion_loglik (self), oracle.
Usage: phase3.py --out results/B3_results.json
"""
import argparse, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auroc import auroc

RUNS = os.path.expanduser('~/runs/proof-selection')
GENS = ('s0', 's1', 's2')
SETS = ('holdout250', 'textbook72')


def rj(p):
    return {r['name']: r for r in map(json.loads, open(p))}


def load(set_, g):
    stem = f'{set_}__{g}_r16__k64'
    pool = rj(f'{RUNS}/pools_v2/{stem}.jsonl'); lab = rj(f'{RUNS}/pools_v2/{stem}.labels.jsonl')
    J = {}
    for j in GENS:
        f = 'loglik_mean_masked' if (g == 's0' and j == 's0') else f'loglik_mean_masked_{j}'
        J[j] = rj(f'{RUNS}/scores/{stem}__{f}.jsonl')
    names = sorted(pool)
    T = []
    for n in names:
        d = np.array([a['done'] for a in pool[n]['attempts']]); v = np.array(lab[n]['valid'], bool)
        L = {j: np.array(J[j][n]['scores'], float) for j in GENS}
        others = [j for j in GENS if j != g]
        L['siblings'] = (L[others[0]] + L[others[1]]) / 2
        L['xseed'] = (L['s0'] + L['s1'] + L['s2']) / 3
        T.append(dict(name=n, d=d, v=v, L=L))
    return names, T


COLS = ('s0', 's1', 's2', 'siblings', 'xseed')


def per_theorem(T, rng_seed=0):
    """per-theorem arrays: within-finished AUROC (nan if undefined) and k=8 / k=64 selection accuracy per column,
    plus random / completion / oracle at k=8, and count-stop (c=1,2,6; N_max 16) for self and siblings."""
    nT = len(T)
    out = {f'auc_{c}': np.full(nT, np.nan) for c in COLS}
    for c in COLS:
        out[f'sel8_{c}'] = np.zeros(nT); out[f'sel64_{c}'] = np.zeros(nT)
    for q in ('random8', 'completion8', 'oracle8'):
        out[q] = np.zeros(nT)
    for c in (1, 2, 6):
        for s in ('self', 'siblings'):
            out[f'count{c}_{s}'] = np.zeros(nT); out[f'count{c}_att'] = np.zeros(nT)
    for ti, t in enumerate(T):
        k = len(t['d']); f = t['d']
        rng = np.random.default_rng([rng_seed, ti])
        subs = np.stack([rng.permutation(k)[:8] for _ in range(200)])
        perms = np.stack([np.random.default_rng([rng_seed + 1, ti, r]).permutation(k) for r in range(50)])[:, :16]
        for c in COLS:
            x = t['L'][c]
            if f.sum() >= 2:
                out[f'auc_{c}'][ti] = auroc(x[f], t['v'][f])
            s = np.where(f, x, -np.inf)
            out[f'sel8_{c}'][ti] = t['v'][subs[np.arange(200), np.argmax(s[subs], 1)]].mean()
            out[f'sel64_{c}'][ti] = float(t['v'][np.argmax(s)])
        out['random8'][ti] = t['v'].mean()
        out['completion8'][ti] = t['v'][subs[np.arange(200), np.argmax(f[subs], 1)]].mean()
        out['oracle8'][ti] = t['v'][subs].any(1).mean()
        g_self = t['self']
        for c in (1, 2, 6):
            D = f[perms]; nf = np.cumsum(D, 1)
            stop = np.where((nf >= c).any(1), (nf >= c).argmax(1), 15)
            out[f'count{c}_att'][ti] = (stop + 1).mean()
            for s, col in (('self', g_self), ('siblings', 'siblings')):
                S = np.where(f, t['L'][col], -np.inf)[perms]
                acc = []
                for r in range(50):
                    pre = S[r, :stop[r] + 1]
                    acc.append(t['v'][perms[r, int(np.argmax(pre))]])
                out[f'count{c}_{s}'][ti] = np.mean(acc)
    return out


def twoway(Y):
    """Y[g, j] 3x3 -> delta (self effect) from least squares cell = mu + gen_g + judge_j + delta*[g==j]."""
    X, y = [], []
    for gi in range(3):
        for ji in range(3):
            row = [1.0, gi == 1, gi == 2, ji == 1, ji == 2, gi == ji]
            X.append(row); y.append(Y[gi, ji])
    b, *_ = np.linalg.lstsq(np.array(X, float), np.array(y), rcond=None)
    return b[5], b


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--B', type=int, default=2000)
    a = ap.parse_args()
    res = {}
    for set_ in SETS:
        P = {}
        names0 = None
        for g in GENS:
            names, T = load(set_, g)
            assert names0 is None or names == names0
            names0 = names
            for t in T:
                t['self'] = g
            P[g] = per_theorem(T)
        nT = len(names0)
        rng = np.random.default_rng(0)
        BS = [rng.integers(0, nT, nT) for _ in range(a.B)]
        R = dict(n_theorems=nT, cells={}, penalty={}, twoway={}, secondary={}, variance={})
        for metric in ('auc', 'sel8', 'sel64'):
            mean = (lambda x, ix=None: np.nanmean(x if ix is None else x[ix]))
            Y = np.array([[mean(P[g][f'{metric}_{j}']) for j in GENS] for g in GENS])
            R['cells'][metric] = {g: {c: float(mean(P[g][f'{metric}_{c}'])) for c in COLS} for g in GENS}
            if metric == 'auc':
                R['cells']['auc_n'] = {g: int(np.isfinite(P[g]['auc_s0']).sum()) for g in GENS}
            d0, b0 = twoway(Y)
            dbs = []
            for ix in BS:
                Yb = np.array([[mean(P[g][f'{metric}_{j}'], ix) for j in GENS] for g in GENS])
                dbs.append(twoway(Yb)[0])
            dbs = np.array(dbs)
            R['twoway'][metric] = dict(delta=float(d0), lo=float(np.nanquantile(dbs, .025)), hi=float(np.nanquantile(dbs, .975)),
                                       coef=dict(zip(['mu', 'gen_s1', 'gen_s2', 'judge_s1', 'judge_s2', 'self'], map(float, b0))))
            pen = {}
            for g in GENS:
                for j in list(GENS) + ['siblings', 'xseed']:
                    if j == g:
                        continue
                    d = P[g][f'{metric}_{j}'] - P[g][f'{metric}_{g}']
                    bs = np.array([np.nanmean(d[ix]) for ix in BS])
                    pen[f'{g}|{j}'] = dict(diff=float(np.nanmean(d)), lo=float(np.nanquantile(bs, .025)), hi=float(np.nanquantile(bs, .975)))
            R['penalty'][metric] = pen
        for g in GENS:
            sec = {}
            for c in (1, 2, 6):
                d = P[g][f'count{c}_siblings'] - P[g][f'count{c}_self']
                bs = np.array([d[ix].mean() for ix in BS])
                sec[f'count{c}'] = dict(att=float(P[g][f'count{c}_att'].mean()), self=float(P[g][f'count{c}_self'].mean()),
                                        siblings=float(P[g][f'count{c}_siblings'].mean()), diff=float(d.mean()),
                                        lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)))
            R['secondary'][g] = sec
            R['variance'][g] = {q: float(P[g][q].mean()) for q in ('random8', 'completion8', 'oracle8')}
            R['variance'][g]['completion_loglik_self8'] = float(P[g][f'sel8_{g}'].mean())
            R['variance'][g]['completion_loglik_siblings8'] = float(P[g]['sel8_siblings'].mean())
        res[set_] = R
        np.savez_compressed(f'{RUNS}/stopping/phase3_{set_}.npz', **{f'{g}__{k}': v for g in GENS for k, v in P[g].items()})
    json.dump(res, open(a.out, 'w'), indent=1)
    for set_, R in res.items():
        print('==', set_, R['n_theorems'])
        for metric in ('auc', 'sel8', 'sel64'):
            print(' ', metric)
            for g in GENS:
                print('   gen', g, ' '.join(f'{c}:{R["cells"][metric][g][c]:.3f}' for c in COLS))
            tw = R['twoway'][metric]
            print(f'   two-way self effect {tw["delta"]:+.4f} [{tw["lo"]:+.4f},{tw["hi"]:+.4f}]  coef {json.dumps({k: round(v, 4) for k, v in tw["coef"].items()})}')
            for k, v in R['penalty'][metric].items():
                print(f'     {k:14s} {v["diff"]:+.4f} [{v["lo"]:+.4f},{v["hi"]:+.4f}]')
        for g in GENS:
            print('  count-stop', g, {k: (round(v['att'], 2), round(v['self'], 3), round(v['siblings'], 3), f'{v["diff"]:+.3f} [{v["lo"]:+.3f},{v["hi"]:+.3f}]') for k, v in R['secondary'][g].items()})
            print('  k=8 variance', g, {k: round(v, 3) for k, v in R['variance'][g].items()})


if __name__ == '__main__':
    main()
