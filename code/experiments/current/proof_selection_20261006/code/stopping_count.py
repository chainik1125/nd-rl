#!/usr/bin/env python3
"""EXPLORATORY (post-hoc, designed after the test read of the threshold policies; see logs/worker_B.md):
a scale-free stopping family that needs no absolute threshold.

  count-stop(c, N_max, selector): consume attempts until c finished attempts have been seen (or N_max), then return
  the best attempt so far under `selector` (ties -> earliest).  c = 1 with any completion-first selector is
  "stop at first finished".  Larger c spends more attempts exactly where finishing is rare relative to ... nothing:
  the budget adapts to the theorem's own finish rate, not to an absolute likelihood scale.

Same orderings (R=50, seed 0), costs, and bootstrap as stopping.py.  Output stopping/<stem><sfx>__countstop__<sel>.npz
with arrays acc/att/gen/tot/disp of shape (len(C), len(N), nT).
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stopping import load_pool, perms_for, scorer_cost, RUNS

CS = (1, 2, 3, 4, 6, 8)
NS = (8, 16, 32, 64)


def simulate_count(T, perms, sel, Cs=CS, Ns=NS):
    K = min(t['k'] for t in T)
    Ns = [n for n in Ns if n <= K]
    nT = len(T)
    out = {q: np.zeros((len(Cs), len(Ns), nT)) for q in ('acc', 'att', 'gen', 'tot', 'disp')}
    for ti, (t, P) in enumerate(zip(T, perms)):
        P = P[:, :K]
        S = t['S'][sel][P]
        V = t['v'][P].astype(float)
        D = t['d'][P]
        cg = np.cumsum(t['gen'][P], 1)
        ct = np.cumsum((t['gen'] + t['pre'] + scorer_cost(sel, t))[P], 1)
        M = np.maximum.accumulate(S, 1)
        new = np.ones_like(S, dtype=bool); new[:, 1:] = S[:, 1:] > M[:, :-1]
        A = np.maximum.accumulate(np.where(new, np.arange(K)[None], 0), 1)
        Vs = np.take_along_axis(V, A, 1)
        ever = np.maximum.accumulate(Vs, 1)
        nf = np.cumsum(D, 1)
        for ci, c in enumerate(Cs):
            hit = (nf >= c)
            first = np.where(hit.any(1), hit.argmax(1), K)          # index at which the c-th finished attempt arrives
            for ni, N in enumerate(Ns):
                stop = np.minimum(first, N - 1)
                g = lambda X: np.take_along_axis(X, stop[:, None], 1)[:, 0]
                out['acc'][ci, ni, ti] = g(Vs).mean(); out['att'][ci, ni, ti] = (stop + 1).mean()
                out['gen'][ci, ni, ti] = g(cg).mean(); out['tot'][ci, ni, ti] = g(ct).mean()
                out['disp'][ci, ni, ti] = ((g(ever) == 1) & (g(Vs) == 0)).mean()
    return dict(Cs=np.array(Cs), Ns=np.array(Ns), **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', required=True)
    ap.add_argument('--split', default=None)
    ap.add_argument('--selectors', default='completion,completion_loglik,completion_loglik_sum')
    ap.add_argument('--out_dir', default=os.path.join(RUNS, 'stopping'))
    a = ap.parse_args()
    names = None
    if a.split:
        sp = json.load(open(os.path.join(RUNS, 'pools_v2', 'splits.json')))[a.split]
        names = set(n for v in sp.values() for n in v)
    sels = a.selectors.split(',')
    stem, T, _ = load_pool(a.pool, os.path.join(RUNS, 'scores'), None, names, [s for s in sels if s != 'completion'])
    perms = perms_for(T, 50, 0)
    sfx = f'__{a.split}' if a.split else ''
    for s in sels:
        r = simulate_count(T, perms, s)
        np.savez_compressed(os.path.join(a.out_dir, f'{stem}{sfx}__countstop__{s}.npz'), **r)
        print(stem, s, 'c x N=16 (att, acc):',
              [(int(c), round(float(r['att'][ci, 1].mean()), 2), round(float(r['acc'][ci, 1].mean()), 3)) for ci, c in enumerate(r['Cs'])])


if __name__ == '__main__':
    main()
