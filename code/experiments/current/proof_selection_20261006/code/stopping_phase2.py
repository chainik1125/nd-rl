#!/usr/bin/env python3
"""Phase 2 stopping / allocation families (Worker B; pre-registered in logs/worker_B.md before any phase-2 test read).

All policies consume attempts in R=50 seeded random orders (same as stopping.py), cap N_max = 16, and (unless stated)
return the incumbent = best completion_loglik so far (ties -> earliest; all -inf -> first consumed).

Families (each a list of parameter values -> per-theorem arrays acc/att/gen/tot/disp, averaged over orderings):
  fixed     k = 1..16
  ff        stop at first finished, N_max = 1..16
  count     stop after the c-th finished attempt, c = 1..8 (N_max 16)
  r1        quantile rule: stop when >= 2 attempts, incumbent finished and its masked loglik >= q-quantile of the masked
            loglik of ALL attempts so far
  r2        z rule: stop when >= 3 attempts, incumbent finished, (inc - mean(others)) / std(all so far) >= z
  r3        agreement: stop when some finished ND text has appeared r times; select the most frequent finished ND
            (ties -> higher completion_loglik), r = 2..6
  da / db   difficulty-aware allocation: d = verifier_ens of the first attempt (da) or mean of the first two (db);
            continue while p(d) * (1 - v(x_inc, d)) >= lambda, with p, v logistic fits from the dev split
            (fit_alloc); v = 0 with no finished incumbent.
Costs: generated tokens; total = prefill + generated + one likelihood pass per attempt (+ 3 verifier passes for each
attempt used to estimate d, in da/db).

Usage:
  stopping_phase2.py --fit --out_fit stopping/phase2_alloc_fit.json        (dev split of dev1108 + rr600)
  stopping_phase2.py --pool pools_v2/<stem>.jsonl [--split dev] --fit_file ... --grids stopping/phase2_grids.json
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stopping import load_pool, perms_for, RUNS

NMAX = 16
SEL = 'completion_loglik'
LL = 'loglik_mean_masked'
VER = 'verifier_ens'
DEV_POOLS = ['dev1108__s0_r16__k16', 'rr600__s0_r16__k32']


def split_names(split):
    sp = json.load(open(os.path.join(RUNS, 'pools_v2', 'splits.json')))[split]
    return set(n for v in sp.values() for n in v)


def load(stem, split=None, selector=None):
    names = split_names(split) if split else None
    _, T, _ = load_pool(os.path.join(RUNS, 'pools_v2', stem + '.jsonl'), os.path.join(RUNS, 'scores'), None, names,
                        sorted({SEL, LL, VER, "completion_loglik", "loglik_mean_masked"}))
    nd = {}
    for l in open(os.path.join(RUNS, 'pools_v2', stem + '.jsonl')):
        r = json.loads(l)
        if names is None or r['name'] in names:
            ids = {}
            nd[r['name']] = np.array([ids.setdefault(a['nd'], len(ids)) if a['done'] else -1 for a in r['attempts']])
    for t in T:
        t['ndid'] = nd[t['name']]
    return T


# ----------------------------------------------------------------------------------------------- logistic fits
def logit_fit(X, y, w=None, l2=1e-3, iters=50):
    X = np.column_stack([np.ones(len(X)), X])
    w = np.ones(len(y)) if w is None else w
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        g = X.T @ (w * (p - y)) + l2 * b
        H = (X * (w * p * (1 - p))[:, None]).T @ X + l2 * np.eye(X.shape[1])
        b -= np.linalg.solve(H, g)
    return b.tolist()


def sig(b, *xs):
    z = b[0] + sum(bi * x for bi, x in zip(b[1:], xs))
    return 1 / (1 + np.exp(-z))


def fit_alloc(rng_seed=0):
    """p(d): P(another attempt valid | d); v(x, d): P(finished attempt valid | its masked loglik x, d).
    d from one (da) or two (db) other attempts of the same theorem.  Dev split only."""
    rng = np.random.default_rng(rng_seed)
    out = {}
    T = sum((load(s, 'dev') for s in DEV_POOLS), [])
    for var, nd in (('da', 1), ('db', 2)):
        Xp, yp, Xv, yv = [], [], [], []
        for t in T:
            k = t['k']; ver = t['S'][VER]; ll = t['S'][LL]
            for _ in range(8):
                idx = rng.permutation(k)
                d = ver[idx[:nd]].mean()
                rest = idx[nd:]
                Xp.append([d]); yp.append(t['v'][rest].mean())
                fin = rest[t['d'][rest]]
                if len(fin):
                    j = rng.choice(fin)
                    Xv.append([ll[j], d]); yv.append(float(t['v'][j]))
        out[var] = dict(p=logit_fit(np.array(Xp), np.array(yp)), v=logit_fit(np.array(Xv), np.array(yv)),
                        n_p=len(yp), n_v=len(yv))
    return out


# ----------------------------------------------------------------------------------------------- simulation
def sim_theorem(t, P, fit, grids, Ns=NMAX):
    K = Ns
    P = P[:, :K]
    R = P.shape[0]
    S = t['S'][SEL][P]; V = t['v'][P].astype(float); D = t['d'][P]; LLp = t['S'][LL][P]
    VERp = t['S'][VER][P]; ND = t['ndid'][P]
    fwd = np.nan_to_num(t['fwd']); base_tot = t['gen'] + t['pre'] + fwd
    cg = np.cumsum(t['gen'][P], 1); ct = np.cumsum(base_tot[P], 1)
    vcost = 3 * np.minimum(1024.0, fwd)[P]
    M = np.maximum.accumulate(S, 1)
    new = np.ones_like(S, dtype=bool); new[:, 1:] = S[:, 1:] > M[:, :-1]
    A = np.maximum.accumulate(np.where(new, np.arange(K)[None], 0), 1)
    Vs = np.take_along_axis(V, A, 1)
    ever = np.maximum.accumulate(Vs, 1)
    incfin = np.isfinite(M)                                  # incumbent is finished iff its completion_loglik is finite
    incll = np.where(incfin, M, -np.inf)                     # completion_loglik == masked loglik for finished attempts
    nf = np.cumsum(D, 1)
    out = {}

    def take(stop, Vsel=Vs, extra_tot=None):
        g = lambda X: np.take_along_axis(X, stop[:, None], 1)[:, 0]
        tot = g(ct) + (0 if extra_tot is None else extra_tot)
        return [g(Vsel).mean(), (stop + 1).mean(), g(cg).mean(), tot.mean(),
                ((g(np.maximum.accumulate(Vsel, 1)) == 1) & (g(Vsel) == 0)).mean()]

    def first_true(sg):                                      # sg (R, K) -> first index, else K-1
        return np.where(sg.any(1), sg.argmax(1), K - 1)
    out['fixed'] = [take(np.full(R, k - 1)) for k in range(1, K + 1)]
    out['ff'] = [take(np.minimum(first_true(nf >= 1), N - 1)) for N in range(1, K + 1)]
    out['count'] = [take(first_true(nf >= c)) for c in range(1, 9)]
    # r1: quantile of all masked loglik so far
    qs = np.array(grids['r1'])
    qv = np.stack([np.quantile(LLp[:, :i + 1], qs, axis=1) for i in range(K)], -1)       # (Q, R, K)
    ok = (np.arange(K) >= 1)[None, None] & incfin[None] & (incll[None] >= qv)
    out['r1'] = [take(first_true(ok[j])) for j in range(len(qs))]
    # r2: z score of the incumbent against all attempts so far
    zs = np.array(grids['r2'])
    zt = np.full((R, K), -np.inf)
    for i in range(2, K):
        pre = LLp[:, :i + 1]
        sd = pre.std(1) + 1e-9
        others = (pre.sum(1) - np.where(incfin[:, i], incll[:, i], 0)) / i
        zt[:, i] = np.where(incfin[:, i], (incll[:, i] - others) / sd, -np.inf)
    out['r2'] = [take(first_true(zt >= z)) for z in zs]
    # r3: agreement / majority vote over finished ND texts
    vote_V = np.zeros((R, K)); maxc = np.zeros((R, K), dtype=int)
    for r in range(R):
        cnt = {}; best = None
        for i in range(K):
            n = ND[r, i]
            if n >= 0:
                c = cnt.get(n, (0, -np.inf, None))
                cnt[n] = (c[0] + 1, max(c[1], S[r, i]), i if c[2] is None else c[2])
            if cnt:
                best = max(cnt.items(), key=lambda kv: (kv[1][0], kv[1][1]))
                maxc[r, i] = best[1][0]; vote_V[r, i] = V[r, best[1][2]]
            else:
                vote_V[r, i] = V[r, 0]
    out['r3'] = [take(first_true(maxc >= rr), Vsel=vote_V) for rr in grids['r3']]
    # difficulty-aware allocation
    for var, nd in (('da', 1), ('db', 2)):
        f = fit[var]
        d = VERp[:, :nd].mean(1)                                                         # (R,)
        p = sig(f['p'], d)
        vinc = np.where(incfin, sig(f['v'], np.where(incfin, incll, 0.0), d[:, None]), 0.0)
        val = p[:, None] * (1 - vinc)                                                    # (R, K) value of continuing
        lam = np.array(grids[var])
        stopok = (val[None] < lam[:, None, None]) & (np.arange(K) >= nd - 1)[None, None]
        extra = vcost[:, :nd].sum(1)
        out[var] = [take(first_true(stopok[j]), extra_tot=extra) for j in range(len(lam))]
    return {k: np.array(v) for k, v in out.items()}                                     # (P, 5)


def simulate(T, fit, grids, R=50, seed=0):
    perms = perms_for(T, R, seed)
    res = None
    for ti, (t, P) in enumerate(zip(T, perms)):
        o = sim_theorem(t, P, fit, grids)
        if res is None:
            res = {k: np.zeros((v.shape[0], 5, len(T))) for k, v in o.items()}
        for k, v in o.items():
            res[k][:, :, ti] = v
    return res          # family -> (P, [acc, att, gen, tot, disp], nT)


def make_grids(fit):
    T = sum((load(s, 'dev') for s in DEV_POOLS), [])
    g = dict(r1=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0],
             r2=[-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0],
             r3=[2, 3, 4, 5, 6])
    for var, nd in (('da', 1), ('db', 2)):     # lambda grid: quantiles of the dev value distribution
        f = fit[var]; vals = []
        for t in T:
            d = t['S'][VER][:nd].mean()
            p = sig(f['p'], d)
            fin = t['d']
            vinc = np.where(fin, sig(f['v'], np.where(fin, t['S'][LL], 0.0), d), 0.0)
            vals += list(p * (1 - vinc)) + [p]
        g[var] = sorted(set(np.quantile(vals, np.linspace(0, 1, 61)).round(6).tolist()) | {0.0, 1.01})
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit', action='store_true')
    ap.add_argument('--fit_file', default=os.path.join(RUNS, 'stopping', 'phase2_alloc_fit.json'))
    ap.add_argument('--grids', default=os.path.join(RUNS, 'stopping', 'phase2_grids.json'))
    ap.add_argument('--pool', default=None)
    ap.add_argument('--split', default=None)
    ap.add_argument('--selector', default=None, help='secondary: e.g. siblings -> completion_loglik_siblings / '
                    'loglik_mean_masked_siblings for selection and the rank rules (allocation fit unchanged)')
    a = ap.parse_args()
    global SEL, LL
    if a.selector:
        SEL, LL = f'completion_loglik_{a.selector}', f'loglik_mean_masked_{a.selector}'
    if a.fit:
        fit = fit_alloc()
        json.dump(fit, open(a.fit_file, 'w'), indent=1)
        json.dump(make_grids(fit), open(a.grids, 'w'))
        print(json.dumps(fit))
        return
    fit = json.load(open(a.fit_file)); grids = json.load(open(a.grids))
    stem = os.path.basename(a.pool)[:-6]
    T = load(stem, a.split)
    res = simulate(T, fit, grids)
    sfx = (f'__{a.split}' if a.split else '') + (f'__sel_{a.selector}' if a.selector else '')
    np.savez_compressed(os.path.join(RUNS, 'stopping', f'{stem}{sfx}__phase2.npz'),
                        passrate=np.array([t['v'].mean() for t in T]), **{k: v for k, v in res.items()},
                        **{f'grid_{k}': np.array(v, dtype=float) for k, v in grids.items()})
    for k, v in res.items():
        print(stem, a.split, k, ' '.join(f'{v[i, 1].mean():.2f}:{v[i, 0].mean():.3f}' for i in range(0, v.shape[0], max(1, v.shape[0] // 8))))


if __name__ == '__main__':
    main()
