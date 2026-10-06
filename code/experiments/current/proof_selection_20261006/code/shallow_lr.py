#!/usr/bin/env python3
"""Supervised shallow baseline (design audit item 8): class-weighted L2 logistic regression on visible per-attempt
features, trained on the train split only, scored on every pool.

Features (pool rows only): done, steps, log1p(gen_tokens), log1p(prefill_tokens), base, mean emitted_len,
plus (if present and --extra given) Worker B's per-attempt scores, e.g. loglik_mean entropy_mean; every feature also
enters multiplied by done (so finished and unfinished attempts get separate slopes).  -inf/NaN extras -> column min.

  python shallow_lr.py --name shallow_lr_env
  python shallow_lr.py --name shallow_lr --extra loglik_mean entropy_mean
"""
import argparse, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pool_lib as PL
from evaluate_selection import auroc

SETS = ['rl_targets', 'dev1108', 'rr600', 'holdout250', 'textbook72']


def feats(s, extra):
    rows = PL.load_pool(s)
    ex = {e: PL.load_scores(s, e) for e in extra}
    X, names = [], []
    for r in rows:
        for a in r['attempts']:
            f = [float(a['done']), a['steps'], math.log1p(a['gen_tokens']), math.log1p(a['prefill_tokens']), a['base'],
                 float(np.mean(a['emitted_len']))]
            f += [ex[e][r['name']][a['i']] for e in extra]
            X.append(f); names.append((r['name'], a['i']))
    return np.array(X, float), names, rows


def design(X, mu=None, sd=None, fill=None):
    X = X.copy()
    if fill is None:
        fill = np.array([np.nanmin(np.where(np.isfinite(c), c, np.nan)) if np.isfinite(c).any() else 0.0 for c in X.T])
    bad = ~np.isfinite(X)
    X[bad] = np.take(fill, np.nonzero(bad)[1])
    if mu is None:
        mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    Z = np.concatenate([Z, Z * X[:, :1]], 1)          # done interactions (column 0 is done in {0,1})
    return np.concatenate([np.ones((len(Z), 1)), Z], 1), mu, sd, fill


def fit(Z, y, l2=1e-3, iters=50):
    w = np.zeros(Z.shape[1])
    pos = y.mean(); sw = np.where(y, (1 - pos) / pos, 1.0)       # balance by weighting
    for _ in range(iters):                                         # Newton / IRLS
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (sw * (p - y)) / len(y) + l2 * np.r_[0, w[1:]]
        H = (Z * (sw * p * (1 - p))[:, None]).T @ Z / len(y) + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='shallow_lr')
    ap.add_argument('--extra', nargs='*', default=[])
    ap.add_argument('--score_only', action='store_true', help='load the saved fit; read NO labels (permutation test)')
    ap.add_argument('--sets', nargs='*', default=SETS)
    a = ap.parse_args()
    mfn = os.path.join(PL.RUNS, 'verifier', f'_{a.name}_model.json')
    if a.score_only:
        M = json.load(open(mfn))
        w, mu, sd, fill = (np.array(M[k]) for k in ('weights', 'mu', 'sd', 'fill'))
        for s in a.sets:
            score(s, feats(s, M['extra']), w, mu, sd, fill, a.name)
        return
    sp = json.load(open(os.path.join(PL.SPRINT, 'results', 'splits.json')))
    data = {s: feats(s, a.extra) for s in SETS}
    def subset(split):
        Xs, ys = [], []
        for s, nm in sp[split].items():
            X, names, _ = data[s]; lab = PL.load_labels(s); keep = set(nm)
            m = np.array([n in keep for n, _ in names])
            Xs.append(X[m]); ys.append(np.array([lab[n][i] for n, i in names], bool)[m])
        return np.concatenate(Xs), np.concatenate(ys)
    Xtr, ytr = subset('train'); Xdv, ydv = subset('dev')
    Ztr, mu, sd, fill = design(Xtr)
    w = fit(Ztr, ytr.astype(float))
    pdv = design(Xdv, mu, sd, fill)[0] @ w
    print(f'{a.name}: train n={len(ytr)} dev AUROC {auroc(pdv, ydv):.4f}  extras={a.extra}')
    json.dump(dict(name=a.name, extra=a.extra, weights=w.tolist(), mu=mu.tolist(), sd=sd.tolist(), fill=fill.tolist(),
                   dev_auroc=auroc(pdv, ydv), n_train=int(len(ytr))), open(mfn, 'w'))
    for s in a.sets:
        score(s, data[s], w, mu, sd, fill, a.name)


def score(s, data, w, mu, sd, fill, name):
    X, names, rows = data
    p = design(X, mu, sd, fill)[0] @ w
    d = {}
    for (n, i), v in zip(names, p):
        d.setdefault(n, []).append(float(v))
    PL.write_jsonl(os.path.join(PL.SCORES, f'{PL.pool_stem(s)}__{name}.jsonl'),
                   [dict(name=r['name'], scores=d[r['name']]) for r in rows])


if __name__ == '__main__':
    main()
