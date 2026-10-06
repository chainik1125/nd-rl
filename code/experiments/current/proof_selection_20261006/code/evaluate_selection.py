#!/usr/bin/env python3
"""Selection evaluation (schema §4).

For m in {1,2,4,8,16,32,64} (m <= k): R=200 seeded random m-subsets per theorem (the same subsets for every scorer),
pick the argmax-score attempt (ties broken by lowest attempt index i; NaN = -inf), and report mean selected-proof accuracy, oracle
pass@m on the same subsets, the random-pick expectation, and the fraction of the (oracle - random) gap recovered.
Paired theorem-level bootstrap (B=1000) CIs for each scorer and for differences between scorers.
Rows: all theorems ('all') and theorems with >= 1 valid attempt among the k ('solvable').
Also: attempt-level AUROC(valid) per scorer, pooled and on finished-only candidates.

  python evaluate_selection.py --set textbook72 [--scorers random first completion ...]
"""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pool_lib as PL

MS = [1, 2, 4, 8, 16, 32, 64]


def auroc(scores, labels):
    """Mann-Whitney AUROC with average ranks for ties; None if one class is empty."""
    s = np.asarray(scores, float); y = np.asarray(labels, bool)
    s = np.where(np.isnan(s), -np.inf, s)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(s, kind='mergesort')
    ss = s[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def subsets(k, m, R, seed):
    rng = np.random.default_rng(seed)
    if m >= k:
        return np.tile(np.arange(k), (1, 1)), rng
    return np.argsort(rng.random((R, k)), axis=1)[:, :m], rng


def per_theorem(valid, scores, m, R, seed):
    """-> (selected acc, oracle, random expectation) for one theorem, averaged over the R subsets."""
    v = np.asarray(valid, float); s = np.asarray(scores, float)
    s = np.where(np.isnan(s), -np.inf, s)
    idx, rng = subsets(len(v), m, R, seed)
    sv, ss = v[idx], s[idx]
    mx = ss.max(1, keepdims=True)
    pick = np.argmax(np.where(ss == mx, -idx.astype(float), -np.inf), axis=1)     # lowest attempt index among ties
    sel = sv[np.arange(len(sv)), pick]
    return sel.mean(), sv.max(1).mean(), sv.mean()


def evaluate(set_name, scorers, R=200, B=1000, seed=0, k=None, only=None):
    k_arg = k
    labels = PL.load_labels(set_name, k)
    pool = PL.load_pool(set_name, k)
    if only is not None:
        pool = [r for r in pool if r['name'] in only]
    names = [r['name'] for r in pool]
    done = {r['name']: [a['done'] for a in r['attempts']] for r in pool}
    k = pool[0]['k']
    S = {sc: PL.load_scores(set_name, sc, k_arg) for sc in scorers}
    for sc in scorers:
        assert set(names) <= set(S[sc]), f'{sc}: missing theorems'
        assert all(len(S[sc][n]) == k for n in names), f'{sc}: wrong length'
    solv = np.array([any(labels[n]) for n in names])
    out = dict(set=set_name, pool=PL.pool_stem(set_name, k_arg), k=k, n_theorems=len(names), n_solvable=int(solv.sum()),
               R=R, B=B, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), scorers=scorers,
               selection={}, paired={}, auroc={}, per_theorem={})
    brng = np.random.default_rng(seed + 7)
    boot = {'all': brng.integers(0, len(names), (B, len(names)))}
    si = np.flatnonzero(solv)
    boot['solvable'] = si[brng.integers(0, len(si), (B, len(si)))] if len(si) else None
    for m in [x for x in MS if x <= k]:
        acc = {}
        orc = rnd = None
        for sc in scorers:
            rows = [per_theorem(labels[n], S[sc][n], m, R, seed * 1000003 + t * 101 + m) for t, n in enumerate(names)]
            a = np.array(rows)
            acc[sc] = a[:, 0]
            orc, rnd = a[:, 1], a[:, 2]       # identical for every scorer (same subsets)
            out['per_theorem'].setdefault(str(m), {})[sc] = [round(float(x), 5) for x in a[:, 0]]
        out['per_theorem'][str(m)]['_oracle'] = [round(float(x), 5) for x in orc]
        out['per_theorem'][str(m)]['_random'] = [round(float(x), 5) for x in rnd]
        for sub in ('all', 'solvable'):
            mask = np.ones(len(names), bool) if sub == 'all' else solv
            bi = boot[sub]
            if bi is None:
                continue
            o, r_ = orc[mask].mean(), rnd[mask].mean()
            for sc in scorers:
                x = acc[sc]
                bs = x[bi].mean(1)
                gap = (x[mask].mean() - r_) / (o - r_) if o - r_ > 1e-12 else None
                bgap = (bs - rnd[bi].mean(1)) / np.maximum(orc[bi].mean(1) - rnd[bi].mean(1), 1e-12)
                out['selection'].setdefault(sub, {}).setdefault(str(m), {})[sc] = dict(
                    acc=float(x[mask].mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    oracle=float(o), random=float(r_), gap_recovered=gap,
                    gap_ci=[float(np.percentile(bgap, 2.5)), float(np.percentile(bgap, 97.5))])
            for i, a_ in enumerate(scorers):
                for b_ in scorers[i + 1:]:
                    d = acc[a_] - acc[b_]
                    bd = d[bi].mean(1)
                    out['paired'].setdefault(sub, {}).setdefault(str(m), {})[f'{a_} - {b_}'] = dict(
                        diff=float(d[mask].mean()), ci=[float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
                        p_le0=float((bd <= 0).mean()))
    for sc in scorers:
        sc_all = np.concatenate([S[sc][n] for n in names]); y = np.concatenate([labels[n] for n in names])
        fin = np.concatenate([done[n] for n in names]).astype(bool)
        within = [auroc(S[sc][n], labels[n]) for n in names]
        within = [w for w in within if w is not None]
        within_f = [auroc(np.asarray(S[sc][n])[np.asarray(done[n], bool)], np.asarray(labels[n])[np.asarray(done[n], bool)])
                    for n in names if any(done[n])]
        within_f = [w for w in within_f if w is not None]
        out['auroc'][sc] = dict(pooled=auroc(sc_all, y), finished_only=auroc(sc_all[fin], y[fin]),
                                within_theorem_mean=float(np.mean(within)) if within else None, n_within=len(within),
                                within_theorem_finished_mean=float(np.mean(within_f)) if within_f else None,
                                n_within_finished=len(within_f))
    # exploratory: per-theorem difficulty buckets (pass rate over the k attempts; labels used for evaluation only)
    BK = [(0, 0.001), (0.001, 0.05), (0.05, 0.25), (0.25, 0.5), (0.5, 0.9), (0.9, 1.01)]
    pr = np.array([np.mean(labels[n]) for n in names])
    m0 = '8' if '8' in out['per_theorem'] else str(max(x for x in MS if x <= k))
    out['by_bucket'] = {}
    for lo, hi in BK:
        tm = (pr >= lo) & (pr < hi)
        if not tm.any():
            continue
        b = dict(n_theorems=int(tm.sum()), m=int(m0),
                 oracle=float(np.array(out['per_theorem'][m0]['_oracle'])[tm].mean()),
                 random=float(np.array(out['per_theorem'][m0]['_random'])[tm].mean()), scorers={})
        sel_names = [n for n, t in zip(names, tm) if t]
        for sc in scorers:
            sc_all = np.concatenate([S[sc][n] for n in sel_names]); y = np.concatenate([labels[n] for n in sel_names])
            fin = np.concatenate([done[n] for n in sel_names]).astype(bool)
            b['scorers'][sc] = dict(acc=float(np.array(out['per_theorem'][m0][sc])[tm].mean()), auroc=auroc(sc_all, y),
                                    auroc_finished=auroc(sc_all[fin], y[fin]), n_finished=int(fin.sum()),
                                    p_valid_given_finished=float(y[fin].mean()) if fin.any() else None)
        out['by_bucket'][f'[{lo},{hi})'] = b
    out['names'] = names
    return out


def table(res, sub='all', ms=(1, 8, 64)):
    lines = [f"### {res['pool']}{' [' + res['split'] + ' split]' if res.get('split') else ''} ({sub}; n={res['n_theorems'] if sub == 'all' else res['n_solvable']})", '',
             '| scorer | ' + ' | '.join(f'acc@{m} [95% CI] | gap@{m}' for m in ms if str(m) in res['selection'][sub]) +
             ' | AUROC | AUROC fin |', '|---|' + '---|---|' * sum(str(m) in res['selection'][sub] for m in ms) + '---|---|']
    first = None
    for sc in res['scorers']:
        cells = []
        for m in ms:
            d = res['selection'][sub].get(str(m), {}).get(sc)
            if d is None:
                continue
            first = first or res['selection'][sub][str(m)][sc]
            g = '' if d['gap_recovered'] is None else f"{d['gap_recovered']:.2f}"
            cells += [f"{d['acc']:.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]", g]
        au = res['auroc'][sc]
        f = lambda x: '' if x is None else f'{x:.3f}'
        lines.append(f'| {sc} | ' + ' | '.join(cells) + f" | {f(au['pooled'])} | {f(au['finished_only'])} |")
    orc = [res['selection'][sub][str(m)][res['scorers'][0]] for m in ms if str(m) in res['selection'][sub]]
    lines.append('| *oracle* | ' + ' | '.join(f"{d['oracle']:.3f} | 1" for d in orc) + ' | | |')
    lines.append('| *random (expect.)* | ' + ' | '.join(f"{d['random']:.3f} | 0" for d in orc) + ' | | |')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', required=True)
    ap.add_argument('--scorers', nargs='*', default=None, help='default: every score file for this pool')
    ap.add_argument('--R', type=int, default=200)
    ap.add_argument('--B', type=int, default=1000)
    ap.add_argument('--out', default=None)
    ap.add_argument('--split', default=None, choices=['train', 'dev'], help='restrict to this split of results/splits.json')
    a = ap.parse_args()
    stem = PL.pool_stem(a.set)
    if not a.scorers:
        fs = sorted(glob.glob(os.path.join(PL.SCORES, f'{stem}__*.jsonl')))
        a.scorers = [os.path.basename(f)[len(stem) + 2:-6] for f in fs]
        pref = ['random', 'first', 'completion', 'shortest']
        a.scorers = [s for s in pref if s in a.scorers] + [s for s in a.scorers if s not in pref]
    only = None
    if a.split:
        only = set(json.load(open(os.path.join(PL.SPRINT, 'results', 'splits.json')))[a.split][a.set])
    res = evaluate(a.set, a.scorers, R=a.R, B=a.B, only=only)
    res['split'] = a.split
    out = a.out or os.path.join(PL.SPRINT, 'results', f'selection_{a.set}' + (f'_{a.split}' if a.split else '') + '.json')
    json.dump(res, open(out, 'w'))
    md = table(res, 'all') + '\n\n' + table(res, 'solvable') + '\n'
    open(out[:-5] + '.md', 'w').write(md)
    print(md)


if __name__ == '__main__':
    main()
