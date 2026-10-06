#!/usr/bin/env python3
"""Phase 6: draw-to-draw variance of the headline numbers. Draw 1 = <set>__s0_r16__k64 (seed 0), draw 2 =
<set>__s0_r16__k64b (seed 100), same generator.  Per draw: per-attempt validity, pass@8 (unbiased), pass@64, k=8
selection (200 shared random 8-subsets per theorem) for random / completion / completion_loglik self /
completion_loglik siblings / oracle, within-theorem finished-only AUROC (self, siblings), count-stop with sibling
selection at c=2 and c=6 (N_max 16, R=50 orderings).  Reports draw 2 - draw 1 with a paired theorem bootstrap and
each draw's theorem-bootstrap 95 % half-width.  -> results/B6_draw_compare.json"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auroc import auroc

RUNS = os.path.expanduser('~/runs/proof-selection')
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')


def rj(p):
    return {r['name']: r for r in map(json.loads, open(p))}


def per_theorem(stem, draw2):
    pool = rj(f'{RUNS}/pools_v2/{stem}.jsonl'); lab = rj(f'{RUNS}/pools_v2/{stem}.labels.jsonl')
    f0 = f'{RUNS}/scores/{stem}__loglik_mean_masked{"_s0" if draw2 else ""}.jsonl'
    J = {'s0': rj(f0), 's1': rj(f'{RUNS}/scores/{stem}__loglik_mean_masked_s1.jsonl'),
         's2': rj(f'{RUNS}/scores/{stem}__loglik_mean_masked_s2.jsonl')}
    names = sorted(pool)
    keys = ['validity', 'pass8', 'pass64', 'sel8_random', 'sel8_completion', 'sel8_self', 'sel8_siblings', 'sel8_oracle',
            'auc_self', 'auc_siblings', 'count2_att', 'count2_acc', 'count6_att', 'count6_acc']
    out = {k: np.full(len(names), np.nan) for k in keys}
    for ti, n in enumerate(names):
        d = np.array([a['done'] for a in pool[n]['attempts']]); v = np.array(lab[n]['valid'], bool); k = len(d)
        L0 = np.array(J['s0'][n]['scores']); Ls = (np.array(J['s1'][n]['scores']) + np.array(J['s2'][n]['scores'])) / 2
        c = v.sum()
        out['validity'][ti] = v.mean()
        out['pass8'][ti] = 1 - (math.comb(k - c, 8) / math.comb(k, 8) if k - c >= 8 else 0.0)
        out['pass64'][ti] = float(v.any())
        rng = np.random.default_rng([0, ti]); subs = np.stack([rng.permutation(k)[:8] for _ in range(200)])
        pick = lambda s: v[subs[np.arange(200), np.argmax(s[subs], 1)]].mean()
        out['sel8_random'][ti] = v.mean()
        out['sel8_completion'][ti] = pick(d.astype(float))
        out['sel8_self'][ti] = pick(np.where(d, L0, -np.inf))
        out['sel8_siblings'][ti] = pick(np.where(d, Ls, -np.inf))
        out['sel8_oracle'][ti] = v[subs].any(1).mean()
        if d.sum() >= 2:
            out['auc_self'][ti] = auroc(L0[d], v[d]); out['auc_siblings'][ti] = auroc(Ls[d], v[d])
        P = np.stack([np.random.default_rng([11, ti, r]).permutation(k) for r in range(50)])[:, :16]
        D = d[P]; V = v[P].astype(float); S = np.where(d, Ls, -np.inf)[P]; nf = np.cumsum(D, 1)
        M = np.maximum.accumulate(S, 1); new = np.ones_like(S, dtype=bool); new[:, 1:] = S[:, 1:] > M[:, :-1]
        A = np.maximum.accumulate(np.where(new, np.arange(16)[None], 0), 1); Vs = np.take_along_axis(V, A, 1)
        for cc in (2, 6):
            st = np.where((nf >= cc).any(1), (nf >= cc).argmax(1), 15)
            out[f'count{cc}_att'][ti] = (st + 1).mean(); out[f'count{cc}_acc'][ti] = Vs[np.arange(50), st].mean()
    return names, out


def main():
    res = {}
    for set_ in ('holdout250', 'textbook72'):
        n1, A = per_theorem(f'{set_}__s0_r16__k64', False)
        n2, B = per_theorem(f'{set_}__s0_r16__k64b', True)
        assert n1 == n2
        nT = len(n1); rng = np.random.default_rng(0); BS = [rng.integers(0, nT, nT) for _ in range(1000)]
        r = {}
        for k in A:
            m = lambda x, ix=None: np.nanmean(x if ix is None else x[ix])
            hw = lambda x: float((np.nanquantile([m(x, ix) for ix in BS], .975) - np.nanquantile([m(x, ix) for ix in BS], .025)) / 2)
            d = B[k] - A[k]; bs = np.array([np.nanmean(d[ix]) for ix in BS])
            r[k] = dict(draw1=float(m(A[k])), draw2=float(m(B[k])), diff=float(np.nanmean(d)),
                        diff_lo=float(np.nanquantile(bs, .025)), diff_hi=float(np.nanquantile(bs, .975)),
                        halfwidth_draw1=hw(A[k]), halfwidth_draw2=hw(B[k]))
        res[set_] = r
        print('==', set_)
        for k, x in r.items():
            print(f'  {k:16s} draw1 {x["draw1"]:.3f} draw2 {x["draw2"]:.3f} | diff {x["diff"]:+.3f} [{x["diff_lo"]:+.3f},{x["diff_hi"]:+.3f}] | theorem-bootstrap half-width {x["halfwidth_draw1"]:.3f} / {x["halfwidth_draw2"]:.3f}')
    json.dump(res, open(os.path.join(RES, 'B6_draw_compare.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
