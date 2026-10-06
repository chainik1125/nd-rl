#!/usr/bin/env python3
"""Phase 5 (pre-registered in logs/worker_B.md): absolute-confidence stopping with sibling / xseed judges.

Policies (N_max 16; R = 50 seeded orderings of the first 16 attempts; incumbent = best completion_loglik so far
under the policy's judge, ties -> earliest):
  T-<judge>  stop when the incumbent's judge loglik >= tau            (tau grid: dev finished-attempt quantiles)
  C-<judge>  stop when P(valid | judge loglik, log steps) >= p*        (logistic fit on dev finished attempts)
  count      stop after the c-th finished attempt, select by the judge (c = 1..8)
  oracle     stop at the first valid attempt (N_max 1..16)
judge in {sib (mean of the two non-generator r16 seeds), x (mean of all three)}.
Usage: phase5.py dev | test
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stopping_phase2 import logit_fit, sig

RUNS = os.path.expanduser('~/runs/proof-selection')
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
GENS = ('s0', 's1', 's2'); M_T = (2, 3, 4, 6, 8); K = 16; R = 50


def rj(p):
    return {r['name']: r for r in map(json.loads, open(p))}


def jfile(stem, g, j):
    return f'{RUNS}/scores/{stem}__loglik_mean_masked.jsonl' if (g == 's0' and j == 's0') else f'{RUNS}/scores/{stem}__loglik_mean_masked_{j}.jsonl'


def load(stem, g, names=None):
    pool = rj(f'{RUNS}/pools_v2/{stem}.jsonl'); lab = rj(f'{RUNS}/pools_v2/{stem}.labels.jsonl')
    J = {j: rj(jfile(stem, g, j)) for j in GENS}
    sib = [j for j in GENS if j != g]
    T = []
    for n in sorted(pool):
        if names is not None and n not in names:
            continue
        L = {j: np.array(J[j][n]['scores'], float) for j in GENS}
        T.append(dict(name=n, d=np.array([a['done'] for a in pool[n]['attempts']]), v=np.array(lab[n]['valid'], bool),
                      steps=np.array([a['steps'] for a in pool[n]['attempts']], float),
                      J={'sib': (L[sib[0]] + L[sib[1]]) / 2, 'x': (L['s0'] + L['s1'] + L['s2']) / 3}))
    return T


def dev_T():
    sp = json.load(open(f'{RUNS}/pools_v2/splits.json'))['dev']
    names = set(n for v in sp.values() for n in v)
    return load('dev1108__s0_r16__k16', 's0', names) + load('rr600__s0_r16__k32', 's0', names)


def fit_and_grids(T):
    out = {}
    for j in ('sib', 'x'):
        X, y = [], []
        for t in T:
            f = t['d']
            X += list(zip(t['J'][j][f], np.log(t['steps'][f]))); y += list(t['v'][f].astype(float))
        X = np.array(X); y = np.array(y)
        b = logit_fit(X, y)
        out[j] = dict(logit=b, tau=sorted(set(np.quantile(X[:, 0], np.linspace(0, 1, 101)).round(6).tolist())),
                      p=np.round(np.linspace(0, 1, 101), 2).tolist())
    return out


def simulate(T, fg):
    """-> dict policy -> (P params, 2 [acc, att], nT)"""
    nT = len(T)
    res = {}
    def put(k, i, ti, acc, att, P):
        res.setdefault(k, np.zeros((P, 2, nT)))[i, :, ti] = (acc, att)
    for ti, t in enumerate(T):
        Pm = np.stack([np.random.default_rng([11, ti, r]).permutation(len(t['d'])) for r in range(R)])[:, :K]
        V = t['v'][Pm].astype(float); D = t['d'][Pm]
        nf = np.cumsum(D, 1); nv = np.cumsum(V, 1)
        ev = np.maximum.accumulate(V, 1)
        first = lambda sg: np.where(sg.any(1), sg.argmax(1), K - 1)
        for N in range(1, K + 1):
            st = np.minimum(np.where((nv >= 1).any(1), (nv >= 1).argmax(1), K), N - 1)
            put('oracle', N - 1, ti, ev[np.arange(R), st].mean(), (st + 1).mean(), K)
        for j in ('sib', 'x'):
            S = np.where(t['d'], t['J'][j], -np.inf)[Pm]
            M = np.maximum.accumulate(S, 1)
            new = np.ones_like(S, dtype=bool); new[:, 1:] = S[:, 1:] > M[:, :-1]
            A = np.maximum.accumulate(np.where(new, np.arange(K)[None], 0), 1)
            Vs = np.take_along_axis(V, A, 1)
            inc_steps = np.log(np.take_along_axis(t['steps'][Pm], A, 1))
            pinc = np.where(np.isfinite(M), sig(fg[j]['logit'], np.where(np.isfinite(M), M, 0.0), inc_steps), 0.0)
            g = lambda st: (Vs[np.arange(R), st].mean(), (st + 1).mean())
            for c in range(1, 9):
                put(f'count_{j}', c - 1, ti, *g(first(nf >= c)), 8)
            taus = fg[j]['tau'] + [np.inf]
            for i, tau in enumerate(taus):
                put(f'T_{j}', i, ti, *g(first(M >= tau)), len(taus))
            for i, p in enumerate(fg[j]['p']):
                put(f'C_{j}', i, ti, *g(first(pinc >= p)), len(fg[j]['p']))
    return res


def main():
    mode = sys.argv[1]
    fgf = os.path.join(RUNS, 'stopping', 'phase5_fit_grids.json')
    chf = os.path.join(RES, 'B5_dev_choice.json')
    if mode == 'dev':
        T = dev_T()
        fg = fit_and_grids(T); json.dump(fg, open(fgf, 'w'))
        res = simulate(T, fg)
        ch = {}
        for k, a in res.items():
            att = a[:, 1].mean(-1); acc = a[:, 0].mean(-1)
            ch[k] = {str(m): dict(idx=int(np.argmin(np.abs(att - m))), dev_att=float(att[np.argmin(np.abs(att - m))]),
                                  dev_acc=float(acc[np.argmin(np.abs(att - m))])) for m in M_T}
        json.dump(ch, open(chf, 'w'), indent=1)
        for k in ('count_sib', 'T_sib', 'C_sib', 'T_x', 'C_x', 'count_x'):
            print(k, ' '.join(f'm{m}:{v["dev_att"]:.2f}/{v["dev_acc"]:.3f}' for m, v in ch[k].items()))
        print('logit', {j: fg[j]['logit'] for j in fg})
        return
    fg = json.load(open(fgf)); ch = json.load(open(chf))
    out = {}
    for set_ in ('holdout250', 'textbook72'):
        per = {g: simulate(load(f'{set_}__{g}_r16__k64', g), fg) for g in GENS}
        nT = per['s0']['oracle'].shape[-1]
        rng = np.random.default_rng(0); BS = [rng.integers(0, nT, nT) for _ in range(2000)]
        for j in ('sib', 'x'):
            for pol in (f'T_{j}', f'C_{j}'):
                for m in M_T:
                    m = str(m)
                    ip, ic = ch[pol][m]['idx'], ch[f'count_{j}'][m]['idx']
                    row = dict(gens={})
                    dsum = np.zeros(nT); agp = []; agc = []; gap = []
                    for g in GENS:
                        a = per[g][pol][ip]; c = per[g][f'count_{j}'][ic]; o = per[g]['oracle']
                        d = a[0] - c[0]; bs = np.array([d[ix].mean() for ix in BS])
                        oatt = o[:, 1].mean(-1); oacc = o[:, 0].mean(-1)
                        o_at = float(np.interp(a[1].mean(), oatt, oacc)) if a[1].mean() <= oatt.max() else float(oacc.max())
                        row['gens'][g] = dict(att=float(a[1].mean()), acc=float(a[0].mean()), count_att=float(c[1].mean()),
                                              count_acc=float(c[0].mean()), diff=float(d.mean()), lo=float(np.quantile(bs, .025)),
                                              hi=float(np.quantile(bs, .975)), oracle_stop_at_att=o_at)
                        dsum += d / 3; agp.append(a[1].mean()); agc.append(c[1].mean())
                        den = o_at - c[0].mean()
                        gap.append((a[0].mean() - c[0].mean()) / den if abs(den) > 1e-9 else np.nan)
                    bs = np.array([dsum[ix].mean() for ix in BS])
                    row.update(pooled_diff=float(dsum.mean()), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)),
                               att=float(np.mean(agp)), count_att=float(np.mean(agc)), target=float(m),
                               acc=float(np.mean([row['gens'][g]['acc'] for g in GENS])),
                               count_acc=float(np.mean([row['gens'][g]['count_acc'] for g in GENS])),
                               gap_closed=float(np.nanmean(gap)))
                    out[f'{set_}|{pol}|m{m}'] = row
                    print(f'{set_[:10]} {pol:6s} m={m}: att {row["att"]:5.2f} acc {row["acc"]:.3f} | count att {row["count_att"]:5.2f} acc {row["count_acc"]:.3f} '
                          f'| diff {row["pooled_diff"]:+.4f} [{row["lo"]:+.4f},{row["hi"]:+.4f}] | gap closed {row["gap_closed"]:+.2f} | per gen '
                          + ' '.join(f'{g}:{v["diff"]:+.3f}[{v["lo"]:+.3f},{v["hi"]:+.3f}]@{v["att"]:.1f}' for g, v in row['gens'].items()))
        # curves for the figure (mean over generators, by parameter index)
        out[f'{set_}|curves'] = {k: dict(att=np.mean([per[g][k][:, 1].mean(-1) for g in GENS], 0).tolist(),
                                         acc=np.mean([per[g][k][:, 0].mean(-1) for g in GENS], 0).tolist(),
                                         per_gen={g: dict(att=per[g][k][:, 1].mean(-1).tolist(), acc=per[g][k][:, 0].mean(-1).tolist()) for g in GENS})
                                 for k in ('T_sib', 'C_sib', 'T_x', 'C_x', 'count_sib', 'oracle')}
    json.dump(out, open(os.path.join(RES, 'B5_results.json'), 'w'))


if __name__ == '__main__':
    main()
