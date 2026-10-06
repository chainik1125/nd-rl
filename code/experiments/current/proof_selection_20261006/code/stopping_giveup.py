#!/usr/bin/env python3
"""POST-HOC (third design iteration on these test sets; pre-registered in logs/worker_B.md before its test read):
give-up + count-stop.  Stop after the c-th finished attempt, OR give up after g attempts with zero finished, N_max 16.
Selection by completion_loglik (s0; --selector siblings for the sibling judges).  Dev choice for target m: among (g, c)
with combined-dev mean attempts within 10 % of m, the highest dev accuracy (else nearest).  Comparators as
phase2_headline.py (dev-frozen fixed-k mixture at m; dev-chosen count-stop and FF from B2_headline*.json).
Usage: stopping_giveup.py [--selector siblings]  -> stopping/<stem>[__dev]__giveup[_sel].npz, results/B2_giveup*.json"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stopping_phase2 as P2
from stopping import perms_for, fixed_at, RUNS

GS = (2, 3, 4, 5, 6, 8, 10, 12, 16)
CS = (1, 2, 3, 4, 6, 8)


def sim(T):
    perms = perms_for(T, 50, 0)
    res = np.zeros((len(GS), len(CS), 2, len(T)))
    for ti, (t, P) in enumerate(zip(T, perms)):
        P = P[:, :16]; K = 16
        S = t['S'][P2.SEL][P]; V = t['v'][P].astype(float); D = t['d'][P]
        M = np.maximum.accumulate(S, 1)
        new = np.ones_like(S, dtype=bool); new[:, 1:] = S[:, 1:] > M[:, :-1]
        A = np.maximum.accumulate(np.where(new, np.arange(K)[None], 0), 1)
        Vs = np.take_along_axis(V, A, 1); nf = np.cumsum(D, 1)
        for gi, g in enumerate(GS):
            for ci, c in enumerate(CS):
                sg = (nf >= c) | ((np.arange(K)[None] >= g - 1) & (nf == 0))
                stop = np.where(sg.any(1), sg.argmax(1), K - 1)
                res[gi, ci, 0, ti] = np.take_along_axis(Vs, stop[:, None], 1).mean()
                res[gi, ci, 1, ti] = (stop + 1).mean()
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--selector', default=None); ap.add_argument('--test', action='store_true')
    a = ap.parse_args()
    if a.selector:
        P2.SEL, P2.LL = f'completion_loglik_{a.selector}', f'loglik_mean_masked_{a.selector}'
    sfx = f'_sel_{a.selector}' if a.selector else ''
    R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
    chf = os.path.join(R, f'B2_giveup_choice{sfx}.json')
    if not a.test:
        dev = [sim(P2.load(s, 'dev')) for s in P2.DEV_POOLS]
        acc = np.concatenate([d[:, :, 0] for d in dev], -1).mean(-1); att = np.concatenate([d[:, :, 1] for d in dev], -1).mean(-1)
        ch = {}
        for m in (2, 4, 8):
            ok = np.abs(att - m) <= 0.1 * m
            if ok.any():
                gi, ci = np.unravel_index(np.argmax(np.where(ok, acc, -1)), acc.shape)
            else:
                gi, ci = np.unravel_index(np.argmin(np.abs(att - m)), att.shape)
            ch[str(m)] = dict(g=GS[gi], c=CS[ci], gi=int(gi), ci=int(ci), dev_att=float(att[gi, ci]), dev_acc=float(acc[gi, ci]))
        json.dump(ch, open(chf, 'w'), indent=1); print(json.dumps(ch)); return
    ch = json.load(open(chf))
    H = json.load(open(os.path.join(R, f'B2_headline{"_siblings" if a.selector else ""}.json')))
    out = {}
    for stem in ('holdout250__s0_r16__k64', 'textbook72__s0_r16__k64'):
        r = sim(P2.load(stem))
        z = np.load(os.path.join(RUNS, 'stopping', f'{stem}{"__sel_" + a.selector if a.selector else ""}__phase2.npz'))
        F = z['fixed'][:, 0]; nT = F.shape[1]; rng = np.random.default_rng(0); BS = [rng.integers(0, nT, nT) for _ in range(1000)]
        for m, c in ch.items():
            acc, att = r[c['gi'], c['ci'], 0], r[c['gi'], c['ci'], 1]
            row = dict(choice=c, att=float(att.mean()), acc=float(acc.mean()))
            cnt = H['choice']['count'][m]['idx']; ff = H['choice']['ff'][m]['idx']
            for lab, (ra, rt) in (('vs_fixed_frozen', (fixed_at(F, float(m)), np.full(nT, float(m)))),
                                  ('vs_count', (z['count'][cnt, 0], z['count'][cnt, 1])), ('vs_ff', (z['ff'][ff, 0], z['ff'][ff, 1]))):
                d = acc - ra; bs = np.array([d[ix].mean() for ix in BS])
                row[lab] = dict(diff=float(d.mean()), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)), att_diff=float(att.mean() - rt.mean()))
            dd = np.array([acc[ix].mean() - fixed_at(F[:, ix], att[ix].mean()).mean() for ix in BS])
            row['vs_fixed_realised_descriptive'] = dict(diff=float(acc.mean() - fixed_at(F, att.mean()).mean()), lo=float(np.quantile(dd, .025)), hi=float(np.quantile(dd, .975)))
            pr = z['passrate']; bi = np.digitize(pr, [0, .05, .25, .5, .9, 1.0001]) - 1
            row['att_by_bucket'] = [float(att[bi == i].mean()) if (bi == i).any() else None for i in range(5)]
            out[f'{stem}|{m}'] = row
            f = lambda k: f'{row[k]["diff"]:+.3f} [{row[k]["lo"]:+.3f},{row[k]["hi"]:+.3f}]'
            print(stem[:10], f'm={m} g={c["g"]} c={c["c"]}: att {row["att"]:.2f} acc {row["acc"]:.3f} | vs fixed@m {f("vs_fixed_frozen")} (Δatt {row["vs_fixed_frozen"]["att_diff"]:+.2f}) '
                  f'| vs count {f("vs_count")} (Δatt {row["vs_count"]["att_diff"]:+.2f}) | vs FF {f("vs_ff")} (Δatt {row["vs_ff"]["att_diff"]:+.2f}) '
                  f'| descr vs fixed@realised {f("vs_fixed_realised_descriptive")} | att by bucket {[round(x, 1) for x in row["att_by_bucket"]]}')
    json.dump(out, open(os.path.join(R, f'B2_giveup_test{sfx}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
