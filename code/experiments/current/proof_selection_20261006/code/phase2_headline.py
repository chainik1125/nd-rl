#!/usr/bin/env python3
"""Phase 2 headline: dev-frozen operating points for every family in stopping_phase2.py, applied once to test.

For each family and target m in {2, 4, 8}: the parameter whose combined-dev (dev1108 + rr600 dev split, theorem-
weighted) mean attempts is nearest m.  Fixed-k comparator: the k-mixture with mean exactly m (weights frozen by m).
On test, every policy is reported at its realised accuracy AND realised attempts; paired theorem bootstrap (B=1000)
of accuracy differences against (a) the dev-frozen fixed-k mixture at m, (b) dev-chosen first-finished, (c) dev-chosen
count-stop, with the attempt difference stated; plus the descriptive comparison against fixed-k at the realised cost.

Usage: phase2_headline.py --dev <npz> <npz> --test <npz> ... --out results/B2_headline.json
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stopping import fixed_at

FAMS = ['ff', 'count', 'r1', 'r2', 'r3', 'da', 'db']
TARGETS = (2, 4, 8)
IDX = dict(acc=0, att=1, gen=2, tot=3, disp=4)


def mix(F, m):            # F (16, nT) fixed-k per-theorem acc -> per-theorem acc of the mixture with mean m
    return fixed_at(F, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dev', nargs='+', required=True)
    ap.add_argument('--test', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--B', type=int, default=1000)
    a = ap.parse_args()
    dev = [dict(np.load(p)) for p in a.dev]
    choice = {}
    for f in FAMS:
        att = np.concatenate([d[f][:, 1] for d in dev], -1).mean(-1)
        acc = np.concatenate([d[f][:, 0] for d in dev], -1).mean(-1)
        choice[f] = {}
        for m in TARGETS:
            j = int(np.argmin(np.abs(att - m)))
            choice[f][str(m)] = dict(idx=j, dev_att=float(att[j]), dev_acc=float(acc[j]))
    fx = np.concatenate([d['fixed'][:, 0] for d in dev], -1)
    out = dict(dev=a.dev, choice=choice, dev_fixed={str(m): float(mix(fx, m).mean()) for m in TARGETS}, test={})
    for p in a.test:
        z = dict(np.load(p))
        nT = z['fixed'].shape[-1]
        F = z['fixed'][:, 0]
        rng = np.random.default_rng(0)
        BS = [rng.integers(0, nT, nT) for _ in range(a.B)]
        res = {}
        for m in TARGETS:
            row = {'fixed_k_frozen': dict(att=float(m), acc=float(mix(F, m).mean()),
                                          tot=float(fixed_at(z['fixed'][:, 3], m).mean()))}
            ref = {'fixed': (mix(F, m), np.full(nT, float(m)))}
            for f in FAMS:
                c = choice[f][str(m)]['idx']
                ref[f] = (z[f][c, 0], z[f][c, 1])
            for f in FAMS:
                acc_t, att_t = ref[f]
                c = choice[f][str(m)]['idx']
                r = dict(param_idx=c, att=float(att_t.mean()), acc=float(acc_t.mean()), gen=float(z[f][c, 2].mean()),
                         tot=float(z[f][c, 3].mean()), disp=float(z[f][c, 4].mean()),
                         fixed_at_realised=float(mix(F, att_t.mean()).mean()))
                for lab, (racc, ratt) in (('vs_fixed_frozen', ref['fixed']), ('vs_ff', ref['ff']), ('vs_count', ref['count'])):
                    d = acc_t - racc
                    bs = np.array([d[ix].mean() for ix in BS])
                    r[lab] = dict(diff=float(d.mean()), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)),
                                  att_diff=float(att_t.mean() - ratt.mean()))
                dd = np.array([acc_t[ix].mean() - mix(F[:, ix], att_t[ix].mean()).mean() for ix in BS])
                r['vs_fixed_realised_descriptive'] = dict(diff=float(acc_t.mean() - mix(F, att_t.mean()).mean()),
                                                         lo=float(np.quantile(dd, .025)), hi=float(np.quantile(dd, .975)))
                row[f] = r
            res[str(m)] = row
        out['test'][os.path.basename(p)] = res
        print(os.path.basename(p))
        for m, row in res.items():
            print(f'  m={m}: fixed-k frozen acc {row["fixed_k_frozen"]["acc"]:.3f}')
            for f in FAMS:
                r = row[f]
                print(f'    {f:5s} att {r["att"]:5.2f} acc {r["acc"]:.3f} | vs fixed@{m} {r["vs_fixed_frozen"]["diff"]:+.3f} '
                      f'[{r["vs_fixed_frozen"]["lo"]:+.3f},{r["vs_fixed_frozen"]["hi"]:+.3f}] (Δatt {r["vs_fixed_frozen"]["att_diff"]:+.2f}) '
                      f'| vs FF {r["vs_ff"]["diff"]:+.3f} [{r["vs_ff"]["lo"]:+.3f},{r["vs_ff"]["hi"]:+.3f}] (Δatt {r["vs_ff"]["att_diff"]:+.2f}) '
                      f'| vs count {r["vs_count"]["diff"]:+.3f} [{r["vs_count"]["lo"]:+.3f},{r["vs_count"]["hi"]:+.3f}] (Δatt {r["vs_count"]["att_diff"]:+.2f}) '
                      f'| descr vs fixed@realised {r["vs_fixed_realised_descriptive"]["diff"]:+.3f}')
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
