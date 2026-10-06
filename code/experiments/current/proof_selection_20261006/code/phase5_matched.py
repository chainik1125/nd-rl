#!/usr/bin/env python3
"""Phase 5 descriptive add-on (logged after the frozen test read): each dev-chosen threshold policy vs count-stop
(same judge) at the policy's REALISED attempts: the count-stop curve over c = 1..8 is linearly mixed between adjacent c
(mean attempts per generator) to match; paired theorem bootstrap pooled over generators.  Also the oracle stop-at-
first-valid at the same attempts and the fraction of the (count-stop -> oracle) gap closed."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase5 as P5
fg = json.load(open(os.path.join(P5.RUNS, 'stopping', 'phase5_fit_grids.json'))); ch = json.load(open(os.path.join(P5.RES, 'B5_dev_choice.json')))
out = {}
for set_ in ('holdout250', 'textbook72'):
    per = {g: P5.simulate(P5.load(f'{set_}__{g}_r16__k64', g), fg) for g in P5.GENS}
    nT = per['s0']['oracle'].shape[-1]; rng = np.random.default_rng(0); BS = [rng.integers(0, nT, nT) for _ in range(2000)]
    for j in ('sib', 'x'):
        for pol in (f'T_{j}', f'C_{j}'):
            for m in ('3', '4', '6', '8'):
                d = np.zeros(nT); acc_p = []; acc_c = []; acc_o = []; atts = []
                for g in P5.GENS:
                    a = per[g][pol][ch[pol][m]['idx']]; C = per[g][f'count_{j}']; O = per[g]['oracle']
                    att = a[1].mean(); catt = C[:, 1].mean(-1)
                    if att <= catt[0]:
                        cm = C[0, 0]
                    elif att >= catt[-1]:
                        cm = C[-1, 0]
                    else:
                        i = int(np.searchsorted(catt, att)) - 1; w = (att - catt[i]) / (catt[i + 1] - catt[i])
                        cm = (1 - w) * C[i, 0] + w * C[i + 1, 0]
                    d += (a[0] - cm) / 3
                    oatt = O[:, 1].mean(-1); oacc = O[:, 0].mean(-1)
                    acc_p.append(a[0].mean()); acc_c.append(cm.mean()); atts.append(att)
                    acc_o.append(float(np.interp(att, oatt, oacc)) if att <= oatt.max() else float(oacc.max()))
                bs = np.array([d[ix].mean() for ix in BS])
                gap = (np.mean(acc_p) - np.mean(acc_c)) / (np.mean(acc_o) - np.mean(acc_c))
                r = dict(att=float(np.mean(atts)), acc=float(np.mean(acc_p)), count_matched=float(np.mean(acc_c)),
                         oracle_stop=float(np.mean(acc_o)), diff=float(d.mean()), lo=float(np.quantile(bs, .025)),
                         hi=float(np.quantile(bs, .975)), gap_closed=float(gap), count_in_range=bool(np.all([a <= per[g][f'count_{j}'][:, 1].mean(-1)[-1] for a, g in zip(atts, P5.GENS)])))
                out[f'{set_}|{pol}|m{m}'] = r
                print(f'{set_[:10]} {pol:5s} m={m}: att {r["att"]:5.2f} acc {r["acc"]:.3f} | count@same att {r["count_matched"]:.3f} | oracle-stop {r["oracle_stop"]:.3f} | diff {r["diff"]:+.4f} [{r["lo"]:+.4f},{r["hi"]:+.4f}] | gap closed {r["gap_closed"]:+.2f}{"" if r["count_in_range"] else " (count beyond its c=8 range: compared with c=8)"}')
json.dump(out, open(os.path.join(P5.RES, 'B5_matched.json'), 'w'), indent=1)
