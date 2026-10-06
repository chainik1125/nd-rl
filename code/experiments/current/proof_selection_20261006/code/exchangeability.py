#!/usr/bin/env python3
"""Exchangeability checks for random-order averaging (reviewer item 12): validity, finished rate and steps by SN name
base (0..32) and by attempt index (blocks of 8), with theorem fixed effects removed (attempt validity minus the
theorem's pass rate), plus duplicate action streams within a theorem.  The original-order vs shuffled stopping curves
come from `stopping.py --orig_order` (compared in the report).

Usage: exchangeability.py --pool <pool.jsonl> --out results/exch_<stem>.json [--fig figures/exch_<stem>.png]
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stopping import load_pool, RUNS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fig', default=None)
    a = ap.parse_args()
    stem, T, _ = load_pool(a.pool, os.path.join(RUNS, 'scores'), None, None, ['__none__'])
    rows = [json.loads(l) for l in open(a.pool)]
    dup = sum(len(r['attempts']) - len({json.dumps(x['actions']) for x in r['attempts']}) for r in rows)
    resid_v, resid_s, base, idx, dn = [], [], [], [], []
    for t in T:
        resid_v.append(t['v'] - t['v'].mean()); resid_s.append(t['steps'] - t['steps'].mean())
        base.append(t['base']); idx.append(np.arange(t['k'])); dn.append(t['d'] - t['d'].mean())
    rv, rs, b, ix, rd = map(np.concatenate, (resid_v, resid_s, base, idx, dn))
    out = dict(pool=stem, n_attempts=int(len(rv)), duplicate_action_streams=int(dup), by_base={}, by_index_block={})
    for lo in range(0, 33, 4):
        m = (b >= lo) & (b < lo + 4)
        if m.any():
            se = rv[m].std() / np.sqrt(m.sum())
            out['by_base'][f'{lo}-{lo + 3}'] = dict(n=int(m.sum()), valid_resid=float(rv[m].mean()), se=float(se),
                                                    finished_resid=float(rd[m].mean()), steps_resid=float(rs[m].mean()))
    K = int(ix.max()) + 1
    for lo in range(0, K, 8):
        m = (ix >= lo) & (ix < lo + 8)
        se = rv[m].std() / np.sqrt(m.sum())
        out['by_index_block'][f'{lo}-{lo + 7}'] = dict(n=int(m.sum()), valid_resid=float(rv[m].mean()), se=float(se),
                                                       finished_resid=float(rd[m].mean()), steps_resid=float(rs[m].mean()))
    # simple linear trend tests (slope of validity residual on base / index, with z score)
    for nm, x in (('base', b), ('index', ix)):
        xc = x - x.mean()
        slope = (xc * rv).sum() / (xc ** 2).sum()
        se = np.sqrt(((rv - slope * xc) ** 2).sum() / (len(rv) - 2) / (xc ** 2).sum())
        out[f'valid_slope_per_{nm}'] = dict(slope=float(slope), z=float(slope / se))
    json.dump(out, open(a.out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith('by_')}))
    if a.fig:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, key, xl in zip(axes, ('by_base', 'by_index_block'), ('SN name base (bins of 4)', 'attempt index (blocks of 8)')):
            d = out[key]
            xs = list(d.keys())
            y = [d[x]['valid_resid'] for x in xs]; e = [1.96 * d[x]['se'] for x in xs]
            ax.errorbar(range(len(xs)), y, yerr=e, fmt='o', color='#2a78d6', capsize=3)
            ax.axhline(0, color='#52514e', lw=1)
            ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, rotation=45, fontsize=8)
            ax.set_xlabel(xl); ax.grid(True, color='#e6e5e0')
        axes[0].set_ylabel('validity minus theorem pass rate (95% CI)')
        fig.suptitle(f'{stem}: is validity exchangeable across attempts? (should sit on 0)')
        fig.tight_layout(); fig.savefig(a.fig, dpi=120)


if __name__ == '__main__':
    main()
