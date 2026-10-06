#!/usr/bin/env python3
"""Reviewer's own floor table: spread, pooled sd, MDD, bimodality, 4xN decomposition."""
import json, sys, math, statistics as st
sys.path.insert(0, '/home/dan/review/noise-floor/rv')
from stats import describe, twoway, wilson, mdd_const

R = '/home/dan/review/noise-floor'
H = json.load(open(f'{R}/rv/heldout_recount.json'))
LC = json.load(open(f'{R}/rv/ladder_cov_recount.json'))

POOLS = (1, 2, 3, 4)

def cells(fn, seeds):
    return {(p, s): fn(p, s) for p in POOLS for s in seeds}

QTY = {}
# held-out, 52 cells and 8 cells
for tag, slc, n in (('heldout_overall', 'overall', 5000), ('heldout_len6', 'len6', 1000),
                    ('heldout_depth3', 'depth3', 500), ('heldout_len6_nopat', 'len6_nopat', 247),
                    ('heldout_len5', 'len5', 1000), ('heldout_len4', 'len4', 1000),
                    ('heldout_len3', 'len3', 1000), ('heldout_len2', 'len2', 1000)):
    for seeds, lab in ((range(13), '52'), (range(2), '8'), (range(3), '12')):
        QTY[f'{tag}@{lab}'] = cells(lambda p, s: H[f'p{p}_s{s}']['num'][slc] / H[f'p{p}_s{s}']['den'][slc], seeds)

QTY['ladder_transfer_solved@8'] = cells(lambda p, s: LC['ladder'][f'la_frozen_p{p}_s{s}']['transfer']['solved'], range(2))
QTY['ladder_transfer_lstar@8'] = cells(lambda p, s: LC['ladder'][f'la_frozen_p{p}_s{s}']['transfer']['lstar_need5'], range(2))
QTY['ladder_targets_solved@8'] = cells(lambda p, s: LC['ladder'][f'la_frozen_p{p}_s{s}']['targets']['solved'], range(2))
QTY['cov_red_hit@8'] = cells(lambda p, s: LC['cov_red'][f'p{p}_s{s}']['hit'], range(2))
QTY['cov_req8_hit@8'] = cells(lambda p, s: LC['cov_req8'][f'p{p}_s{s}']['hit'], range(2))

rows = []
for q, cv in QTY.items():
    xs = list(cv.values())
    d = describe(xs)
    seeds = sorted({s for (_, s) in cv})
    tw = twoway(cv, POOLS, seeds)
    d.update({'q': q, 'var_pool': tw['var_pool'], 'var_seed': tw['var_seed'],
              'var_resid': tw['var_resid'], 'MS_pool': tw['MS_pool'], 'MS_seed': tw['MS_seed'],
              'MS_resid': tw['MS_resid'], 'df': tw['df'],
              'pool_means': tw['pool_means']})
    rows.append(d)

print(f"{'quantity':30s} {'n':>3} {'min':>9} {'max':>9} {'max/min':>8} {'sd':>10} {'MDD n=2':>10} {'/mean':>7} {'b':>6} {'b_sas':>6}")
for d in rows:
    print(f"{d['q']:30s} {d['n']:3d} {d['min']:9.4g} {d['max']:9.4g} {d['ratio_maxmin']:8.3f} "
          f"{d['sd']:10.4g} {d['mdd_n2']:10.4g} {d['mdd_n2_frac_mean']:7.3f} {d['bimod_pop']:6.3f} {d['bimod_sas']:6.3f}")

print()
print(f"{'quantity':30s} {'var_pool':>12} {'var_seed':>12} {'var_resid':>12} {'pool%':>8} {'seed%':>8}")
for d in rows:
    tot = max(d['var_pool'], 0) + max(d['var_seed'], 0) + d['var_resid']
    print(f"{d['q']:30s} {d['var_pool']:12.4g} {d['var_seed']:12.4g} {d['var_resid']:12.4g} "
          f"{100*max(d['var_pool'],0)/tot:8.2f} {100*max(d['var_seed'],0)/tot:8.2f}")

# depth-3 high-mode proportion
for lab, seeds in (('52', range(13)), ('12', range(3)), ('8', range(2))):
    xs = [H[f'p{p}_s{s}']['num']['depth3'] / 500 for p in POOLS for s in seeds]
    hi = sum(1 for x in xs if x > 0.44); lo = sum(1 for x in xs if x < 0.11)
    gap = len(xs) - hi - lo
    w = wilson(hi, len(xs))
    print(f'depth3 n={len(xs)}: high(>0.44) {hi} low(<0.11) {lo} gap {gap} '
          f'prop {hi/len(xs):.4f} Wilson [{w[0]:.3f},{w[1]:.3f}] halfwidth {(w[1]-w[0])/2:.3f}')
for p in POOLS:
    xs = [H[f'p{p}_s{s}']['num']['depth3'] / 500 for s in range(13)]
    hi = sum(1 for x in xs if x > 0.44)
    print(f'  pool p{p} high-mode {hi}/13 = {hi/13:.3f}')

json.dump([{k: v for k, v in d.items() if k != 'pool_means'} for d in rows],
          open(f'{R}/rv/floor_table.json', 'w'), indent=1)
