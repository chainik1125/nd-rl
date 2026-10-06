#!/usr/bin/env python3
"""Emit the quoteable table of NOISE_FLOOR.md from artifacts/nf/summary.json (nf_analysis.py).
The prose around it is written by hand; this keeps every number in it derived from the pulled files.
  python3 nf_floor_md.py"""
import json

S = json.load(open('artifacts/nf/summary.json'))
F = S['floors']

# quantity key -> (name as a future run would say it, unit, how to print a value, published control value)
Q = [
    ('ladder_frozen_transfer_solved', 'frozen ladder, transfer theorems solved (of 2,285)', 'count', 0),
    ('ladder_frozen_transfer_lstar', 'frozen ladder, transfer `L*`', 'count', 0),
    ('ladder_frozen_targets_solved', 'frozen ladder, RL targets solved (of 4,495)', 'count', 0),
    ('cov_red_solved', '`targets_reductio_req` solved at pass@2,000 (of 300)', 'count', 0),
    ('cov_req8_solved', '`r3_1/depth3_req` solved at pass@2,000 (of 300)', 'count', 0),
    ('heldout_overall', 'held-out greedy, overall (5,000)', 'rate', 4),
    ('heldout_len6', 'held-out greedy, 6-line bin (1,000)', 'rate', 4),
    ('heldout_depth3_slice', 'held-out greedy, depth-3 slice (500)', 'rate', 4),
    ('heldout_len6_nopattern', 'held-out greedy, 6-line no-pattern (247)', 'rate', 4),
    ('heldout_len5', 'held-out greedy, 5-line bin (1,000)', 'rate', 4),
    ('heldout_len4', 'held-out greedy, 4-line bin (1,000)', 'rate', 4),
    ('heldout_len3', 'held-out greedy, 3-line bin (1,000)', 'rate', 4),
    ('heldout_len2', 'held-out greedy, 2-line bin (1,000)', 'rate', 4),
]

print('| quantity | cells | observed range over the null cells | pooled sd | '
      '**smallest difference worth reporting at n = 2** | as a ratio |\n|---|---|---|---|---|---|')
for k, name, kind, dp in Q:
    f = F.get(k)
    if not f:
        continue
    fmt = (lambda x: f'{x:.{dp}f}') if kind == 'rate' else (lambda x: f'{x:g}')
    ratio = f'{1 + f["mdd_n2_frac_of_mean"]:.2f}×' if f['mdd_n2_frac_of_mean'] else '—'
    mdd = fmt(f['mdd_n2_abs']) if kind == 'rate' else f'{f["mdd_n2_abs"]:.0f}'
    if kind == 'rate':
        mdd = f'**{100 * f["mdd_n2_abs"]:.1f} pp**' if f['mdd_n2_abs'] < 1 else '**not resolvable**'
    else:
        mdd = f'**{f["mdd_n2_abs"]:.0f}** ({100 * f["mdd_n2_frac_of_mean"]:.0f} % of the mean)'
    print(f'| {name} | {f["n_cells"]} | {fmt(f["min"])} – {fmt(f["max"])} ({f["max_over_min"]:.2f}×) | '
          f'{f["sd"]:.4g} | {mdd} | {ratio} |')

print('\n<!-- bimodality: b = (skew^2 + 1)/kurtosis, > 5/9 is bimodal-consistent -->')
print('| quantity | bimodality b | high-mode proportion (Wilson 95 %) |\n|---|---|---|')
for k, name, _, _ in Q:
    f = F.get(k)
    if not f or f['bimodality_coefficient'] is None:
        continue
    hm = f.get('high_mode')
    hs = (f'{hm["n_high"]}/{hm["n_high"] + hm["n_low"] + hm["n_between"]} = {hm["p_high"]:.3f} '
          f'[{hm["wilson95"][0]:.3f}, {hm["wilson95"][1]:.3f}]') if hm else '—'
    print(f'| {name} | {f["bimodality_coefficient"]:.3f}'
          + (' **bimodal**' if f['bimodal_consistent'] else '') + f' | {hs} |')
