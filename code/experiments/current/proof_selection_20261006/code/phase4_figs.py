#!/usr/bin/env python3
"""figures/B4_headline.png from results/B4_headline_curves.json + a compact table (stdout, markdown) of the policy point
whose generator-mean attempts is nearest m in {2, 4, 8} (parameter fixed across generators)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
H = json.load(open(os.path.join(HERE, '..', 'results', 'B4_headline_curves.json')))
SPEC = [('random', 'pick one proof at random (fixed k)', '#52514e', '--'),
        ('fixed_completion', 'pick a proof the environment marked finished (fixed k)', '#2a78d6', '--'),
        ('ff', 'stop at the first finished proof', '#2a78d6', '-'),
        ('count_self', 'stop after c finished; the model keeps its own most-likely one', '#eb6834', '-'),
        ('count_sib', 'stop after c finished; two sibling checkpoints judge', '#1baf7a', '-'),
        ('oracle_fixed', 'oracle: any valid proof among k (pass@k)', '#0b0b0b', ':'),
        ('oracle_stop', 'oracle: stop at the first valid proof', '#0b0b0b', '-')]
fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
for ax, set_ in zip(axes, ('holdout250', 'textbook72')):
    S = H[set_]
    G = np.array(S['random']['grid'])
    for key, lab, col, ls in SPEC:
        v = S[key]
        ok = np.array(v['all_defined'])
        m, lo, hi = np.array(v['mean'], float), np.array(v['min'], float), np.array(v['max'], float)
        if key in ('count_self', 'count_sib', 'ff', 'oracle_stop'):
            # also show each generator's own points faintly (their attempt ranges differ)
            for g, pg in v['per_generator'].items():
                ax.plot(pg['att'], pg['acc'], color=col, lw=0.8, alpha=0.35, ls=ls)
        ax.plot(G[ok], m[ok], color=col, ls=ls, lw=2.2, label=lab)
        ax.fill_between(G[ok], lo[ok], hi[ok], color=col, alpha=0.15, lw=0)
    ax.set_xscale('log', base=2); ax.set_xlim(1, 16)
    ax.set_xticks([1, 2, 4, 8, 16]); ax.set_xticklabels(['1', '2', '4', '8', '16'])
    ax.set_xlabel('mean attempts per theorem (at most 16)')
    ax.set_title(f'{set_} (test)', fontsize=11)
    ax.grid(True, color='#e6e5e0'); [ax.spines[s].set_visible(False) for s in ('top', 'right')]
axes[0].set_ylabel('accuracy of the returned proof (Lean-valid)')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=2, fontsize=9, frameon=False)
fig.suptitle('Can the model pick and stop on its own? Mean over 3 generator seeds (r16 s0/s1/s2, T 0.8); band = min-max across seeds;\n'
             'thin lines = each seed (count/stop policies reach different attempt ranges per seed)', fontsize=10.5)
fig.subplots_adjust(bottom=0.25, top=0.86, wspace=0.18)
fig.savefig(os.path.join(HERE, '..', 'figures', 'B4_headline.png'), dpi=130)

# table: parameter fixed across generators, generator-mean attempts nearest m
rows = []
for set_ in ('holdout250', 'textbook72'):
    for key, lab, _, _ in SPEC:
        pg = H[set_][key]['per_generator']
        att = np.mean([pg[g]['att'] for g in pg], 0); acc = np.array([pg[g]['acc'] for g in pg])
        cells = []
        for m in (2, 4, 8):
            i = int(np.argmin(np.abs(att - m)))
            note = '' if abs(att[i] - m) <= 0.2 * m else ' (closest reachable)'
            cells.append(f'{acc[:, i].mean():.3f} [{acc[:, i].min():.3f}-{acc[:, i].max():.3f}] @ {att[i]:.2f}{note}')
        rows.append(f'| {set_} | {lab} | ' + ' | '.join(cells) + ' |')
print('| set | policy | m ≈ 2 | m ≈ 4 | m ≈ 8 |\n|---|---|---|---|---|\n' + '\n'.join(rows))
