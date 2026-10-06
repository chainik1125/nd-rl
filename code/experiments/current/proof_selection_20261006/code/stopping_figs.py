#!/usr/bin/env python3
"""Figures for the stopping study (Worker B task 3), from stopping/<prefix>__<scorer>.npz.

(a) accuracy of the returned proof vs mean attempts per theorem;  (b) the same vs mean generated tokens and vs mean
total tokens (prefill + generated + scorer forward);  (c) per-theorem attempts used by the adaptive policies at the
dev-chosen m = 4 operating point vs the theorem's pass rate in the pool.
Fixed-k = dashed, adaptive = solid; colour = selector (blue completion, orange likelihood, aqua verifier, grey random,
black oracle).  Dev-chosen headline operating points (results/stopping_headline.json) are marked with stars.

Usage: stopping_figs.py --prefix ~/runs/proof-selection/stopping/<stem> --title "holdout250, s0_r16 T0.8 k=64"
       --loglik completion_loglik [--verifier verifier_ens] [--headline results/stopping_headline.json] --out figures/x
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from curve_table import hull

C = dict(random='#52514e', completion='#2a78d6', loglik='#eb6834', verifier='#1baf7a', oracle='#0b0b0b')


def load(prefix, s):
    p = f'{prefix}__{s}.npz'
    return {k: v for k, v in np.load(p).items()} if os.path.exists(p) else None


def curves(prefix, loglik, verifier):
    """[(label, colour, linestyle, x_att, x_gen, x_tot, acc)]"""
    out = []
    r = load(prefix, 'random'); c = load(prefix, 'completion'); o = load(prefix, 'oracle')
    K = r['fixed_acc'].shape[0]
    k = np.arange(1, K + 1)
    out.append(('fixed-k, random pick', C['random'], '--', k, r['fixed_gen'].mean(1), r['fixed_tot'].mean(1), r['fixed_acc'].mean(1)))
    out.append(('fixed-k, completion pick', C['completion'], '--', k, c['fixed_gen'].mean(1), c['fixed_tot'].mean(1), c['fixed_acc'].mean(1)))
    out.append(('stop at first finished (N_max swept)', C['completion'], '-', c['nsweep_att'].mean(1), c['nsweep_gen'].mean(1),
                c['nsweep_tot'].mean(1), c['nsweep_acc'].mean(1)))
    for name, col, lab in ((loglik, C['loglik'], loglik), (verifier, C['verifier'], verifier)):
        z = load(prefix, name) if name else None
        if z is None:
            continue
        out.append((f'fixed-k, {lab} pick', col, '--', k, z['fixed_gen'].mean(1), z['fixed_tot'].mean(1), z['fixed_acc'].mean(1)))
        # frontier of the (N_max, tau) family on each cost axis (upper concave envelope; achievable by randomising)
        att = z['adaptive_att'].mean(-1).ravel(); acc = z['adaptive_acc'].mean(-1).ravel()
        gen = z['adaptive_gen'].mean(-1).ravel(); tot = z['adaptive_tot'].mean(-1).ravel()
        fr = [hull(x, acc) for x in (att, gen, tot)]
        out.append((f'adaptive {lab}-stop: frontier over N_max x tau (optimistic, read on this pool)', col, '-', fr[0], fr[1], fr[2], None))
    out.append(('oracle pass@k (fixed-k)', C['oracle'], ':', k, o['fixed_gen'].mean(1), o['fixed_tot'].mean(1), o['fixed_acc'].mean(1)))
    out.append(('oracle: stop at first valid', C['oracle'], '-', o['nsweep_att'].mean(1), o['nsweep_gen'].mean(1),
                o['nsweep_tot'].mean(1), o['nsweep_acc'].mean(1)))
    return out


def style(ax):
    ax.grid(True, color='#e6e5e0', lw=0.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--title', required=True)
    ap.add_argument('--loglik', default='completion_loglik')
    ap.add_argument('--verifier', default=None)
    ap.add_argument('--headline', default=None)
    ap.add_argument('--pool_key', default=None, help='key of this pool in headline test_results')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cv = curves(a.prefix, a.loglik, a.verifier)
    hl = json.load(open(a.headline)) if a.headline else None
    hpts = (hl or {}).get('test_results', {}).get(a.pool_key or os.path.basename(a.prefix), {})
    hmap = {'completion': C['completion'], a.loglik: C['loglik'], a.verifier: C['verifier']}

    # (a) and (b)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), sharey=True)
    for ax, xi, xl in zip(axes, (3, 4, 5), ('mean attempts per theorem', 'mean generated tokens per theorem',
                                            'mean total tokens per theorem (prefill + generated + scorer forward)')):
        for lab, col, ls, *xs in cv:
            if xs[3] is None:                     # frontier: (x, y) per axis
                x, y = xs[xi - 3]
            else:
                x, y = xs[xi - 3], xs[3]
            ax.plot(x, y, ls=ls, color=col, lw=2, label=lab)
        for s, d in hpts.items():
            if s not in hmap or hmap[s] is None:
                continue
            for m, r in d.items():
                xv = {3: r['test_att'], 4: r['test_gen'], 5: r['test_tot']}[xi]
                ax.plot([xv], [r['test_acc']], marker='*', ms=14, color=hmap[s], mec='white', mew=1, ls='none')
        ax.set_xscale('log', base=2 if xi == 3 else 10)
        ax.set_xlabel(xl)
        style(ax)
    axes[0].set_ylabel('accuracy of the returned proof (Lean-valid)')
    axes[0].legend(fontsize=8, loc='lower right', frameon=True, framealpha=0.9)
    fig.suptitle(f'{a.title}: accuracy vs inference cost (stars: dev-chosen operating points, m = 2, 4, 8)', fontsize=12)
    fig.tight_layout()
    fig.savefig(a.out + '_acc_vs_cost.png', dpi=130)
    plt.close(fig)

    # (c) per-theorem allocation at the dev-chosen m=4 point
    if hl:
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        pr = load(a.prefix, 'random')['passrate']
        rng = np.random.default_rng(0)
        for s, col in hmap.items():
            if s is None or s not in hl['choices'] or '4' not in hl['choices'][s]:
                continue
            ch = hl['choices'][s]['4']
            z = load(a.prefix, s)
            if z is None:
                continue
            if s == 'completion':
                att = z['nsweep_att'][ch['N'] - 1]
            else:
                ni = list(z['Ns']).index(ch['N']); g = int(np.where(z['taus'] == ch['tau'])[0][0])
                att = z['adaptive_att'][ni, g]
            ax.scatter(pr + rng.uniform(-0.006, 0.006, len(pr)), att, s=14, color=col, alpha=0.55,
                       label=f'{s}-stop (N_max {ch["N"]}; mean {att.mean():.2f})', edgecolors='none')
            bins = np.array([0, 0.05, 0.25, 0.5, 0.9, 1.0001])
            bi = np.digitize(pr, bins) - 1
            xm = [(bins[i] + min(bins[i + 1], 1)) / 2 for i in range(5) if (bi == i).any()]
            ym = [att[bi == i].mean() for i in range(5) if (bi == i).any()]
            ax.plot(xm, ym, color=col, lw=2)
        ax.set_xlabel('theorem pass rate in the pool (fraction of attempts Lean-valid)')
        ax.set_ylabel('attempts used (mean over orderings)')
        ax.set_yscale('log', base=2)
        style(ax)
        ax.legend(fontsize=8, frameon=False)
        ax.set_title(f'{a.title}\nattempts spent per theorem at the dev-chosen m = 4 point (lines = bucket means)',
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(a.out + '_allocation.png', dpi=130)
        plt.close(fig)


if __name__ == '__main__':
    main()
