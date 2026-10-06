#!/usr/bin/env python3
"""Phase-2 tables + figures: within-theorem finished-only AUROC vs finished-invalid training examples.
  python phase2_report.py --dev_only     (before the freeze: dev numbers only, reads NO test pools)
  python phase2_report.py                (after the freeze + test evaluation)"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pool_lib as PL
from evaluate_selection import auroc

R = os.path.join(PL.SPRINT, 'results'); F = os.path.join(PL.SPRINT, 'figures'); V2 = os.path.join(PL.RUNS, 'verifier2')
RUNGS = ['r25', 'r50', 'r100', 'rall_s0', 'rall_s1', 'rall_s2']
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9})


def within(sets_names, scorer):
    """mean over theorems (with both classes among finished attempts) of per-theorem AUROC among finished attempts."""
    a = []
    for s, names in sets_names:
        pool = {r['name']: r for r in PL.load_pool(s)}
        lab = PL.load_labels(s); sc = PL.load_scores(s, scorer)
        for n in names:
            d = [x['done'] for x in pool[n]['attempts']]
            y = [v for v, f in zip(lab[n], d) if f]; z = [v for v, f in zip(sc[n], d) if f]
            u = auroc(z, y)
            if u is not None:
                a.append(u)
    return float(np.mean(a)), len(a)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dev_only', action='store_true'); a = ap.parse_args()
    sp = json.load(open(os.path.join(R, 'splits.json')))
    devset = [('dev1108', sp['dev']['dev1108']), ('rr600', sp['dev']['rr600'])]
    evals = {'dev (orig pools)': devset}
    if not a.dev_only:
        for s in ('holdout250', 'textbook72'):
            evals[s] = [(s, [r['name'] for r in PL.load_pool(s)])]
    scorers = RUNGS + ['ens', 'll']
    res = {'runs': {}, 'within': {}}
    for n in RUNGS + ['ll']:
        S = json.load(open(os.path.join(V2, n, 'summary.json')))
        res['runs'][n] = {k: S[k] for k in ('n_finished_invalid', 'n_finished_invalid_in_pair_theorems', 'n_pair_theorems',
                                            'best_dev_within', 'best_dev_within_orig', 'best_step', 'train_secs')}
        res['runs'][n]['hist'] = S['hist']
    for ev, sn in evals.items():
        res['within'][ev] = {}
        for sc in ['loglik_mean_masked', 'verifier_ens', 'verifier_hist_ens'] + [f'verifier2_{x}' for x in scorers]:
            try:
                res['within'][ev][sc] = within(sn, sc)
            except FileNotFoundError:
                pass
    tag = '_devonly' if a.dev_only else ''
    json.dump(res, open(os.path.join(R, f'phase2_within{tag}.json'), 'w'), indent=1)
    # table
    evs = list(evals)
    lines = ['| model | finished-invalid train examples (in pair theorems) | pair theorems | best step | ' + ' | '.join(f'within-thm finished AUROC: {e}' for e in evs) + ' |',
             '|---|---|---|---|' + '---|' * len(evs)]
    for n in RUNGS + ['ens', 'll']:
        r = res['runs'].get(n if n != 'ens' else 'rall_s0')
        cnt = f"{r['n_finished_invalid']} ({r['n_finished_invalid_in_pair_theorems']})" if r else ''
        lab = {'ens': 'verifier2_ens (mean of rall s0-s2)', 'll': 'verifier2_ll (rall + loglik feature)'}.get(n, f'verifier2_{n}')
        lines.append(f"| {lab} | {cnt} | {r['n_pair_theorems'] if r else ''} | {r['best_step'] if r and n != 'ens' else ''} | " +
                     ' | '.join(f"{res['within'][e].get('verifier2_' + n, (float('nan'), 0))[0]:.3f}" for e in evs) + ' |')
    for b in ('loglik_mean_masked', 'verifier_ens', 'verifier_hist_ens'):
        lines.append(f'| *{b}* (reference) | | | | ' + ' | '.join(f"{res['within'][e][b][0]:.3f}" for e in evs) + ' |')
    lines.append('\nn theorems with both classes among finished attempts: ' + ', '.join(f"{e} {res['within'][e]['loglik_mean_masked'][1]}" for e in evs))
    open(os.path.join(R, f'phase2_ladder_table{tag}.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    # figure 1: AUROC vs data
    cols = {'dev (orig pools)': '#2a78d6', 'holdout250': '#eb6834', 'textbook72': '#1baf7a'}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for e in evs:
        x = [res['runs'][n]['n_finished_invalid'] for n in ('r25', 'r50', 'r100')]
        y = [res['within'][e]['verifier2_' + n][0] for n in ('r25', 'r50', 'r100')]
        xt = res['runs']['rall_s0']['n_finished_invalid']
        ys = [res['within'][e][f'verifier2_rall_s{i}'][0] for i in range(3)]
        ye = res['within'][e]['verifier2_ens'][0] if 'verifier2_ens' in res['within'][e] else np.mean(ys)
        ax.plot(x + [xt], y + [ye], '-o', color=cols[e], lw=2, ms=6, label=f'verifier2, {e} (top rung = 3-seed ens)')
        ax.scatter([xt] * 3, ys, color=cols[e], marker='x', s=30, zorder=3)
        ax.axhline(res['within'][e]['loglik_mean_masked'][0], color=cols[e], ls='--', lw=1.2)
        ax.text(x[0], res['within'][e]['loglik_mean_masked'][0] + 0.004, f'loglik_mean_masked, {e}', color='#444', fontsize=7)
    ax.set_xscale('log'); ax.set_xlabel('finished-invalid training attempts (log scale)')
    ax.set_ylabel('within-theorem AUROC among finished attempts')
    ax.set_title('Does more hard-negative data close the gap to likelihood?\n(solid: learned ranker; dashed: loglik baseline; x: individual seeds)')
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc='lower right')
    fig.tight_layout(); fig.savefig(os.path.join(F, f'A2_auroc_vs_data{tag}.png'), dpi=150); plt.close(fig)
    # figure 2: top-rung dev curve vs steps (undertraining check)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, c in zip(range(3), ('#2a78d6', '#eb6834', '#1baf7a')):
        h = res['runs'][f'rall_s{i}']['hist']
        ax.plot([x['step'] for x in h], [x['dev_within_orig'] for x in h], '-', color=c, lw=2, label=f'rall s{i}: original dev pools')
        ax.plot([x['step'] for x in h], [x['dev_within'] for x in h], ':', color=c, lw=1.5, label=f'rall s{i}: all dev pools (early-stop metric)')
    ax.axhline(res['within']['dev (orig pools)']['loglik_mean_masked'][0], color='k', ls='--', lw=1, label='loglik_mean_masked (orig dev)')
    ax.set_xlabel('training step (16 theorems per step)'); ax.set_ylabel('dev within-theorem finished AUROC')
    ax.set_title('Top rung: dev within-theorem AUROC over training'); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(F, f'A2_top_rung_dev_curve{tag}.png'), dpi=150); plt.close(fig)


if __name__ == '__main__':
    main()
