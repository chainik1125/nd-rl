#!/usr/bin/env python3
"""Phase 2 figures: figures/B2_xseed_auroc.png and figures/B2_allocation_acc_vs_attempts.png."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results'); FIG = os.path.join(HERE, '..', 'figures')
ST = os.path.expanduser('~/runs/proof-selection/stopping')
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']


def style(ax):
    ax.grid(True, color='#e6e5e0', lw=0.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def xseed_fig():
    sets = [('B2_auroc_dev1108__dev.json', 'dev1108 (dev)'), ('B2_auroc_rr600__dev.json', 'rr600 (dev)'),
            ('B2_auroc_holdout250__s0_r16__k64.json', 'holdout250 (test)'), ('B2_auroc_textbook72__s0_r16__k64.json', 'textbook72 (test)')]
    judges = [('loglik_mean_masked', 's0 = the generator itself'), ('loglik_mean_masked_s1', 's1 (sibling)'),
              ('loglik_mean_masked_s2', 's2 (sibling)'), ('loglik_mean_masked_siblings', 'mean of s1, s2'),
              ('loglik_mean_masked_xseed', 'mean of s0, s1, s2')]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=False)
    for ax, key, title in ((axes[0], 'within_finished', 'within-theorem AUROC, finished attempts (recognition)'),
                           (axes[1], 'pooled_finished', 'pooled AUROC, finished attempts (global threshold)')):
        w = 0.16
        for j, (s, lab) in enumerate(judges):
            ys, lo, hi = [], [], []
            for f, _ in sets:
                m = json.load(open(os.path.join(RES, f)))['scorers'][s]
                ys.append(m[key]); ci = m.get(key + '_ci') or [m[key], m[key]]
                lo.append(m[key] - ci[0]); hi.append(ci[1] - m[key])
            x = np.arange(len(sets)) + (j - 2) * w
            ax.bar(x, ys, w * 0.92, color=C[j], label=lab)
            ax.errorbar(x, ys, yerr=[lo, hi], fmt='none', ecolor='#52514e', lw=1, capsize=2)
        ax.set_xticks(range(len(sets))); ax.set_xticklabels([s[1] for s in sets], fontsize=9)
        ax.set_ylim(0.5, 1.0); ax.set_title(title, fontsize=10); style(ax)
    axes[0].set_ylabel('AUROC for Lean-valid (95% theorem-bootstrap CI)')
    axes[0].legend(fontsize=8, frameon=False, loc='upper left', ncol=3, title='judge (teacher-forced masked mean log p)', title_fontsize=8)
    fig.suptitle('Sibling judges: likelihood of s0_r16 proofs under the three r16 seeds (s0 generated every proof)', fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'B2_xseed_auroc.png'), dpi=130); plt.close(fig)


def alloc_fig():
    H = json.load(open(os.path.join(RES, 'B2_headline.json')))
    H1 = json.load(open(os.path.join(RES, 'stopping_headline.json')))
    fams = [('ff', 'stop at first finished (N_max swept)', C[0], '-'), ('count', 'count-stop: c-th finished (c swept)', C[1], '-'),
            ('r1', 'rank rule R1: quantile of own attempts', C[6], '-'), ('r2', 'rank rule R2: z-score vs own attempts', C[2], '-'),
            ('r3', 'agreement R3: same ND r times, majority pick', C[4], '-'),
            ('da', 'difficulty-aware: verifier on attempt 1', C[5], '--'), ('db', 'difficulty-aware: verifier on attempts 1-2', C[3], '--')]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
    for ax, stem, title in ((axes[0], 'holdout250__s0_r16__k64', 'holdout250 (test, 250 theorems)'),
                            (axes[1], 'textbook72__s0_r16__k64', 'textbook72 (test, 72 theorems)')):
        z = np.load(os.path.join(ST, f'{stem}__phase2.npz'))
        o = np.load(os.path.join(ST, f'{stem}__oracle.npz'))
        k = np.arange(1, 17)
        ax.plot(k, z['fixed'][:, 0].mean(-1), ls='--', color='#52514e', lw=2, label='fixed-k, completion_loglik pick (uniform)')
        for f, lab, col, ls in fams:
            att, acc = z[f][:, 1].mean(-1), z[f][:, 0].mean(-1)
            oo = np.argsort(att)
            ax.plot(att[oo], acc[oo], ls=ls, color=col, lw=2, marker='o', ms=3, label=lab)
        ax.plot(k, o['fixed_acc'][:16].mean(1), ls=':', color='#0b0b0b', lw=2, label='oracle pass@k')
        ax.plot(o['nsweep_att'][:16].mean(1), o['nsweep_acc'][:16].mean(1), color='#0b0b0b', lw=1.5, label='oracle: stop at first valid')
        row = H['test'][f'{stem}__phase2.npz']
        for m in ('2', '4', '8'):
            for f, lab, col, ls in fams:
                r = row[m][f]
                ax.plot([r['att']], [r['acc']], marker='*', ms=12, color=col, mec='white', mew=0.8, ls='none')
        sec = H1['test_results'].get(stem, {})
        for s, mk in (('completion_minstep', 'P'), ('completion_loglik_sum', 'X')):
            for m, r in sec.get(s, {}).items():
                if r['test_att'] <= 16:
                    ax.plot([r['test_att']], [r['test_acc']], marker=mk, ms=8, color='#e34948', ls='none',
                            label=f'{s}-threshold (phase-1 secondary, dev-chosen)' if m == '2' else None)
        ax.set_xscale('log', base=2); ax.set_xlim(0.9, 17)
        ax.set_xlabel('mean attempts per theorem (N_max 16)'); ax.set_title(title, fontsize=10); style(ax)
    axes[0].set_ylabel('accuracy of the returned proof (Lean-valid)')
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, loc='lower center', ncol=4)
    fig.subplots_adjust(bottom=0.27)
    fig.suptitle('Allocation and threshold-free stopping, r16 s0 generator, T 0.8 (stars: dev-chosen points for m = 2, 4, 8)', fontsize=11)
    fig.savefig(os.path.join(FIG, 'B2_allocation_acc_vs_attempts.png'), dpi=130); plt.close(fig)


if __name__ == '__main__':
    xseed_fig(); alloc_fig()
