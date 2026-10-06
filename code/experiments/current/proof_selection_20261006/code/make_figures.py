#!/usr/bin/env python3
"""Worker A figures (PNG, matplotlib) from results/selection_<set>.json and verifier summaries."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pool_lib as PL

R = os.path.join(PL.SPRINT, 'results'); F = os.path.join(PL.SPRINT, 'figures')
SHOW = ['random', 'completion', 'completion_loglik', 'shallow_lr', 'verifier_ens', 'verifier_hist_ens']
# reference categorical palette slots 1-5 (validated: dataviz validate_palette.js, light); random = neutral gray
COL = {'random': '#8a8a85', 'completion': '#2a78d6', 'completion_loglik': '#eb6834', 'shallow_lr': '#1baf7a',
       'verifier_ens': '#eda100', 'verifier_hist_ens': '#e87ba4', 'shortest': '#4a3aa7'}
MK = {'random': 'x', 'completion': 'o', 'completion_loglik': 's', 'shallow_lr': '^', 'verifier_ens': 'D',
      'verifier_hist_ens': 'v', 'shortest': 'P'}
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9})


def sel_curves(sets=('holdout250', 'textbook72')):
    fig, axs = plt.subplots(1, len(sets), figsize=(6 * len(sets), 4.5))
    for ax, s in zip(np.atleast_1d(axs), sets):
        r = json.load(open(os.path.join(R, f'selection_{s}.json')))
        sel = r['selection']['all']; ms = sorted(int(m) for m in sel)
        any_sc = r['scorers'][0]
        ax.plot(ms, [sel[str(m)][any_sc]['oracle'] for m in ms], 'k--', label='oracle pass@m')
        for sc in SHOW:
            if sc not in r['scorers']:
                continue
            y = [sel[str(m)][sc]['acc'] for m in ms]
            lo = [sel[str(m)][sc]['ci'][0] for m in ms]; hi = [sel[str(m)][sc]['ci'][1] for m in ms]
            ax.plot(ms, y, marker=MK.get(sc, 'o'), ms=5, lw=2, color=COL.get(sc), label=sc,
                    ls='--' if sc == 'random' else '-')
            ax.fill_between(ms, lo, hi, color=COL.get(sc), alpha=0.07)
        ax.set_xscale('log', base=2); ax.set_xticks(ms); ax.set_xticklabels(ms)
        ax.set_xlabel('candidate pool size m (attempts sampled)'); ax.set_ylabel('accuracy of the selected proof (Lean-valid)')
        ax.set_title(f'{s} (n={r["n_theorems"]} theorems, all incl. unsolvable)')
        ax.grid(alpha=0.3)
    np.atleast_1d(axs)[0].legend(fontsize=8, loc='lower right')
    fig.suptitle('Selected-proof accuracy vs pool size (95% theorem-bootstrap bands)')
    fig.tight_layout(); fig.savefig(os.path.join(F, 'A_selection_vs_m.png'), dpi=150); plt.close(fig)


def bucket_auroc(sets=('holdout250', 'textbook72')):
    fig, axs = plt.subplots(1, len(sets), figsize=(6 * len(sets), 4.5))
    for ax, s in zip(np.atleast_1d(axs), sets):
        r = json.load(open(os.path.join(R, f'selection_{s}.json')))
        bk = list(r['by_bucket']); x = np.arange(len(bk))
        scs = [sc for sc in ['shortest'] + SHOW if sc in r['scorers'] and sc not in ('random', 'completion')]
        w = 0.8 / max(1, len(scs))
        for j, sc in enumerate(scs):
            y = [r['by_bucket'][b]['scorers'][sc]['auroc_finished'] for b in bk]
            ax.bar(x + j * w - 0.4 + w / 2, [np.nan if v is None else v for v in y], w * 0.9, color=COL.get(sc), label=sc)
        ax.axhline(0.5, color='k', lw=0.8, ls=':')
        ax.text(len(bk) - 0.5, 0.505, 'chance', fontsize=7, ha='right', va='bottom')
        ax.set_xticks(x); ax.set_xticklabels([f'{b}\nn={r["by_bucket"][b]["n_theorems"]}' for b in bk], fontsize=7)
        ax.set_xlabel('per-theorem pass rate bucket (exploratory)'); ax.set_ylabel('AUROC(valid) among finished attempts')
        ax.set_ylim(0.2, 1.0); ax.set_title(s + ' (small-n buckets are noisy)'); ax.grid(alpha=0.3, axis='y')
    np.atleast_1d(axs)[0].legend(fontsize=8)
    fig.suptitle('Does the scorer separate valid from invalid FINISHED proofs? (pooled within bucket)')
    fig.tight_layout(); fig.savefig(os.path.join(F, 'A_auroc_finished_by_difficulty.png'), dpi=150); plt.close(fig)


def pooled_vs_within(sets=('dev1108_dev', 'rr600_dev', 'holdout250', 'textbook72')):
    """one message: high pooled AUROC does not mean within-theorem ranking of finished proofs."""
    scs = ['completion_loglik', 'shallow_lr', 'verifier_ens', 'verifier_hist_ens']
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, key, ttl in ((axs[0], 'pooled', 'pooled over all attempts (mixes theorems)'),
                         (axs[1], 'within_theorem_finished_mean', 'within theorem, finished attempts only (what selection needs)')):
        x = np.arange(len(sets)); w = 0.8 / len(scs)
        for j, sc in enumerate(scs):
            y = [json.load(open(os.path.join(R, f'selection_{s}.json')))['auroc'][sc][key] for s in sets]
            ax.bar(x + j * w - 0.4 + w / 2, y, w * 0.9, color=COL[sc], label=sc)
        ax.set_xticks(x); ax.set_xticklabels([s.replace('_dev', ' (dev)').replace('holdout250', 'holdout250 (test)').replace('textbook72', 'textbook72 (test)') for s in sets], fontsize=8)
        ax.set_ylim(0.5, 1.0); ax.set_title(ttl, fontsize=9); ax.grid(alpha=0.3, axis='y')
    axs[0].set_ylabel('AUROC(Lean-valid)'); fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=4, fontsize=8, frameon=False)
    fig.suptitle('The learned head wins on pooled AUROC but loses on within-theorem ranking of finished proofs')
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(os.path.join(F, 'A_pooled_vs_within_auroc.png'), dpi=150); plt.close(fig)


def train_curves():
    fig, ax = plt.subplots(figsize=(6, 4))
    for fmt, ls in (('', '-'), ('hist_', '--')):
        for sd in range(3):
            fn = os.path.join(PL.RUNS, 'verifier', f'{fmt}s{sd}', 'summary.json')
            if not os.path.exists(fn):
                continue
            h = json.load(open(fn))['hist']
            ax.plot([x['step'] for x in h], [x['dev_auroc'] for x in h], ls, label=f'{fmt or "pairs_"}s{sd}')
    ax.set_xlabel('training step'); ax.set_ylabel('dev AUROC (pooled, selector-dev split)')
    ax.set_title('Verifier head: dev AUROC during training'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(F, 'A_verifier_dev_auroc.png'), dpi=150); plt.close(fig)


if __name__ == '__main__':
    for f in (train_curves, sel_curves, bucket_auroc, pooled_vs_within):
        try:
            f(); print('ok', f.__name__)
        except Exception as e:
            print('skip', f.__name__, repr(e))
