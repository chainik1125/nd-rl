#!/usr/bin/env python3
"""Markdown tables for results/worker_A_report.md from results/selection_*.json (+ clipping analysis)."""
import json, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pool_lib as PL
from evaluate_selection import auroc

R = os.path.join(PL.SPRINT, 'results')
MAIN = ['random', 'first', 'completion', 'shortest', 'loglik_mean', 'entropy_mean', 'completion_loglik',
        'completion_loglik_sum', 'completion_entropy', 'shallow_lr_env', 'shallow_lr', 'verifier_s0', 'verifier_s1',
        'verifier_s2', 'verifier_ens', 'verifier_hist_s0', 'verifier_hist_s1', 'verifier_hist_s2', 'verifier_hist_ens',
        'completion_verifier_ens', 'completion_verifier_hist_ens', 'verifier_hist_rank_s0']
f3 = lambda x: '–' if x is None else f'{x:.3f}'
out = []


def sel_table(fn, title, sub='all', ms=('8', '64')):
    r = json.load(open(os.path.join(R, fn)))
    ms = [m for m in ms if m in r['selection'][sub]]
    out.append(f"\n#### {title} — {sub} (n={r['n_theorems'] if sub == 'all' else r['n_solvable']} theorems)\n")
    out.append('| scorer | ' + ' | '.join(f'acc@{m} [95% CI] | gap@{m}' for m in ms) + ' | AUROC pooled | AUROC finished | within-thm AUROC | within-thm finished |')
    out.append('|---|' + '---|---|' * len(ms) + '---|---|---|---|')
    for sc in [s for s in MAIN if s in r['scorers']]:
        c = []
        for m in ms:
            d = r['selection'][sub][m][sc]
            c += [f"{d['acc']:.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]", f3(d['gap_recovered'])]
        a = r['auroc'][sc]
        out.append(f'| {sc} | ' + ' | '.join(c) + f" | {f3(a['pooled'])} | {f3(a['finished_only'])} | {f3(a['within_theorem_mean'])} | {f3(a['within_theorem_finished_mean'])} |")
    d0 = r['selection'][sub]
    out.append('| *oracle pass@m* | ' + ' | '.join(f"{d0[m]['random']['oracle']:.3f} | 1" for m in ms) + ' | | | | |')
    out.append('| *random pick (expectation)* | ' + ' | '.join(f"{d0[m]['random']['random']:.3f} | 0" for m in ms) + ' | | | | |')
    return r


def paired(r, title, m='8'):
    P = r['paired']['all'][m]
    out.append(f'\n#### Paired differences at k={m}, {title} (theorem bootstrap, 95% CI)\n')
    out.append('| comparison | diff | 95% CI |'); out.append('|---|---|---|')
    want = [('completion_loglik', 'verifier_ens'), ('shallow_lr', 'verifier_ens'), ('completion_loglik', 'shallow_lr'),
            ('completion_loglik', 'verifier_s0'), ('completion_loglik', 'verifier_s1'), ('completion_loglik', 'verifier_s2'),
            ('completion_loglik', 'verifier_hist_ens'), ('verifier_ens', 'completion'), ('verifier_ens', 'shortest'),
            ('verifier_hist_ens', 'verifier_ens'), ('completion_loglik', 'completion')]
    for a, b in want:
        k1, k2 = f'{a} - {b}', f'{b} - {a}'
        if k1 in P:
            d = P[k1]; out.append(f"| {a} − {b} | {d['diff']:+.4f} | [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}] |")
        elif k2 in P:
            d = P[k2]; out.append(f"| {a} − {b} | {-d['diff']:+.4f} | [{-d['ci'][1]:+.4f}, {-d['ci'][0]:+.4f}] |")


def buckets(r, title):
    out.append(f"\n#### Exploratory: by per-theorem pass-rate bucket, {title} (acc at m={list(r['by_bucket'].values())[0]['m']}; AUROC among finished)\n")
    scs = ['completion', 'shortest', 'completion_loglik', 'shallow_lr', 'verifier_ens', 'verifier_hist_ens']
    out.append('| bucket | n thm | P(valid\\|fin) | random | oracle | ' + ' | '.join(f'{s} acc / AUROC-fin' for s in scs) + ' |')
    out.append('|---|---|---|---|---|' + '---|' * len(scs))
    for b, d in r['by_bucket'].items():
        sc0 = d['scorers']['completion']
        out.append(f"| {b} | {d['n_theorems']} | {f3(sc0['p_valid_given_finished'])} | {d['random']:.3f} | {d['oracle']:.3f} | " +
                   ' | '.join(f"{d['scorers'][s]['acc']:.3f} / {f3(d['scorers'][s]['auroc_finished'])}" for s in scs) + ' |')


def clipping():
    out.append('\n#### Pairs-format clipping vs discrimination (verifier_ens, AUROC among finished attempts)\n')
    out.append('| pool | finished attempts | clipped share | AUROC fin, clipped | AUROC fin, not clipped | verifier_hist_ens AUROC fin (same split: clipped / not) |')
    out.append('|---|---|---|---|---|---|')
    for s in ('holdout250', 'textbook72'):
        d = pickle.load(open(os.path.join(PL.RUNS, 'verifier', 'seqs', PL.pool_stem(s) + '.pkl'), 'rb'))
        lab = PL.load_labels(s); pool = PL.load_pool(s)
        v = PL.load_scores(s, 'verifier_ens'); h = PL.load_scores(s, 'verifier_hist_ens')
        X = []
        for r in pool:
            for a in r['attempts']:
                if a['done']:
                    X.append((d[r['name']][1][a['i']], lab[r['name']][a['i']], v[r['name']][a['i']], h[r['name']][a['i']]))
        cl = np.array([x[0] for x in X]); y = np.array([x[1] for x in X]); sv = np.array([x[2] for x in X]); sh = np.array([x[3] for x in X])
        out.append(f'| {s} | {len(X)} | {cl.mean():.3f} | {f3(auroc(sv[cl], y[cl]))} | {f3(auroc(sv[~cl], y[~cl]))} | '
                   f'{f3(auroc(sh[cl], y[cl]))} / {f3(auroc(sh[~cl], y[~cl]))} |')


def verifier_runs():
    out.append('\n#### Verifier training runs (dev early stopping)\n')
    out.append('| run | criterion | best dev value | best step | steps run | train s |'); out.append('|---|---|---|---|---|---|')
    for d in ('s0', 's1', 's2', 'hist_s0', 'hist_s1', 'hist_s2', 'hist_rank_s0'):
        fn = os.path.join(PL.RUNS, 'verifier', d, 'summary.json')
        if os.path.exists(fn):
            S = json.load(open(fn))
            crit = 'dev within-thm finished AUROC' if 'rank' in d else 'dev pooled AUROC'
            out.append(f"| {d} | {crit} | {S['best_dev_auroc']:.4f} | {S['best_step']} | {S['steps']} | {S['train_secs']:.0f} |")


h = sel_table('selection_holdout250.json', 'holdout250 (TEST, k=64 pool)')
t = sel_table('selection_textbook72.json', 'textbook72 (TEST, k=64 pool)')
sel_table('selection_textbook72.json', 'textbook72 (TEST, k=64 pool)', sub='solvable')
paired(h, 'holdout250'); paired(t, 'textbook72')
paired(h, 'holdout250', m='64'); paired(t, 'textbook72', m='64')
buckets(h, 'holdout250'); buckets(t, 'textbook72')
clipping()
verifier_runs()
sel_table('selection_dev1108_dev.json', 'dev1108 (selector-dev split, k=16 pool)', ms=('8', '16'))
sel_table('selection_rr600_dev.json', 'rr600 (selector-dev split, k=32 pool)', ms=('8', '32'))
open(os.path.join(R, 'worker_A_tables.md'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
