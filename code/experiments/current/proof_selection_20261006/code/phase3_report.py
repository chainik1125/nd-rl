#!/usr/bin/env python3
"""Phase 3: head-init x pool-generator matrix (within-theorem finished-only AUROC) + primary paired bootstraps.
  python phase3_report.py --build      ensembles/composites (score files only, no labels)
  python phase3_report.py              tables, CIs, figure (reads labels)"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pool_lib as PL
from evaluate_selection import auroc

R = os.path.join(PL.SPRINT, 'results'); F = os.path.join(PL.SPRINT, 'figures')
GENS = ['s0_r16', 's1_r16', 's2_r16']
TEST = ['holdout250', 'textbook72']
ALL_S0 = [('rl_targets', None), ('dev1108', None), ('rr600', None), ('holdout250', None), ('textbook72', None),
          ('rr600', '96b'), ('rl_targets', '48b'), ('dev1108', '48b')]
POOLS = ALL_S0 + [(s, 64, g) for g in GENS[1:] for s in TEST]
ENS = {'verifier3_init_s1': ['verifier3_init_s1_seed0', 'verifier3_init_s1_seed1'],
       'verifier3_init_s2': ['verifier3_init_s2_seed0', 'verifier3_init_s2_seed1'],
       'verifier3_sibinit_ens': ['verifier3_init_s1_seed0', 'verifier3_init_s1_seed1', 'verifier3_init_s2_seed0', 'verifier3_init_s2_seed1'],
       'verifier3_init_s0': ['verifier2_rall_s0', 'verifier2_rall_s1'],                       # seed-matched s0-init (2 runs)
       'verifier2_ens': ['verifier2_rall_s0', 'verifier2_rall_s1', 'verifier2_rall_s2']}
# B's published sibling-pool names: init{s1,s2} for the matrix use the user-facing names verifier3_init{s1,s2}
ALIAS = {'verifier3_inits1': 'verifier3_init_s1', 'verifier3_inits2': 'verifier3_init_s2'}


def spec(p):
    return (p[0], p[1], p[2] if len(p) > 2 else PL.GEN)


def build():
    for p in POOLS:
        s, k, g = spec(p)
        rows = PL.load_pool(s, k, g)
        for name, mem in list(ENS.items()) + [(a, ENS[b]) for a, b in ALIAS.items()]:
            if name == 'verifier2_ens' and g == PL.GEN:
                continue                                           # already written in phase 2 for s0 pools
            M = [PL.load_scores(s, m, k, g) for m in mem]
            sc = {r['name']: np.mean([m[r['name']] for m in M], 0) for r in rows}
            PL.write_jsonl(os.path.join(PL.SCORES, f'{PL.pool_stem(s, k, g)}__{name}.jsonl'),
                           [dict(name=r['name'], scores=[float(x) for x in sc[r['name']]]) for r in rows])
            if name in ('verifier3_sibinit_ens', 'verifier3_init_s1', 'verifier3_init_s2', 'verifier3_init_s0'):
                PL.write_jsonl(os.path.join(PL.SCORES, f'{PL.pool_stem(s, k, g)}__completion_{name}.jsonl'),
                               [dict(name=r['name'], scores=[(1e4 if a['done'] else 0.0) + float(sc[r['name']][a['i']])
                                                             for a in r['attempts']]) for r in rows])
        print('built', s, k, g)


def per_theorem(s, g, scorers):
    pool = PL.load_pool(s, 64, g); lab = PL.load_labels(s, 64, g)
    S = {sc: PL.load_scores(s, sc, 64, g) for sc in scorers}
    out = {}
    for r in pool:
        n = r['name']; d = [a['done'] for a in r['attempts']]
        y = [v for v, f in zip(lab[n], d) if f]
        row = {sc: auroc([v for v, f in zip(S[sc][n], d) if f], y) for sc in scorers}
        if all(v is not None for v in row.values()):
            out[n] = row
    return out


def boot(per, a, b, B=1000, seed=0):
    keys = list(per); d = np.array([per[k][a] - per[k][b] for k in keys])
    bi = np.random.default_rng(seed).integers(0, len(keys), (B, len(keys)))
    bs = d[bi].mean(1)
    return dict(diff=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n=len(keys))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--build', action='store_true'); a = ap.parse_args()
    if a.build:
        return build()
    heads = {'s0': 'verifier3_init_s0', 's1': 'verifier3_init_s1', 's2': 'verifier3_init_s2'}
    judges = {'s0': 'loglik_mean_masked', 's1': 'loglik_mean_masked_s1', 's2': 'loglik_mean_masked_s2'}
    res = {'matrix_heads': {}, 'matrix_judges': {}, 'primary': {}, 'secondary': {}}
    for s in TEST:
        for g in GENS:
            gj = g.split('_')[0]
            jn = dict(judges)
            if gj != 's0':
                jn = {'s0': 'loglik_mean_masked_s0', 's1': 'loglik_mean_masked_s1', 's2': 'loglik_mean_masked_s2'}
            sc = list(heads.values()) + list(jn.values()) + ['verifier2_ens', 'verifier3_sibinit_ens'] + \
                 (['loglik_mean_masked_siblings'] if gj == 's0' else []) + \
                 [f'verifier3_init_s{i}_seed{j}' for i in (1, 2) for j in (0, 1)] + ['verifier2_rall_s0', 'verifier2_rall_s1', 'verifier2_rall_s2']
            per = per_theorem(s, g, sc)
            mean = lambda c: float(np.mean([per[k][c] for k in per]))
            res['matrix_heads'][f'{s}|{gj}'] = {i: mean(c) for i, c in heads.items()}
            res['matrix_judges'][f'{s}|{gj}'] = {i: mean(c) for i, c in jn.items()}
            res['matrix_heads'][f'{s}|{gj}']['n'] = len(per)
            res['matrix_heads'][f'{s}|{gj}']['per_seed'] = {c: mean(c) for c in sc if c.startswith('verifier')}
            if gj == 's0':
                res['primary'][s] = {
                    'P1 sibinit_ens - verifier2_ens (s0 init, 3 seeds)': boot(per, 'verifier3_sibinit_ens', 'verifier2_ens'),
                    'P1b sibinit_ens - s0-init 2-seed mean': boot(per, 'verifier3_sibinit_ens', 'verifier3_init_s0'),
                    'P2 sibinit_ens - loglik_mean_masked_siblings': boot(per, 'verifier3_sibinit_ens', 'loglik_mean_masked_siblings'),
                    'sibinit_ens - loglik_mean_masked (self judge)': boot(per, 'verifier3_sibinit_ens', 'loglik_mean_masked'),
                    'init_s1 - init_s0': boot(per, 'verifier3_init_s1', 'verifier3_init_s0'),
                    'init_s2 - init_s0': boot(per, 'verifier3_init_s2', 'verifier3_init_s0'),
                    'abs': {c: mean(c) for c in ('verifier3_sibinit_ens', 'verifier2_ens', 'verifier3_init_s0', 'loglik_mean_masked_siblings', 'loglik_mean_masked')}}
            else:
                o = [x for x in ('s0', 's1', 's2') if x != gj]
                res['secondary'][f'{s}|{gj}'] = {f'init_{gj} (self) - init_{x}': boot(per, heads[gj], heads[x]) for x in o}
    json.dump(res, open(os.path.join(R, 'phase3_matrix.json'), 'w'), indent=1)
    # figure
    fig, axs = plt.subplots(2, 2, figsize=(10, 8.4))
    for row, (key, ttl) in enumerate((('matrix_heads', 'learned head (init = column)'), ('matrix_judges', 'likelihood judge (judge = column)'))):
        for col, s in enumerate(TEST):
            ax = axs[row, col]
            M = np.array([[res[key][f'{s}|{g.split("_")[0]}'][i] for i in ('s0', 's1', 's2')] for g in GENS])
            im = ax.imshow(M, cmap='Blues', vmin=0.55, vmax=0.9)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f'{M[i, j]:.3f}', ha='center', va='center', fontsize=10, color='white' if M[i, j] > 0.78 else 'black')
                ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec='#e34948', lw=2.5))
            ax.set_xticks(range(3)); ax.set_xticklabels(['s0', 's1', 's2']); ax.set_yticks(range(3)); ax.set_yticklabels(['s0', 's1', 's2'])
            ax.set_xlabel('head initialisation' if row == 0 else 'likelihood judge model'); ax.set_ylabel('generator of the pool')
            n = res['matrix_heads'][f'{s}|s0']['n']
            ax.set_title(f'{s}: {ttl}', fontsize=9)
    fig.colorbar(im, ax=axs, shrink=0.6, label='within-theorem AUROC among finished attempts')
    fig.suptitle('Init x generator: red outline = self (head/judge from the pool\'s own generator)')
    fig.savefig(os.path.join(F, 'A3_init_matrix.png'), dpi=150, bbox_inches='tight'); plt.close(fig)
    print(json.dumps({k: res[k] for k in ('primary', 'secondary')}, indent=1))
    for key in ('matrix_heads', 'matrix_judges'):
        print(key)
        for c, v in res[key].items():
            print(' ', c, {i: round(v[i], 3) for i in ('s0', 's1', 's2')}, v.get('n', ''))


if __name__ == '__main__':
    main()
