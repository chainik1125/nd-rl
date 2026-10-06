#!/usr/bin/env python3
"""Phase 4: score the trained judges with Worker B's scorer, publish judge_* / completion_judge_*, analyse.
  python phase4_report.py --score  [--gpu 5]     runs score_loglik.py --ckpt for every judge x test pool (no labels)
  python phase4_report.py --build                  ensembles + completion composites + sibling means (no labels)
  python phase4_report.py                          tables, paired bootstraps, figure (reads labels)"""
import argparse, json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pool_lib as PL
from evaluate_selection import auroc

R = os.path.join(PL.SPRINT, 'results'); F = os.path.join(PL.SPRINT, 'figures'); J = os.path.join(PL.RUNS, 'judge')
JUDGES = ['sft_s0', 'sft_s1', 'pair_s0', 'pair_s1']
ENS = {'judge_sft': ['judge_sft_s0', 'judge_sft_s1'], 'judge_pair': ['judge_pair_s0', 'judge_pair_s1']}
POOLS = [('holdout250', '64', 's0_r16'), ('textbook72', '64', 's0_r16'), ('holdout250', '64b', 's0_r16'),
         ('textbook72', '64b', 's0_r16'), ('holdout250', '64', 's1_r16'), ('textbook72', '64', 's1_r16'),
         ('holdout250', '64', 's2_r16'), ('textbook72', '64', 's2_r16')]


def judge_file(s, k, g, j):     # B's per-judge masked likelihood file names
    if g == 's0_r16' and k == '64':
        return {'s0': 'loglik_mean_masked', 's1': 'loglik_mean_masked_s1', 's2': 'loglik_mean_masked_s2'}[j]
    return f'loglik_mean_masked_{j}'


def score(gpu):
    py = os.path.expanduser('~/venvs/proof-state/bin/python')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), ND_OFFLINE='1', ND_REGISTRY='0')
    for j in (os.environ.get('JUDGES_ONLY', '').split(',') if os.environ.get('JUDGES_ONLY') else JUDGES):
        ck = os.path.join(J, j, 'best.pt')
        for s, k, g in POOLS:
            pool = PL.pool_path(s, k, g)
            subprocess.run([py, 'score_loglik.py', '--pool', pool, '--ckpt', ck, '--tag', f'judge_{j}'], check=True, env=env,
                           cwd=os.path.dirname(os.path.abspath(__file__)), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            src = os.path.join(PL.SCORES, f'{PL.pool_stem(s, k, g)}__loglik_mean_masked_judge_{j}.jsonl')
            assert os.path.exists(src), src
            print('scored', j, s, k, g, flush=True)


def build():
    for s, k, g in POOLS:
        rows = PL.load_pool(s, k, g)
        def put(name, sc):
            PL.write_jsonl(os.path.join(PL.SCORES, f'{PL.pool_stem(s, k, g)}__{name}.jsonl'),
                           [dict(name=r['name'], scores=[float(x) for x in sc[r['name']]]) for r in rows])
        S = {f'judge_{j}': PL.load_scores(s, f'loglik_mean_masked_judge_{j}', k, g) for j in JUDGES}
        for j in JUDGES:
            put(f'judge_{j}', S[f'judge_{j}'])
        for e, mem in ENS.items():
            S[e] = {r['name']: np.mean([S[m][r['name']] for m in mem], 0) for r in rows}
            put(e, S[e])
        # sibling mean relative to the pool's generator (B's definition: mean of the two other judges' masked mean logp)
        gj = g.split('_')[0]; sib = [x for x in ('s0', 's1', 's2') if x != gj]
        Js = {x: PL.load_scores(s, judge_file(s, k, g, x), k, g) for x in sib}
        S['loglik_sibmean'] = {r['name']: np.mean([Js[x][r['name']] for x in sib], 0) for r in rows}
        put('loglik_sibmean', S['loglik_sibmean'])
        for name in list(ENS) + [f'judge_{j}' for j in JUDGES] + ['loglik_sibmean']:
            fin = lambda v: v if np.isfinite(v) else -1e3
            PL.write_jsonl(os.path.join(PL.SCORES, f'{PL.pool_stem(s, k, g)}__completion_{name}.jsonl'),
                           [dict(name=r['name'], scores=[(1e4 if a['done'] else 0.0) + fin(float(S[name][r['name']][a['i']]))
                                                         for a in r['attempts']]) for r in rows])
        print('built', s, k, g)


def per_theorem(s, k, g, scorers):
    pool = PL.load_pool(s, k, g); lab = PL.load_labels(s, k, g)
    S = {sc: PL.load_scores(s, sc, k, g) for sc in scorers}
    out = {}
    for r in pool:
        n = r['name']; d = [a['done'] for a in r['attempts']]
        y = [v for v, f in zip(lab[n], d) if f]
        row = {sc: auroc([v for v, f in zip(S[sc][n], d) if f], y) for sc in scorers}
        if all(v is not None for v in row.values()):
            out[n] = row
    return out


def boot_mean(per, c, B=1000):
    keys = list(per); v = np.array([per[k][c] for k in keys]); bi = np.random.default_rng(0).integers(0, len(keys), (B, len(keys)))
    bs = v[bi].mean(1); return [float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def boot_diff(per, a, b, B=1000):
    keys = list(per); d = np.array([per[k][a] - per[k][b] for k in keys]); bi = np.random.default_rng(0).integers(0, len(keys), (B, len(keys)))
    bs = d[bi].mean(1); return dict(diff=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n=len(keys))


def analyse():
    res = {}
    for s, k, g in POOLS:
        gj = g.split('_')[0]
        start = judge_file(s, k, g, 's1')
        sc = [start, 'loglik_sibmean', judge_file(s, k, g, gj)] + list(ENS) + [f'judge_{j}' for j in JUDGES]
        sc = list(dict.fromkeys(sc))
        per = per_theorem(s, k, g, sc)
        key = f'{s}|{gj}|k{k}'
        res[key] = dict(n=len(per), start=start, self=judge_file(s, k, g, gj),
                        means={c: boot_mean(per, c) for c in sc},
                        diffs={f'{e} - s1 start': boot_diff(per, e, start) for e in list(ENS) + [f'judge_{j}' for j in JUDGES]})
        for e in ENS:
            res[key]['diffs'][f'{e} - sibling mean'] = boot_diff(per, e, 'loglik_sibmean')
        res[key]['diffs']['judge_pair - judge_sft'] = boot_diff(per, 'judge_pair', 'judge_sft')
    json.dump(res, open(os.path.join(R, 'phase4_within.json'), 'w'), indent=1)
    for k, v in res.items():
        print(k, 'n', v['n'], {c: round(m[0], 3) for c, m in v['means'].items()})
        for d, x in v['diffs'].items():
            if 'judge_s' not in d and 'judge_p' not in d or d.startswith('judge_pair -') or d.startswith('judge_sft -'):
                print('   ', d, '%+.3f [%+.3f, %+.3f]' % (x['diff'], x['ci'][0], x['ci'][1]))
    # figure
    keys = list(res); lab = [k.replace('|s0|k64b', '\ns0, 2nd draw').replace('|s0|k64', '\ns0').replace('|s1|k64', '\ns1-generated').replace('|s2|k64', '\ns2-generated') for k in keys]
    series = [('start', 'plain s1 likelihood (start)', '#8a8a85'), ('judge_sft', 'trained judge: SFT on valid', '#2a78d6'),
              ('judge_pair', 'trained judge: pairwise + KL', '#eb6834'), ('loglik_sibmean', 'sibling-likelihood mean', '#1baf7a')]
    fig, ax = plt.subplots(figsize=(12, 4.8)); x = np.arange(len(keys)); w = 0.2
    for i, (c, l, col) in enumerate(series):
        m = [res[k]['means'][res[k]['start'] if c == 'start' else c] for k in keys]
        ax.bar(x + (i - 1.5) * w, [v[0] for v in m], w * 0.9, color=col, label=l,
               yerr=[[v[0] - v[1] for v in m], [v[2] - v[0] for v in m]], capsize=2, error_kw=dict(elinewidth=0.8, ecolor='#333'))
    ax.set_xticks(x); ax.set_xticklabels([f'{l}\n(n={res[k]["n"]})' for l, k in zip(lab, keys)], fontsize=7.5)
    ax.set_ylabel('within-theorem AUROC among finished attempts'); ax.set_ylim(0.5, 1.0); ax.grid(alpha=0.3, axis='y')
    ax.set_xlabel('test pool (theorem set, generator of the attempts); s1-gen pools are the judge\'s own init generator')
    ax.set_title('Training the sibling likelihood judge on s0\'s train pools (95% theorem-bootstrap bars)')
    fig.legend(*ax.get_legend_handles_labels(), fontsize=8, ncol=4, loc='lower center', frameon=False)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(os.path.join(F, 'A4_trained_judge.png'), dpi=150); plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--score', action='store_true'); ap.add_argument('--build', action='store_true')
    ap.add_argument('--gpu', default='5'); a = ap.parse_args()
    if a.score:
        score(a.gpu)
    elif a.build:
        build()
    else:
        analyse()
