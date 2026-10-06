#!/usr/bin/env python3
"""Run 5 summary: acquisition on the REQUIRED pools (strict pattern predicate on the model's normalised proofs, min-round
rule via phase2_metrics.arm_metrics), oracle violations (targets solved with no pattern proof), frozen controls, base
pass@1e4 coverage, next to the campaign's numbers; figures run5_dials.png and run5_curves.png.
  python run5_analysis.py --out artifacts/r5/summary.json --figs figures
"""
import argparse, json, os, sys, glob, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_metrics import arm_metrics
from normalize import norm
from patterns import classify

CAMPAIGN = {   # the campaign's numbers on the non-required pools (numbers.md, blocks B and C)
    'reductio': {'pool': 'targets_reductio2 (606, DN needed, ≤10-line proof not guaranteed)',
                 'ei': {(0, 0): 58 / 606, (0, 1): 0.0, (0, 2): 0.0, (0.1, 0): 95 / 606, (0.1, 1): 63 / 606},
                 'frozen': {(0, 0): 3 / 606, (0, 1): 0.0, (0, 2): 0.0, (0.1, 0): 25 / 606, (0.1, 1): 29 / 606},
                 'solved': {(0, 0): 58 / 606, (0, 1): 0.0, (0, 2): 0.0, (0.1, 0): 95 / 606, (0.1, 1): 63 / 606}},
    'derived_ore_strict': {'pool': 'targets_c8 (500, generating proof uses the strict ORE, no ≤8-line proof)',
                 'ei': {(0, 0): 4 / 500, (0, 1): 4 / 500, (0.001, 0): 11 / 500, (0.01, 0): 11 / 500},
                 'frozen': {(0, 0): 3 / 500, (0, 1): 2 / 500},
                 'solved': {(0, 0): 197 / 500, (0, 1): 200 / 500, (0.001, 0): 202 / 500, (0.01, 0): 218 / 500}}}
ARMS = {'reductio': [('ei', 0, 0), ('ei', 0, 1), ('ei', 0, 2), ('ei', 0.1, 0), ('ei', 0.1, 1), ('frozen', 0, 0), ('frozen', 0, 1), ('frozen', 0, 2), ('frozen', 0.1, 0), ('frozen', 0.1, 1)],
        'derived_ore_strict': [('ei', 0, 0), ('ei', 0, 1), ('ei', 0.01, 0), ('ei', 0.01, 1), ('frozen', 0, 0), ('frozen', 0, 1), ('frozen', 0.01, 0), ('frozen', 0.01, 1)]}
DIR = {'reductio': lambda k, f, s: f'artifacts/r5/{k}_reductio_f{f:g}_s{s}_req', 'derived_ore_strict': lambda k, f, s: f'artifacts/r5/{k}_derived_ore_strict_f{f:g}_c8_s{s}_req'}
COV = {'reductio': lambda s: f'artifacts/r5/cov_reductio_f0_s{s}_req.s0.jsonl', 'derived_ore_strict': lambda s: f'artifacts/r5/cov_derived_ore_strict_f0_c8_s{s}_req.s0.jsonl'}
TARGETS = {'reductio': 'data/p2/targets_reductio_req.jsonl', 'derived_ore_strict': 'data/p2/targets_derived_ore_req.jsonl'}


def violations(arm, pattern, r):
    """targets solved by round r with NO pattern-containing proof (the oracle's empirical error), with one proof each."""
    fn = f'{arm}/found_{r}.jsonl'
    if not os.path.exists(fn):
        fn = sorted(glob.glob(f'{arm}/found_*.jsonl'), key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
    by = collections.defaultdict(list)
    for l in open(fn):
        x = json.loads(l)
        if x['round'] <= r:
            by[x['name']].append(x)
    out = []
    for name, xs in by.items():
        cls = [classify(norm(x['proof'])) for x in xs]
        if not any(c and c[pattern] for c in cls):
            out.append({'name': name, 'thm': xs[0]['thm'], 'proof': norm(xs[0]['proof']), 'n_proofs': len(xs)})
    return out


def coverage(fn):
    if not os.path.exists(fn):
        return None
    rs = [json.loads(l) for l in open(fn)]
    pat = 'reductio' if 'reductio' in fn else 'derived_ore_strict'
    return {'targets': len(rs), 'samples': sum(r['n_tried'] for r in rs), 'solved': sum(r['n_ok'] > 0 for r in rs), 'n_ok': sum(r['n_ok'] for r in rs),
            'solved_within_256': sum(r['first_hit'] is not None and r['first_hit'] <= 256 for r in rs),
            'pattern_hits': sum(r['hits_by_pattern'].get(pat, 0) for r in rs), 'targets_with_pattern_proof': sum(r['distinct_by_pattern'].get(pat, 0) > 0 for r in rs),
            'solved_without_pattern': sum(r['n_ok'] > 0 and r['distinct_by_pattern'].get(pat, 0) == 0 for r in rs),
            'rate_per_sample': sum(r['hits_by_pattern'].get(pat, 0) for r in rs) / max(1, sum(r['n_tried'] for r in rs))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='artifacts/r5/summary.json'); ap.add_argument('--figs', default='figures')
    a = ap.parse_args()
    res = {}
    for pattern, arms in ARMS.items():
        P = {'targets_file': TARGETS[pattern], 'arms': {}, 'coverage': {}, 'campaign': {k: ({str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in CAMPAIGN[pattern].items()}}
        if os.path.exists(TARGETS[pattern]):
            T = [json.loads(l) for l in open(TARGETS[pattern])]
            P['n_targets'] = len(T); P['targets_min_lines_hist'] = dict(sorted(collections.Counter(t['n_lines'] for t in T).items()))
            P['targets_by_schema'] = dict(sorted(collections.Counter(str(t.get('schema')) for t in T).items()))
        for kind, f, s in arms:
            d = DIR[pattern](kind, f, s)
            if not glob.glob(f'{d}/round_*.json'):
                continue
            m = arm_metrics(d, pattern); last = m[-1]
            v = violations(d, pattern, last['round'])
            P['arms'][f'{kind}_f{f:g}_s{s}'] = {
                'kind': kind, 'f': f, 'seed': s, 'round': last['round'], 'solved': last['targets_solved'], 'n': last['targets_n'],
                'acq_theorems': last['acq_targets_theorems'], 'acq': last['acq_targets'], 'pattern_proofs': last['n_pattern_proofs_targets'],
                'first_round': last['first_round_pattern_targets'], 'per_round_solved': [r['targets_solved'] for r in m],
                'per_round_acq': [r['acq_targets_theorems'] for r in m], 'violations': len(v), 'violation_examples': v[:5],
                'transfer_solved': last['transfer_solved'], 'transfer_n': last['transfer_n'], 'transfer_acq_theorems': last['acq_transfer_theorems'],
                'transfer_violations': None, 'heldout_greedy': last['heldout_greedy'], 'transfer_greedy': last['transfer_greedy'],
                'written_hist': last['written_hist_targets'], 'other_patterns': last['other_patterns_targets'], 'examples': last['pattern_examples_targets'][:5],
                'solved_by_schema': dict(collections.Counter(t.get('schema') for t in T if t['name'] in set(last['solved_names_targets']))) if os.path.exists(TARGETS[pattern]) else None}
        for s in (0, 1, 2):
            c = coverage(COV[pattern](s))
            if c:
                P['coverage'][f'f0_s{s}'] = c
        res[pattern] = P
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    for pattern, P in res.items():
        print(f"== {pattern}: {P.get('n_targets')} required targets")
        for k, v in P['arms'].items():
            print(f"  {k:16s} solved {v['solved']:3d}/{v['n']} acq {v['acq_theorems']:3d} ({v['acq']:.3f}) proofs {v['pattern_proofs']:4d} first {str(v['first_round']):>4s} viol {v['violations']:2d} per-round {v['per_round_acq']} transfer {v['transfer_solved']}/{v['transfer_acq_theorems']}")
        for k, v in P['coverage'].items():
            print(f"  cov {k}: {v}")
    # figures
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    F0 = {'reductio': 1e-3, 'derived_ore_strict': 1e-4}
    for ax, pattern in zip(axes, res):
        P = res[pattern]; f0 = F0[pattern]
        px = lambda f: f0 if f == 0 else f
        for k, v in P['arms'].items():
            ax.scatter([px(v['f'])], [v['acq']], s=60 if v['kind'] == 'ei' else 45, facecolors='#2a78d6' if v['kind'] == 'ei' else 'none', edgecolors='white' if v['kind'] == 'ei' else '#2a78d6', linewidths=1.5, zorder=4)
        for key, col in (('ei', '#b4b2ad'), ('frozen', '#d9d7d2')):
            for kk, val in CAMPAIGN[pattern][key].items():
                ax.scatter([px(kk[0])], [val], marker='s', s=40, facecolors=col if key == 'ei' else 'none', edgecolors=col, zorder=3)
        xs = sorted({px(v['f']) for v in P['arms'].values() if v['kind'] == 'ei'})
        if xs:
            ys = [sum(v['acq'] for v in P['arms'].values() if v['kind'] == 'ei' and px(v['f']) == x) / sum(1 for v in P['arms'].values() if v['kind'] == 'ei' and px(v['f']) == x) for x in xs]
            ax.plot(xs, ys, color='#2a78d6', lw=2, zorder=2)
        cx = sorted({px(k[0]) for k in CAMPAIGN[pattern]['ei']})
        cy = [sum(v for k, v in CAMPAIGN[pattern]['ei'].items() if px(k[0]) == x) / sum(1 for k in CAMPAIGN[pattern]['ei'] if px(k[0]) == x) for x in cx]
        ax.plot(cx, cy, color='#b4b2ad', lw=1.5, zorder=1)
        ax.set_xscale('log')
        ticks = [f0] + sorted({px(v['f']) for v in P['arms'].values() if v['f'] > 0} | {px(k[0]) for k in CAMPAIGN[pattern]['ei'] if k[0] > 0})
        ax.set_xticks(ticks); ax.set_xticklabels(['0'] + [f'{t:g}' for t in ticks[1:]])
        ax.set_xlabel(f'pattern frequency f in pretraining (f = 0 at the left edge)', fontsize=8)
        ax.set_ylabel('acquisition (fraction of targets solved with the pattern)', fontsize=8)
        ax.set_title({'reductio': 'reductio: required pool (300) vs block B (606)', 'derived_ore_strict': 'strict derived ORE, cap 8: required (300) vs block C (500)'}[pattern], fontsize=9)
        ax.spines[['top', 'right']].set_visible(False); ax.grid(color='#e8e7e3', lw=0.6); ax.set_axisbelow(True); ax.set_ylim(-0.01, None)
    axes[0].legend(handles=[Line2D([], [], marker='o', color='#2a78d6', lw=2, label='EI, required pool (this run)'), Line2D([], [], marker='o', mfc='none', color='#2a78d6', lw=0, label='frozen, required pool'),
                            Line2D([], [], marker='s', color='#b4b2ad', lw=1.5, label='EI, campaign pool'), Line2D([], [], marker='s', mfc='none', color='#d9d7d2', lw=0, label='frozen, campaign pool')], fontsize=7, frameon=False, loc='upper left')
    fig.tight_layout(); fig.savefig(f'{a.figs}/run5_dials.png', dpi=160)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, pattern in zip(axes, res):
        P = res[pattern]; n = P.get('n_targets', 300)
        COL = {0: '#2a78d6', 0.1: '#eb6834', 0.01: '#eb6834', 0.001: '#7aa6db'}
        for k, v in P['arms'].items():
            y = [c / n for c in v['per_round_acq']]; x = list(range(1, len(y) + 1))
            ax.plot(x, y, '-' if v['kind'] == 'ei' else '--', color=COL[v['f']], lw=2 if v['kind'] == 'ei' else 1, marker='o' if v['kind'] == 'ei' else 'x', ms=3)
            ax.annotate(f"{'EI' if v['kind'] == 'ei' else 'fr'} f={v['f']:g} s{v['seed']}", (x[-1], y[-1]), textcoords='offset points', xytext=(4, 0), fontsize=6, color='#52514e', va='center')
        ax.set_xlabel('expert-iteration round'); ax.set_ylabel(f'acquisition (of {n} required targets)'); ax.set_title(pattern, fontsize=9)
        ax.spines[['top', 'right']].set_visible(False); ax.grid(color='#e8e7e3', lw=0.6); ax.set_axisbelow(True); ax.set_xlim(0.8, 9.6)
    fig.tight_layout(); fig.savefig(f'{a.figs}/run5_curves.png', dpi=160)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
