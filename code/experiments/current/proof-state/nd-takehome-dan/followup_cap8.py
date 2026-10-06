#!/usr/bin/env python3
"""Follow-up block C figure: strict derived-ORE acquisition (cap 8) vs f, EI points and frozen controls.
  python followup_cap8.py --out artifacts/fu/blockC_summary.json --fig figures/followup_cap8_dial.png"""
import argparse, json, os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_metrics import arm_metrics

ARMS = [('ei', 0, 0), ('ei', 0, 1), ('ei', 0.001, 0), ('ei', 0.01, 0), ('frozen', 0, 0), ('frozen', 0, 1)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='artifacts/fu/blockC_summary.json'); ap.add_argument('--fig', default='figures/followup_cap8_dial.png')
    a = ap.parse_args()
    res = {'cap': 8, 'n_targets': 500, 'n_transfer': 250, 'arms': {}}
    for kind, f, s in ARMS:
        d = f'artifacts/p2/{kind}_derived_ore_strict_f{f:g}_c8_s{s}'
        if not glob.glob(f'{d}/round_*.json'):
            continue
        m = arm_metrics(d, 'derived_ore_strict'); last = m[-1]
        res['arms'][f'{kind}_f{f:g}_s{s}'] = {'kind': kind, 'f': f, 'seed': s, 'solved': last['targets_solved'], 'strict': last['acq_targets_theorems'],
                                              'strict_proofs': last['n_pattern_proofs_targets'], 'acq': last['acq_targets'], 'first_round': last['first_round_pattern_targets'],
                                              'per_round_strict': [r['acq_targets_theorems'] for r in m], 'per_round_solved': [r['targets_solved'] for r in m],
                                              'transfer_solved': last['transfer_solved'], 'transfer_strict': last['acq_transfer_theorems'],
                                              'heldout_greedy': last['heldout_greedy'], 'transfer_greedy': last['transfer_greedy'],
                                              'other_patterns': last['other_patterns_targets'], 'written_hist': last['written_hist_targets']}
    cov = 'artifacts/p2/cov_derived_ore_strict_f0_c8_s0_targets.s0.jsonl'
    if os.path.exists(cov):
        rs = [json.loads(l) for l in open(cov)]
        res['coverage_f0_s0'] = {'targets': len(rs), 'samples': sum(r['n_tried'] for r in rs), 'solved': sum(r['n_ok'] > 0 for r in rs),
                                 'solved_within_512': sum(r['solved_within']['512'] for r in rs),
                                 'hits_by_pattern': {k: sum(r['hits_by_pattern'][k] for r in rs) for k in rs[0]['hits_by_pattern']},
                                 'targets_with_strict_proof': sum(r['distinct_by_pattern']['derived_ore_strict'] > 0 for r in rs),
                                 'n_ok_total': sum(r['n_ok'] for r in rs)}
    json.dump(res, open(a.out, 'w'), indent=1)
    for k, v in res['arms'].items():
        print(f"{k:14s} solved {v['solved']:3d}/500 strict {v['strict']:2d} ({v['acq']:.3f}) first {v['first_round']} per-round {v['per_round_strict']} transfer {v['transfer_solved']}/{v['transfer_strict']}")
    print('coverage:', res.get('coverage_f0_s0'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    F0 = 1e-4
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for k, v in res['arms'].items():
        x = F0 if v['f'] == 0 else v['f']
        ax.scatter([x], [v['acq']], s=55 if v['kind'] == 'ei' else 45, facecolors='#2a78d6' if v['kind'] == 'ei' else 'none', edgecolors='white' if v['kind'] == 'ei' else '#2a78d6', linewidths=1.5, zorder=3)
        ax.annotate(f"{'EI' if v['kind'] == 'ei' else 'frozen'} s{v['seed']}", (x, v['acq']), textcoords='offset points', xytext=(7, 0), fontsize=6.5, color='#52514e', va='center')
    xs = sorted({F0 if v['f'] == 0 else v['f'] for v in res['arms'].values() if v['kind'] == 'ei'})
    ys = [sum(v['acq'] for v in res['arms'].values() if v['kind'] == 'ei' and (F0 if v['f'] == 0 else v['f']) == x) / sum(1 for v in res['arms'].values() if v['kind'] == 'ei' and (F0 if v['f'] == 0 else v['f']) == x) for x in xs]
    ax.plot(xs, ys, color='#2a78d6', lw=2, zorder=2)
    ax.set_xscale('log'); ax.set_xticks([F0, 1e-3, 1e-2]); ax.set_xticklabels(['0', '1e-3', '1e-2']); ax.set_xlim(F0 * 0.6, 0.03)
    ax.set_ylim(-0.005, 0.12)
    ax.set_xlabel('strict derived-ORE frequency f in the 155k cap-8 pretraining set (f = 0 at the left edge)', fontsize=8)
    ax.set_ylabel('strict acquisition (fraction of 500 cap-8 targets)', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False); ax.grid(color='#e8e7e3', lw=0.6); ax.set_axisbelow(True)
    ax.set_title('Block C (cap 8): non-degenerate derived ORE is not acquired at any f', fontsize=9)
    fig.tight_layout(); fig.savefig(a.fig, dpi=160); print('wrote', a.fig)


if __name__ == '__main__':
    main()
