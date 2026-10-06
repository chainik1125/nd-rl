#!/usr/bin/env python3
"""Follow-up block B summary: strict / loose reductio acquisition on targets_reductio2 for every arm, frozen controls,
per-schema breakdown, base log-probs; figure figures/followup_reductio.png (acquisition per round, EI solid, frozen dashed).
  python followup_reductio.py --out artifacts/fu/blockB_summary.json --fig figures/followup_reductio.png
"""
import argparse, json, os, sys, glob, math, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_metrics import arm_metrics
from patterns import classify

ARMS = [('ei', 0, 0), ('ei', 0, 1), ('ei', 0, 2), ('ei', 0.1, 0), ('ei', 0.1, 1), ('frozen', 0, 0), ('frozen', 0, 1), ('frozen', 0, 2), ('frozen', 0.1, 0), ('frozen', 0.1, 1)]


def novelty(f, s):
    fn = f'artifacts/p2/novelty_reductio_f{f:g}_s{s}_t2_proofs.jsonl'
    if not os.path.exists(fn):
        return None
    lp = []
    for l in open(fn):
        r = json.loads(l)
        if r['src'] != 'ei_targets':
            continue
        cl = classify(r['proof'])
        if cl and cl['reductio']:
            lp.append((r['base_logp_T08'], r['round'], r['name'], r['base_max_line_rule']))
    if not lp:
        return {'n': 0}
    lp.sort(reverse=True)
    return {'n': len(lp), 'max_logp': lp[0][0], 'max_round': lp[0][1], 'median_logp': lp[len(lp) // 2][0],
            'below_1_256': sum(v[0] < math.log(1 / 256) for v in lp), 'below_1e-4': sum(v[0] < math.log(1e-4) for v in lp),
            'below_1e-5': sum(v[0] < math.log(1e-5) for v in lp), 'theorems_above_1_256': len({v[2] for v in lp if v[0] > math.log(1 / 256)}),
            'max_surprisal_rule': collections.Counter(v[3] for v in lp).most_common(3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='artifacts/fu/blockB_summary.json'); ap.add_argument('--fig', default='figures/followup_reductio.png')
    a = ap.parse_args()
    T = {json.loads(l)['name']: json.loads(l) for l in open('data/p2/targets_reductio2.jsonl')}
    n = len(T)
    res = {'n_targets': n, 'n_transfer': sum(1 for _ in open('data/p2/transfer_reductio2.jsonl')), 'arms': {}}
    for kind, f, s in ARMS:
        d = f'artifacts/p2/{kind}_reductio_f{f:g}_s{s}_t2'
        if not glob.glob(f'{d}/round_*.json'):
            continue
        m = arm_metrics(d, 'reductio')
        last = m[-1]
        res['arms'][f'{kind}_f{f:g}_s{s}'] = {
            'kind': kind, 'f': f, 'seed': s, 'round': last['round'], 'solved': last['targets_solved'],
            'strict': last['acq_targets_theorems'], 'loose': last['acq_loose_targets_theorems'], 'any_dn': last['acq_anydn_targets_theorems'],
            'strict_proofs': last['n_pattern_proofs_targets'], 'first_round': last['first_round_pattern_targets'],
            'per_round_solved': [r['targets_solved'] for r in m], 'per_round_strict': [r['acq_targets_theorems'] for r in m],
            'per_round_loose': [r['acq_loose_targets_theorems'] for r in m],
            'transfer_solved': last['transfer_solved'], 'transfer_strict': last['acq_transfer_theorems'], 'transfer_loose': last['acq_loose_transfer_theorems'],
            'heldout_greedy': last['heldout_greedy'], 'transfer_greedy': last['transfer_greedy'],
            'solved_by_schema': dict(collections.Counter(T[x]['schema'] for x in last['solved_names_targets'])),
            'strict_by_schema': dict(collections.Counter(T[x]['schema'] for x in last['pattern_names_targets'])),
            'novelty': novelty(f, s) if kind == 'ei' else None}
    res['schema_sizes'] = dict(collections.Counter(r['schema'] for r in T.values()))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    print(f"{'arm':16s} {'solved':>6s} {'strict':>6s} {'loose':>6s} {'anyDN':>6s} {'first':>5s} {'xfer':>9s}  schemata (strict)")
    for k, v in res['arms'].items():
        print(f"{k:16s} {v['solved']:6d} {v['strict']:6d} {v['loose']:6d} {v['any_dn']:6d} {str(v['first_round']):>5s} {v['transfer_solved']:4d}/{v['transfer_strict']:<4d}  {v['strict_by_schema']}")
        if v['novelty']:
            print('   novelty:', v['novelty'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    COL = {0: '#2a78d6', 0.1: '#eb6834'}
    fig, ax = plt.subplots(figsize=(8, 4.2))
    zeros = []
    for k, v in res['arms'].items():
        x = list(range(1, len(v['per_round_strict']) + 1)); y = [c / n for c in v['per_round_strict']]
        ax.plot(x, y, '-' if v['kind'] == 'ei' else '--', color=COL[v['f']], lw=2 if v['kind'] == 'ei' else 1.2, marker='o' if v['kind'] == 'ei' else 'x', ms=4)
        lab = f"{'EI' if v['kind'] == 'ei' else 'frozen'} f={v['f']:g} s{v['seed']}"
        if y[-1] == 0:
            zeros.append(lab); continue
        ax.annotate(lab, (x[-1], y[-1]), textcoords='offset points', xytext=(4, 0), fontsize=6.5, color='#52514e', va='center')
    if zeros:
        ax.annotate('at 0 throughout: ' + ', '.join(zeros), (4.5, 0), textcoords='offset points', xytext=(0, -11), fontsize=6.5, color='#52514e', va='center', ha='center')
    ax.set_xlabel('expert-iteration round (32 attempts per target per round)'); ax.set_ylabel(f'strict reductio acquisition (fraction of {n} targets)')
    ax.set_xlim(0.8, 9.3); ax.set_ylim(-0.012, 0.17); ax.spines[['top', 'right']].set_visible(False); ax.grid(color='#e8e7e3', lw=0.6); ax.set_axisbelow(True)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=COL[0], lw=2, label='f = 0 (no reductio in pretraining)'), Line2D([], [], color=COL[0.1], lw=2, label='f = 0.1'),
                       Line2D([], [], color='#52514e', ls='--', label='frozen control')], fontsize=7.5, frameon=False, loc='upper left')
    ax.set_title('Block B: targets that need a derived double negation, solved with the strict reductio shape', fontsize=9)
    fig.tight_layout(); fig.savefig(a.fig, dpi=160); print('wrote', a.out, a.fig)


if __name__ == '__main__':
    main()
