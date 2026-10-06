#!/usr/bin/env python3
"""Follow-up block A: depth-3 replication statistics and dot-strip figure.

  python followup_analysis.py --out artifacts/fu/blockA_summary.json --fig figures/followup_depth3_strips.png

Arms (artifacts/p2/): ei_depth3_f0_s{0,1} (campaign-1 set, "a0"), ei_depth3_f0_a{1,2,3}_s{0,1}, ei_depth3_f0.1_s{0,1} ("b0"),
ei_depth3_f0.1_b{1,2}_s{0,1}; frozen_* likewise. Metrics via phase2_metrics.arm_metrics (model's proof classified,
start-index normalised). Variance decomposition: one-way random-effects, method of moments, groups = assembler sets,
2 training seeds per set:  MS_between = n * Var(set means), MS_within = mean within-set variance,
sigma2_seed = MS_within, sigma2_set = max(0, (MS_between - MS_within)/n).
"""
import argparse, json, os, sys, glob, re, collections, math, itertools, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_metrics import arm_metrics

SETS = {0: ['', '_a1', '_a2', '_a3'], 0.1: ['', '_b1', '_b2']}


def arm_name(kind, f, suffix, seed):
    return f'artifacts/p2/{kind}_depth3_f{f:g}{suffix}_s{seed}'


def summarize(vals):
    n = len(vals)
    if n == 0:
        return {'n': 0}
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return {'n': n, 'mean': m, 'sd': sd, 'min': min(vals), 'max': max(vals), 'values': vals}


def decompose(groups):
    """groups: list of lists (one per set) of per-seed values."""
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return None
    n = min(len(g) for g in groups)
    means = [sum(g) / len(g) for g in groups]
    gm = sum(means) / len(means)
    ms_between = n * sum((m - gm) ** 2 for m in means) / (len(means) - 1)
    ms_within = sum(sum((v - sum(g) / len(g)) ** 2 for v in g) / (len(g) - 1) for g in groups) / len(groups)
    s2_seed = ms_within
    s2_set = max(0.0, (ms_between - ms_within) / n)
    return {'n_sets': len(groups), 'seeds_per_set': n, 'ms_between': ms_between, 'ms_within': ms_within,
            'sigma2_set': s2_set, 'sigma2_seed': s2_seed, 'sd_set': math.sqrt(s2_set), 'sd_seed': math.sqrt(s2_seed),
            'F': ms_between / ms_within if ms_within > 0 else None,
            'set_share': s2_set / (s2_set + s2_seed) if (s2_set + s2_seed) > 0 else None}


def perm_test(a, b, n_perm=20000, seed=0):
    """two-sided permutation test on the difference of means."""
    rng = random.Random(seed)
    obs = abs(sum(a) / len(a) - sum(b) / len(b))
    pool = list(a) + list(b); cnt = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        x, y = pool[:len(a)], pool[len(a):]
        if abs(sum(x) / len(x) - sum(y) / len(y)) >= obs - 1e-12:
            cnt += 1
    return {'diff_of_means': sum(a) / len(a) - sum(b) / len(b), 'p_two_sided': (cnt + 1) / (n_perm + 1)}


def novelty_stats(tag_suffix, seed, f):
    fn = f'artifacts/p2/novelty_depth3_f{f:g}{tag_suffix}_s{seed}_proofs.jsonl'
    if not os.path.exists(fn):
        return None
    from patterns import classify
    lp = []; above = set(); above4 = set()
    for l in open(fn):
        r = json.loads(l)
        if r['src'] != 'ei_targets':
            continue
        cl = classify(r['proof'])
        if cl and cl['depth3']:
            lp.append(r['base_logp_T08'])
            if r['base_logp_T08'] > math.log(1 / 256):
                above.add(r['name'])
            if r['base_logp_T08'] > math.log(1e-4):
                above4.add(r['name'])
    if not lp:
        return {'n': 0}
    lp.sort()
    return {'n': len(lp), 'below_1_256': sum(v < math.log(1 / 256) for v in lp), 'below_1e-4': sum(v < math.log(1e-4) for v in lp),
            'below_1e-5': sum(v < math.log(1e-5) for v in lp), 'max_logp': max(lp), 'median_logp': lp[len(lp) // 2],
            'theorems_above_1_256': len(above), 'theorems_above_1e-4': len(above4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='artifacts/fu/blockA_summary.json')
    ap.add_argument('--fig', default='figures/followup_depth3_strips.png')
    ap.add_argument('--round', type=int, default=8)
    a = ap.parse_args()
    arms = []
    for f, sufs in SETS.items():
        for suf in sufs:
            for seed in (0, 1):
                for kind in ('ei', 'frozen'):
                    d = arm_name(kind, f, suf, seed)
                    if not glob.glob(f'{d}/round_*.json'):
                        continue
                    m = arm_metrics(d, 'depth3')
                    rr = [r for r in m if r['round'] <= a.round]
                    last = rr[-1]
                    arms.append({'kind': kind, 'f': f, 'set': 'a0' if (suf == '' and f == 0) else ('b0' if suf == '' else suf.strip('_')),
                                 'seed': seed, 'round': last['round'], 'campaign1': suf == '',
                                 'targets_solved': last['targets_solved'], 'acq': last['acq_targets'], 'acq_theorems': last['acq_targets_theorems'],
                                 'n_depth3_proofs': last['n_pattern_proofs_targets'], 'first_round': last['first_round_pattern_targets'],
                                 'transfer_solved': last['transfer_solved'], 'acq_transfer': last['acq_transfer'],
                                 'heldout_greedy': last['heldout_greedy'], 'transfer_greedy': last['transfer_greedy'],
                                 'novelty': novelty_stats(suf, seed, f) if kind == 'ei' else None})
    res = {'arms': arms, 'round': a.round}
    for f in (0, 0.1):
        ei = [x for x in arms if x['kind'] == 'ei' and x['f'] == f and x['round'] == a.round]
        fr = [x for x in arms if x['kind'] == 'frozen' and x['f'] == f]
        vals = [x['acq'] for x in ei]
        groups = collections.defaultdict(list)
        for x in ei:
            groups[x['set']].append(x['acq'])
        res[f'f{f:g}'] = {'acq': summarize(vals), 'acq_by_set': dict(groups), 'decomposition': decompose(list(groups.values())),
                          'first_round': {f"{x['set']}_s{x['seed']}": x['first_round'] for x in ei},
                          'frozen_depth3_proofs': {f"{x['set']}_s{x['seed']}": x['n_depth3_proofs'] for x in fr},
                          'frozen_solved': {f"{x['set']}_s{x['seed']}": x['targets_solved'] for x in fr},
                          'solved': summarize([x['targets_solved'] for x in ei])}
    e0 = [x['acq'] for x in arms if x['kind'] == 'ei' and x['f'] == 0 and x['round'] == a.round]
    e1 = [x['acq'] for x in arms if x['kind'] == 'ei' and x['f'] == 0.1 and x['round'] == a.round]
    if e0 and e1:
        res['f0_vs_f0.1'] = perm_test(e0, e1)
        res['f0_vs_f0.1']['overlap'] = {'f0_range': [min(e0), max(e0)], 'f0.1_range': [min(e1), max(e1)],
                                        'n_f0_within_f0.1_range': sum(min(e1) <= v <= max(e1) for v in e0),
                                        'n_f0.1_within_f0_range': sum(min(e0) <= v <= max(e0) for v in e1)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    # table
    print(f"{'kind':6s} {'f':4s} {'set':4s} {'seed':4s} {'r':2s} {'solved':7s} {'acq':6s} {'thms':5s} {'proofs':6s} {'first':5s} {'xfer acq':8s} {'H greedy':8s}  novelty(depth-3 proofs: n / <1e-5 / max logp)")
    for x in sorted(arms, key=lambda x: (x['f'], x['set'], x['seed'], x['kind'])):
        nv = x['novelty']
        nvs = f"{nv['n']} / {nv.get('below_1e-5')} / {nv.get('max_logp', float('nan')):.1f} / thms>1/256 {nv.get('theorems_above_1_256')}" if nv and nv.get('n') else ('-' if nv is None else '0')
        print(f"{x['kind']:6s} {x['f']:<4g} {x['set']:4s} {x['seed']:<4d} {x['round']:<2d} {x['targets_solved']:<7d} {x['acq']:<6.3f} {x['acq_theorems']:<5d} {x['n_depth3_proofs']:<6d} {str(x['first_round']):5s} {x['acq_transfer']:<8.3f} {x['heldout_greedy']:<8.3f}  {nvs}")
    for f in (0, 0.1):
        r = res[f'f{f:g}']
        print(f"f={f:g}: acquisition n={r['acq'].get('n')} mean {r['acq'].get('mean', float('nan')):.3f} sd {r['acq'].get('sd', float('nan')):.3f} values {[round(v, 3) for v in r['acq'].get('values', [])]}")
        print('   decomposition', r['decomposition'])
        print('   frozen depth-3 proofs', r['frozen_depth3_proofs'])
    if 'f0_vs_f0.1' in res:
        print('f0 vs f0.1:', res['f0_vs_f0.1'])
    # figure: dot strips
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    COL = {0: '#2a78d6', 0.1: '#eb6834'}     # validated categorical slots 1 and 2 (dataviz reference palette)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    rng = random.Random(1)
    order = ['a0', 'a1', 'a2', 'a3', 'b0', 'b1', 'b2']
    for i, f in enumerate((0, 0.1)):
        y0 = 1 - i
        ei_arms = sorted([x for x in arms if x['f'] == f and x['kind'] == 'ei' and x['round'] == a.round], key=lambda x: (order.index(x['set']), x['seed']))
        fr_arms = sorted([x for x in arms if x['f'] == f and x['kind'] == 'frozen'], key=lambda x: (order.index(x['set']), x['seed']))
        for arr in (ei_arms, fr_arms):
            k = len(arr)
            for j, x in enumerate(arr):
                jit = (j - (k - 1) / 2) * (0.5 / max(k - 1, 1)) if k > 1 else 0.0
                if x['kind'] == 'ei':
                    ax.scatter([x['acq']], [y0 + jit], s=70 if x['campaign1'] else 55, facecolors=COL[f], edgecolors='white', linewidths=1.5,
                               marker='D' if x['campaign1'] else 'o', zorder=3)
                    ax.annotate(f"{x['set']} s{x['seed']}", (x['acq'], y0 + jit), textcoords='offset points', xytext=(9, -2.5), ha='left', fontsize=6, color='#52514e')
                else:
                    ax.scatter([x['acq']], [y0 + jit], s=45, facecolors='none', edgecolors=COL[f], linewidths=1.5, marker='o', zorder=3)
        vals = [x['acq'] for x in ei_arms]
        if vals:
            m = sum(vals) / len(vals)
            ax.plot([m, m], [y0 - 0.32, y0 + 0.32], color=COL[f], lw=2, zorder=2)
    ax.set_yticks([1, 0]); ax.set_yticklabels(['f = 0\n(zero depth-3 proofs\nin pretraining)', 'f = 0.1'])
    ax.set_ylim(-0.5, 1.5); ax.set_xlim(-0.02, 0.46)
    ax.set_xlabel('depth-3 acquisition after 8 rounds (fraction of 1,000 targets solved with a depth-3 proof)', fontsize=8.5)
    ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='x', color='#e8e7e3', lw=0.6); ax.set_axisbelow(True)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker='o', ls='', color='#52514e', label='EI arm (new set)'),
                       Line2D([], [], marker='D', ls='', color='#52514e', label='EI arm (campaign-1 set)'),
                       Line2D([], [], marker='o', ls='', markerfacecolor='none', color='#52514e', label='frozen control'),
                       Line2D([], [], color='#52514e', lw=2, label='mean of EI arms')], fontsize=7.5, loc='center left', bbox_to_anchor=(0.22, 0.5), frameon=False)
    ax.set_title('Depth-3 acquisition: every pretraining set x training seed (label = set, seed)', fontsize=9.5)
    fig.tight_layout(); os.makedirs(os.path.dirname(a.fig), exist_ok=True); fig.savefig(a.fig, dpi=160)
    print('wrote', a.out, a.fig)


if __name__ == '__main__':
    main()
