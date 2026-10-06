#!/usr/bin/env python3
"""Run 4 summary: GRPO arms (artifacts/r4/grpo_g{8,32}_{a1,a2,a3}_s{0,1}) vs the follow-up's EI arms on the same draws
(artifacts/p2/ei_depth3_f0_{a1,a2,a3}_s{0,1}), at matched sample budgets. Per round-equivalent: targets solved (distinct
verified proofs sampled during training), depth-3 acquisition (patterns.depth3 on the normalised proof, min-round rule),
transfer pass@32 at the boundary and its depth-3 count, held-out greedy, mean reward and fraction of groups with reward
variance (GRPO only). Figure run4_grpo_vs_ei.png.
  python run4_analysis.py --out artifacts/r4/summary.json --figs figures
"""
import argparse, json, os, sys, glob, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_metrics import arm_metrics

EI_ACQ = {'a1_s0': 0.335, 'a1_s1': 0.364, 'a2_s0': 0.350, 'a2_s1': 0.341, 'a3_s0': 0.361, 'a3_s1': 0.352}   # numbers.md, follow-up block A
EI_SOLVED = {'a1_s0': 645, 'a1_s1': 698, 'a2_s0': 589, 'a2_s1': 652, 'a3_s0': 605, 'a3_s1': 583}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='artifacts/r4/summary.json'); ap.add_argument('--figs', default='figures')
    a = ap.parse_args()
    res = {'grpo': {}, 'ei': {}}
    for d in sorted(glob.glob('artifacts/r4/grpo_g*')):
        if not glob.glob(f'{d}/round_*.json'): continue
        m = arm_metrics(d, 'depth3'); last = m[-1]
        steps = []
        for r in m:
            st = json.load(open(f'{d}/round_{r["round"]}.json'))
            steps.append({'round': r['round'], 'samples': st['samples'], 'mean_reward': st['targets_round']['rate'], 'frac_var': st['frac_groups_with_variance_mean'], 'heldout': st['heldout_greedy']['rate']})
        name = os.path.basename(d)
        res['grpo'][name] = {'group': int(name.split('_g')[1].split('_')[0]), 'draw': name.split('_', 2)[2], 'final_round': last['round'], 'solved': last['targets_solved'],
                             'acq': last['acq_targets'], 'acq_theorems': last['acq_targets_theorems'], 'pattern_proofs': last['n_pattern_proofs_targets'], 'first_round': last['first_round_pattern_targets'],
                             'per_round_acq': [r['acq_targets_theorems'] for r in m], 'per_round_solved': [r['targets_solved'] for r in m],
                             'transfer_solved': last['transfer_solved'], 'transfer_acq': last['acq_transfer_theorems'], 'heldout_greedy': last['heldout_greedy'], 'per_round': steps,
                             'written_hist': last['written_hist_targets'], 'examples': last['pattern_examples_targets'][:3]}
    for d in sorted(glob.glob('artifacts/p2/ei_depth3_f0_a[123]_s[01]')):
        if not glob.glob(f'{d}/round_*.json'): continue
        m = arm_metrics(d, 'depth3'); last = m[-1]
        res['ei'][os.path.basename(d)] = {'solved': last['targets_solved'], 'acq': last['acq_targets'], 'per_round_acq': [r['acq_targets_theorems'] for r in m], 'per_round_solved': [r['targets_solved'] for r in m], 'heldout_greedy': last['heldout_greedy']}
    json.dump(res, open(a.out, 'w'), indent=1)
    for k, v in res['grpo'].items():
        print(f"{k:16s} G={v['group']:2d} r{v['final_round']} solved {v['solved']:4d} acq {v['acq_theorems']:3d} ({v['acq']:.3f}) first {v['first_round']} per-round {v['per_round_acq']} var {[round(s['frac_var'], 2) for s in v['per_round']]} reward {[round(s['mean_reward'], 2) for s in v['per_round']]} heldout {v['heldout_greedy']:.3f} | EI {v['draw']}: acq {EI_ACQ.get(v['draw'])} solved {EI_SOLVED.get(v['draw'])}")
    for k, v in res['ei'].items():
        print(f"  EI {k}: solved {v['solved']} acq {v['acq']:.3f} per-round {v['per_round_acq']}")
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    for k, v in res['ei'].items():
        ax.plot(range(1, len(v['per_round_acq']) + 1), [c / 1000 for c in v['per_round_acq']], color='#b4b2ad', lw=1.2, ls=':')
    for k, v in res['grpo'].items():
        col = '#2a78d6' if v['group'] == 8 else '#eb6834'
        ax.plot(range(1, len(v['per_round_acq']) + 1), [c / 1000 for c in v['per_round_acq']], color=col, lw=1.8, marker='o', ms=3)
        ax.annotate(k.replace('grpo_', ''), (len(v['per_round_acq']), v['per_round_acq'][-1] / 1000), textcoords='offset points', xytext=(3, 0), fontsize=6, color='#52514e', va='center')
    ax.set_xlabel('round-equivalent (32 samples per target each)'); ax.set_ylabel('depth-3 acquisition (of 1,000 targets)'); ax.set_title('GRPO G = 8 (blue), G = 32 (orange), EI (grey dotted)', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False); ax.grid(color='#e8e7e3', lw=0.6); ax.set_axisbelow(True); ax.set_xlim(0.8, 9.5); ax.set_ylim(-0.01, None)
    ax = axes[1]
    for k, v in res['grpo'].items():
        col = '#2a78d6' if v['group'] == 8 else '#eb6834'
        ax.plot([s['round'] for s in v['per_round']], [s['frac_var'] for s in v['per_round']], color=col, lw=1.5, marker='o', ms=3)
    ax.set_xlabel('round-equivalent'); ax.set_ylabel('fraction of groups with reward variance'); ax.set_title('training signal per step', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False); ax.grid(color='#e8e7e3', lw=0.6); ax.set_axisbelow(True); ax.set_xlim(0.8, 8.5); ax.set_ylim(-0.01, 1.0)
    fig.tight_layout(); fig.savefig(f'{a.figs}/run4_grpo_vs_ei.png', dpi=160); print('wrote', a.out)


if __name__ == '__main__':
    main()
