#!/usr/bin/env python3
"""Run 3 summary: per arm x condition, cumulative pattern theorems per round 5..8 (min-round rule on found_8.jsonl, counting
only proofs found in rounds >= 5 as new; the parent's round-4 state is the baseline), ignition round, round-8 acquisition,
the injected records (manifest), and the parent's own trajectory for comparison; figure run3_ignition.png.
  python run3_analysis.py --out artifacts/r3/summary.json --figs figures
"""
import argparse, json, os, sys, glob, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import norm
from patterns import classify

ARMS = [('depth3', 'ei_depth3_f0_a1_s4', 4), ('depth3', 'ei_depth3_f0_a1_s5', 5), ('depth3', 'ei_depth3_f0_a1_s3', 3), ('reductio', 'ei_reductio_f0_s1_t2', 1), ('reductio', 'ei_reductio_f0_s2_t2', 2)]
CONDS = ['sib1', 'sib4', 'sib16', 'gen4', 'other4', 'inv4']
N = {'depth3': 1000, 'reductio': 606}; THRESH = {'depth3': 20, 'reductio': 12}
PARENT = {'ei_depth3_f0_a1_s4': [0, 0, 0, 0, 0, 0, 1, 4], 'ei_depth3_f0_a1_s5': [0] * 8, 'ei_depth3_f0_a1_s3': [0, 0, 1, 3, 28, 166, 265, 295], 'ei_reductio_f0_s1_t2': [0] * 8, 'ei_reductio_f0_s2_t2': [0] * 8}


def per_round(d, pattern):
    rounds = sorted(int(f.split('_')[-1].split('.')[0]) for f in glob.glob(f'{d}/round_*.json'))
    if not rounds:
        return None
    last = max(rounds); fn = f'{d}/found_{last}.jsonl'
    recs = [json.loads(l) for l in open(fn)] if os.path.exists(fn) else []
    minround = {}
    for x in recs:
        k = (x['name'], norm(x['proof'])); minround[k] = min(minround.get(k, 99), x['round'])
    cache = {}; out = []; solved_r = []
    for r in rounds:
        thms = set(); solved = set(); seen = set()
        for x in recs:
            pn = norm(x['proof']); k = (x['name'], pn)
            if minround[k] > r or k in seen: continue
            seen.add(k); solved.add(x['name'])
            if pn not in cache:
                c = classify(pn); cache[pn] = bool(c and c[pattern])
            if cache[pn]: thms.add(x['name'])
        out.append(len(thms)); solved_r.append(len(solved))
    return {'rounds': rounds, 'pattern_theorems': out, 'solved': solved_r}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='artifacts/r3/summary.json'); ap.add_argument('--figs', default='figures')
    a = ap.parse_args()
    res = {}
    for pattern, arm, seed in ARMS:
        A = {'pattern': pattern, 'seed': seed, 'parent_per_round': PARENT[arm], 'conditions': {}}
        for cond in CONDS:
            d = f'artifacts/r3/{arm}_{cond}'
            m = per_round(d, pattern)
            if not m: continue
            man = json.load(open(f'artifacts/p2/{arm}_{cond}_manifest.json')) if os.path.exists(f'artifacts/p2/{arm}_{cond}_manifest.json') else {}
            ign = next((r for r, c in zip(m['rounds'], m['pattern_theorems']) if c >= THRESH[pattern]), None)
            A['conditions'][cond] = {'rounds': m['rounds'], 'pattern_theorems': m['pattern_theorems'], 'solved': m['solved'], 'ignition_round': ign,
                                     'acq_r8': m['pattern_theorems'][-1] / N[pattern] if m['rounds'][-1] == 8 else None, 'final_round': m['rounds'][-1],
                                     'n_injected': man.get('n_injected'), 'injected_valid': man.get('injected_valid'), 'injected_has_pattern': man.get('injected_has_pattern'),
                                     'own_proofs_in_mix': man.get('own_proofs'), 'injected': man.get('injected')}
        res[arm] = A
    json.dump(res, open(a.out, 'w'), indent=1)
    for arm, A in res.items():
        print(f"== {arm} ({A['pattern']}), parent per round {A['parent_per_round']}")
        for cond, v in A['conditions'].items():
            print(f"  {cond:7s} rounds {v['rounds'][0]}-{v['final_round']} pattern {v['pattern_theorems']} solved {v['solved']} ignition {v['ignition_round']} acq_r8 {v['acq_r8']} injected {v['n_injected']} valid {sum(v['injected_valid'] or [])} pattern {sum(v['injected_has_pattern'] or [])}")
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(res), figsize=(3.2 * len(res), 3.4), sharey=False)
    COL = {'sib1': '#9ec5f0', 'sib4': '#5a9be0', 'sib16': '#1f5fb0', 'gen4': '#2ca02c', 'other4': '#eb6834', 'inv4': '#7f7f7f'}
    for ax, (arm, A) in zip(axes if len(res) > 1 else [axes], res.items()):
        n = N[A['pattern']]
        ax.plot(range(1, 9), [c / n for c in A['parent_per_round']], color='black', lw=1.5, ls=':', label='parent')
        for cond, v in A['conditions'].items():
            ax.plot(v['rounds'], [c / n for c in v['pattern_theorems']], marker='o', ms=3, color=COL[cond], lw=1.8, label=cond)
        ax.set_title(arm.replace('ei_', ''), fontsize=8); ax.set_xlabel('round'); ax.set_xlim(0.8, 8.4); ax.set_ylim(-0.01, None)
        ax.spines[['top', 'right']].set_visible(False); ax.grid(color='#e8e7e3', lw=0.6); ax.set_axisbelow(True)
    (axes[0] if len(res) > 1 else axes).set_ylabel('pattern theorems / targets'); (axes[-1] if len(res) > 1 else axes).legend(fontsize=6, frameon=False)
    fig.suptitle('Run 3: one injected training step at round 4, then expert iteration (rounds 5–8)', fontsize=9)
    fig.tight_layout(); fig.savefig(f'{a.figs}/run3_ignition.png', dpi=160); print('wrote', a.out)


if __name__ == '__main__':
    main()
