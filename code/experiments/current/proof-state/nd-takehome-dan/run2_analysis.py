#!/usr/bin/env python3
"""Run 2 summary: per arm and round, targets solved and targets solved with a proof containing the run-2 pattern
(patterns2.classify2 on the normalised proof, min-round rule), violations on the required stratum (required target solved
with no pattern proof), frozen controls, f>0 control arms, pre-RL pass@2,000 (coverage files re-classified with
patterns2), and the drift measurement (pattern rate of round-1..4 checkpoints, if measured).
  python run2_analysis.py --out artifacts/r2/summary.json --figs figures
"""
import argparse, json, os, sys, glob, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import norm
from patterns2 import classify2, CLASS

PATS = ['depth4', 'impe_chain4', 'nested_ore', 'impi_ore', 'negi_ande_hyp', 'ori_ore']
THRESH = 0.02
ARMS = {p: [('ei', f's{s}') for s in (0, 1)] + [('frozen', f's{s}') for s in (0, 1)] for p in PATS}
ARMS['depth4'] += [('ei', f'c8ctl_s{s}') for s in (0, 1)] + [('frozen', f'c8ctl_s{s}') for s in (0, 1)] + [('ei', f'c8f0_s{s}') for s in (0, 1)] + [('frozen', f'c8f0_s{s}') for s in (0, 1)]
for p in ('impi_ore', 'negi_ande_hyp'):
    ARMS[p] += [('ei', f'natctl_s{s}') for s in (0, 1)] + [('frozen', f'natctl_s{s}') for s in (0, 1)]


def rd(fn):
    return [json.loads(l) for l in open(fn) if l.strip()]


def arm_rounds(d, pattern, T):
    rounds = sorted(int(f.split('_')[-1].split('.')[0]) for f in glob.glob(f'{d}/round_*.json'))
    if not rounds:
        return None
    last = max(rounds)
    fn = f'{d}/found_{last}.jsonl' if os.path.exists(f'{d}/found_{last}.jsonl') else None
    recs = rd(fn) if fn else []
    minround = {}
    for x in recs:
        k = (x['name'], norm(x['proof'])); minround[k] = min(minround.get(k, 99), x['round'])
    cache = {}
    per = []
    for r in rounds:
        st = json.load(open(f'{d}/round_{r}.json'))
        solved, pat_thms, n_pat, seen = set(), set(), 0, set()
        for x in recs:
            pn = norm(x['proof']); k = (x['name'], pn)
            if minround[k] > r or k in seen:
                continue
            seen.add(k); solved.add(x['name'])
            if pn not in cache:
                c = classify2(pn); cache[pn] = bool(c and c[pattern])
            if cache[pn]:
                n_pat += 1; pat_thms.add(x['name'])
        req = {t['name'] for t in T if t.get('requires')}
        viol = sorted((solved & req) - pat_thms)
        per.append({'round': r, 'solved': len(solved), 'pattern_theorems': len(pat_thms), 'pattern_proofs': n_pat, 'distinct_proofs': len(seen),
                    'solved_required': len(solved & req), 'pattern_required': len(pat_thms & req), 'violations_required': len(viol),
                    'heldout_greedy': st['heldout_greedy']['rate'], 'transfer_greedy': st['transfer_greedy']['rate'], 'transfer_solved': st['transfer_cum']['solved'],
                    'reasons_top': sorted(st['targets_round'].get('reasons', {}).items(), key=lambda kv: -kv[1])[:3]})
    ex = []
    for x in recs:
        pn = norm(x['proof'])
        if cache.get(pn) and len(ex) < 5:
            ex.append({'thm': x['thm'], 'proof': pn, 'round': minround[(x['name'], pn)], 'written': x['written']})
    n = len(T)
    ign = next((p['round'] for p in per if p['pattern_theorems'] >= THRESH * n), None)
    return {'rounds': per, 'n_targets': n, 'final': per[-1], 'acq': per[-1]['pattern_theorems'] / n, 'ignition_round': ign,
            'first_pattern_round': min((minround[(x['name'], norm(x['proof']))] for x in recs if cache.get(norm(x['proof']))), default=None), 'examples': ex}


def coverage(fn, pattern):
    if not os.path.exists(fn):
        return None
    rs = rd(fn)
    hits = 0; thms = 0; first = []
    for r in rs:
        h = 0; f = None
        for p in r['proofs']:
            c = classify2(p['proof'])
            if c and c[pattern]:
                h += p['count']; f = p['first'] if f is None else min(f, p['first'])
        hits += h; thms += h > 0
        if f is not None: first.append(f)
    return {'targets': len(rs), 'samples': sum(r['n_tried'] for r in rs), 'solved': sum(r['n_ok'] > 0 for r in rs), 'pattern_hits': hits, 'targets_with_pattern': thms,
            'rate': hits / max(1, sum(r['n_tried'] for r in rs)), 'frozen256_pattern_targets': sum(f <= 256 for f in first)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='artifacts/r2/summary.json'); ap.add_argument('--figs', default='figures')
    a = ap.parse_args()
    res = {}
    for p in PATS:
        T = rd(f'data/r2/targets_{p}.jsonl')
        P = {'class': CLASS[p], 'n_targets': len(T), 'n_required': sum(bool(t.get('requires')) for t in T), 'min_lines_hist': dict(sorted(collections.Counter(t['n_lines'] for t in T).items())), 'arms': {}, 'coverage': {}, 'drift': {}}
        for kind, tag in ARMS[p]:
            d = f'artifacts/r2/{kind}_{p}_{tag}'
            m = arm_rounds(d, p, T)
            if m: P['arms'][f'{kind}_{tag}'] = m
        for tag in ('s0', 's1', 'c8ctl_s0', 'c8ctl_s1', 'natctl_s0', 'natctl_s1', 'c8f0_s0', 'c8f0_s1'):
            c = coverage(f'artifacts/r2/cov_{p}_{tag}.s0.jsonl', p)
            if c: P['coverage'][tag] = c
        for fn in sorted(glob.glob(f'artifacts/r2/drift_{p}_*.s0.jsonl')):
            P['drift'][os.path.basename(fn).split(f'drift_{p}_')[1].split('.s0')[0]] = coverage(fn, p)
        res[p] = P
    json.dump(res, open(a.out, 'w'), indent=1)
    for p, P in res.items():
        print(f"== {p} [{P['class']}] targets {P['n_targets']} (required {P['n_required']}) min-lines {P['min_lines_hist']}")
        for k, v in P['arms'].items():
            f = v['final']
            print(f"  {k:14s} r{f['round']} solved {f['solved']:3d} pattern {f['pattern_theorems']:3d} ({v['acq']:.3f}) proofs {f['pattern_proofs']:4d} req-solved {f['solved_required']:3d} req-pattern {f['pattern_required']:3d} viol {f['violations_required']:2d} ign {v['ignition_round']} first {v['first_pattern_round']} per-round {[r['pattern_theorems'] for r in v['rounds']]} solved/round {[r['solved'] for r in v['rounds']]} heldout {f['heldout_greedy']:.3f}")
        for k, v in P['coverage'].items():
            print(f"  cov {k}: {v}")
        for k, v in P['drift'].items():
            print(f"  drift {k}: {v}")
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5)); axes = axes.ravel()
    COL = {'s0': '#2a78d6', 's1': '#2a78d6', 'c8ctl_s0': '#eb6834', 'c8ctl_s1': '#eb6834', 'natctl_s0': '#eb6834', 'natctl_s1': '#eb6834', 'c8f0_s0': '#2ca02c', 'c8f0_s1': '#2ca02c'}
    for ax, p in zip(axes, PATS):
        P = res[p]; n = P['n_targets']
        for k, v in P['arms'].items():
            kind, tag = k.split('_', 1)
            y = [r['pattern_theorems'] / n for r in v['rounds']]; x = [r['round'] for r in v['rounds']]
            ax.plot(x, y, '-' if kind == 'ei' else '--', color=COL.get(tag, '#52514e'), lw=2 if kind == 'ei' else 1, marker='o' if kind == 'ei' else 'x', ms=3)
            ax.annotate(f"{'EI' if kind == 'ei' else 'fr'} {tag}", (x[-1], y[-1]), textcoords='offset points', xytext=(4, 0), fontsize=6, color='#52514e', va='center')
        ax.set_title(f"{p} ({P['class']}, n = {n})", fontsize=9); ax.set_xlabel('round'); ax.set_ylabel('acquisition'); ax.set_xlim(0.8, 9.8); ax.set_ylim(-0.01, None)
        ax.spines[['top', 'right']].set_visible(False); ax.grid(color='#e8e7e3', lw=0.6); ax.set_axisbelow(True)
    fig.suptitle('Run 2: acquisition per round at f = 0 (blue) and f > 0 controls (orange); dashed = frozen', fontsize=10)
    fig.tight_layout(); fig.savefig(f'{a.figs}/run2_curves.png', dpi=160)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    xs, ys, cs = [], [], []
    for i, p in enumerate(PATS):
        for k, v in res[p]['arms'].items():
            kind, tag = k.split('_', 1)
            if kind != 'ei': continue
            xs.append(i + (0.25 if 'ctl' in tag else -0.1) + (0.1 if tag.endswith('s1') else 0)); ys.append(v['acq']); cs.append('#eb6834' if 'ctl' in tag else '#2a78d6')
    ax.scatter(xs, ys, c=cs, s=50, zorder=3)
    ax.set_xticks(range(len(PATS))); ax.set_xticklabels([f"{p}\n({CLASS[p][:4]})" for p in PATS], fontsize=8); ax.set_ylabel('round-8 acquisition'); ax.set_ylim(-0.01, None)
    ax.spines[['top', 'right']].set_visible(False); ax.grid(color='#e8e7e3', lw=0.6); ax.set_axisbelow(True)
    ax.set_title('Run 2: f = 0 arms (blue) vs f > 0 controls (orange), two seeds each', fontsize=9)
    fig.tight_layout(); fig.savefig(f'{a.figs}/run2_acq.png', dpi=160)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
