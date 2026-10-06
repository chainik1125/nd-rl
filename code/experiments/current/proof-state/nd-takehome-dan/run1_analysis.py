#!/usr/bin/env python3
"""Run 1 summary. Step 1: nd_verify vs Lean agreement table from artifacts/r1/lean_*.jsonl. Step 2: accuracy by surface form
and length bin (greedy pass@1 and pass@8 over samples) from artifacts/r1/scored_qwen30b.jsonl, paired delta Lean - tokens
per (theorem, draw) with a bootstrap CI over theorems. Step 3: scale of first success per theorem class from
artifacts/r1/scored_scale_<model>.jsonl vs Phase-1 base log-probabilities. Figures run1_forms.png, run1_scale.png.
  python run1_analysis.py --out artifacts/r1/summary.json --figs figures
"""
import argparse, json, os, sys, glob, collections, random, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SIZES = ['Qwen3-0.6B', 'Qwen3-1.7B', 'Qwen3-4B', 'Qwen3-8B', 'Qwen3-14B', 'Qwen3-32B']
BINS = ['val36_<=6', 'val36_>6'] + [f'transfer_{L}' for L in range(7, 17)]


def rd(fn):
    return [json.loads(l) for l in open(fn) if l.strip()]


def agreement():
    out = {}
    for fn in sorted(glob.glob('artifacts/r1/lean_*.jsonl')):
        rs = rd(fn); c = collections.Counter((r['nd_ok'], r['lean_ok']) for r in rs)
        struct = sum(1 for r in rs if not r['lean_ok'] and str(r.get('lean_reason', '')).startswith('structural'))
        out[os.path.basename(fn)[5:-6]] = {'n': len(rs), 'both_accept': c[(True, True)], 'nd_only': c[(True, False)], 'lean_only': c[(False, True)], 'both_reject': c[(False, False)], 'lean_rejects_structural': struct,
                                           'disagreements': [{'nd': r['nd_reason'], 'lean': str(r.get('lean_reason'))[:100], 'proof': r['proof'][:160]} for r in rs if r['nd_ok'] != r['lean_ok']][:5]}
    return out


def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def step2(fn):
    rs = rd(fn)
    by = collections.defaultdict(list)     # (form, theorem, draw) -> rows
    for r in rs: by[(r['form'], r['theorem_name'], r['draw'])].append(r)
    per = {}                                # (form, theorem, draw) -> (greedy_ok, any_sample_ok, bin)
    for k, rows in by.items():
        g = any(r['ok'] for r in rows if r['greedy']); s = any(r['ok'] for r in rows if not r['greedy'])
        per[k] = (g, s or g, rows[0]['bin'])
    forms = sorted({k[0] for k in per})
    table = {}
    for form in forms:
        for b in BINS + ['all']:
            ks = [k for k in per if k[0] == form and (b == 'all' or per[k][2] == b)]
            if not ks: continue
            g = sum(per[k][0] for k in ks); s = sum(per[k][1] for k in ks); n = len(ks)
            table[f'{form}|{b}'] = {'n': n, 'greedy': g / n, 'greedy_ci': wilson(g, n), 'pass8': s / n, 'pass8_ci': wilson(s, n)}
    # paired delta lean - tokens per (theorem, draw), bootstrap over theorems
    thms = sorted({k[1] for k in per})
    def deltas(metric):
        d = collections.defaultdict(list)
        for t in thms:
            for dr in range(5):
                if ('lean', t, dr) in per and ('tokens', t, dr) in per:
                    d[t].append(int(per[('lean', t, dr)][metric]) - int(per[('tokens', t, dr)][metric]))
        return {t: sum(v) / len(v) for t, v in d.items() if v}
    paired = {}
    for metric, name in ((0, 'greedy'), (1, 'pass8')):
        d = deltas(metric); vals = list(d.values())
        if not vals: continue
        rng = random.Random(0); boots = []
        for _ in range(2000):
            sm = [vals[rng.randrange(len(vals))] for _ in vals]; boots.append(sum(sm) / len(sm))
        boots.sort()
        paired[name] = {'mean_delta': sum(vals) / len(vals), 'ci95': (boots[50], boots[1949]), 'n_theorems': len(vals)}
        for b in BINS:
            vb = [v for t, v in d.items() if any(per[k][2] == b for k in per if k[1] == t)]
            if vb: paired[f'{name}|{b}'] = {'mean_delta': sum(vb) / len(vb), 'n': len(vb)}
    return {'table': table, 'paired_lean_minus_tokens': paired, 'n_rows': len(rs)}


def step3():
    out = {}; per_thm = collections.defaultdict(dict)
    for m in SIZES:
        fn = f'artifacts/r1/scored_scale_{m}.jsonl'
        if not os.path.exists(fn): continue
        rs = rd(fn); by = collections.defaultdict(list)
        for r in rs: by[r['theorem_name']].append(r)
        for t, rows in by.items():
            per_thm[t][m] = (sum(r['ok'] for r in rows), len(rows), rows[0]['bin'])
    S = {json.loads(l)['theorem_name']: json.loads(l) for l in open('data/r1/scale_prompts.jsonl')} if os.path.exists('data/r1/scale_prompts.jsonl') else {}
    first = {}
    for t, d in per_thm.items():
        fs = next((m for m in SIZES if m in d and d[m][0] > 0), None)
        first[t] = {'class': S.get(t, {}).get('class'), 'first_size': fs, 'per_size': {m: f'{d[m][0]}/{d[m][1]}' for m in SIZES if m in d}, 'phase1': S.get(t, {}).get('phase1'), 'rl_round': S.get(t, {}).get('rl_round')}
    by_cls = collections.defaultdict(collections.Counter)
    for t, v in first.items(): by_cls[v['class']][v['first_size'] or 'none'] += 1
    return {'first_success': first, 'by_class': {c: dict(v) for c, v in by_cls.items()}, 'sizes_run': [m for m in SIZES if os.path.exists(f'artifacts/r1/scored_scale_{m}.jsonl')]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='artifacts/r1/summary.json'); ap.add_argument('--figs', default='figures')
    a = ap.parse_args()
    res = {'agreement': agreement()}
    if os.path.exists('artifacts/r1/scored_qwen30b.jsonl'): res['step2'] = step2('artifacts/r1/scored_qwen30b.jsonl')
    res['step3'] = step3()
    json.dump(res, open(a.out, 'w'), indent=1)
    print('== step 1 agreement')
    for k, v in res['agreement'].items():
        print(f"  {k:22s} n {v['n']:6d} both-accept {v['both_accept']:6d} nd-only {v['nd_only']:5d} lean-only {v['lean_only']:4d} both-reject {v['both_reject']:5d}")
    if 'step2' in res:
        print('== step 2 (rows', res['step2']['n_rows'], ')')
        for k, v in res['step2']['table'].items():
            print(f"  {k:24s} n {v['n']:4d} greedy {v['greedy']:.3f} [{v['greedy_ci'][0]:.2f},{v['greedy_ci'][1]:.2f}] pass@8 {v['pass8']:.3f} [{v['pass8_ci'][0]:.2f},{v['pass8_ci'][1]:.2f}]")
        for k, v in res['step2']['paired_lean_minus_tokens'].items():
            print(f"  paired {k}: {v}")
    print('== step 3', res['step3']['by_class'])
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    if 'step2' in res:
        fig, ax = plt.subplots(figsize=(9, 3.6))
        COL = {'tokens': '#2a78d6', 'lean': '#eb6834', 'english': '#7f7f7f'}
        xs = list(range(len(BINS)))
        for form in ('tokens', 'lean', 'english'):
            ys = [res['step2']['table'].get(f'{form}|{b}', {}).get('pass8') for b in BINS]
            ax.plot(xs, [y if y is not None else float('nan') for y in ys], marker='o', ms=4, color=COL[form], label=f'{form} pass@8')
            ys = [res['step2']['table'].get(f'{form}|{b}', {}).get('greedy') for b in BINS]
            ax.plot(xs, [y if y is not None else float('nan') for y in ys], marker='x', ms=4, ls='--', color=COL[form], label=f'{form} greedy')
        ax.set_xticks(xs); ax.set_xticklabels(BINS, rotation=45, fontsize=7); ax.set_ylabel('accuracy'); ax.set_ylim(0, 1)
        ax.legend(fontsize=7, frameon=False, ncol=3); ax.spines[['top', 'right']].set_visible(False); ax.grid(color='#e8e7e3', lw=0.6); ax.set_axisbelow(True)
        ax.set_title('Qwen3-Coder-30B-A3B in context (20 examples, 5 draws): accuracy by surface form and length', fontsize=9)
        fig.tight_layout(); fig.savefig(f'{a.figs}/run1_forms.png', dpi=160)
    if res['step3']['sizes_run']:
        fig, ax = plt.subplots(figsize=(7, 3.4))
        classes = sorted(res['step3']['by_class'])
        for i, c in enumerate(classes):
            cnt = res['step3']['by_class'][c]; tot = sum(cnt.values())
            bottom = 0
            for j, m in enumerate(SIZES + ['none']):
                v = cnt.get(m, 0) / tot
                ax.bar(i, v, bottom=bottom, color=plt.cm.viridis(j / len(SIZES)) if m != 'none' else '#d9d7d2', width=0.6, label=m if i == 0 else None)
                bottom += v
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes); ax.set_ylabel('fraction of theorems'); ax.legend(fontsize=6, frameon=False, ncol=2)
        ax.set_title('smallest Qwen3 size that proves the theorem in Lean (k = 16, T = 0.7)', fontsize=9); ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout(); fig.savefig(f'{a.figs}/run1_scale.png', dpi=160)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
