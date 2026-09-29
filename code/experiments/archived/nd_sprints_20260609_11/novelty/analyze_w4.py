"""W4 correlation: do formal proof measures predict ICL difficulty?

Merges the ICL ladder (per-model solve) with formal measures and reports
Spearman correlations. Defines per-target:
  icl_solve_frac = mean over models of (n_ok / samples)
  ladder_height  = capability-rank of the weakest model that solves it
                   (models ordered by empirical overall solve rate);
                   unsolved-by-all = n_models (hardest).
"""
import argparse
import json
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def spearman(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pairs)
    if n < 5:
        return None, n
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    xr = rank([p[0] for p in pairs])
    yr = rank([p[1] for p in pairs])
    mx = sum(xr) / n
    my = sum(yr) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xr, yr))
    dx = sum((a - mx) ** 2 for a in xr) ** 0.5
    dy = sum((b - my) ** 2 for b in yr) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0, n
    return num / (dx * dy), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--icl', required=True)
    ap.add_argument('--formal', required=True)
    ap.add_argument('--samples', type=int, default=2)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    icl = [json.loads(l) for l in open(args.icl) if l.strip()]
    formal = {json.loads(l)['thm']: json.loads(l) for l in open(args.formal) if l.strip()}

    # empirical model capability order
    by_model = defaultdict(lambda: [0, 0])
    for r in icl:
        by_model[r['model']][0] += r['solved']
        by_model[r['model']][1] += 1
    model_rate = {m: s / n for m, (s, n) in by_model.items()}
    order = sorted(model_rate, key=lambda m: model_rate[m])  # weakest first
    rank_of = {m: i for i, m in enumerate(order)}
    nmod = len(order)
    print('model capability (empirical solve rate):')
    for m in reversed(order):
        print(f'  {m}: {model_rate[m]:.3f}')

    # per-target aggregate
    per = defaultdict(lambda: {'solved_by': set(), 'frac': []})
    for r in icl:
        per[r['thm']]['frac'].append(r['n_ok'] / args.samples)
        if r['solved']:
            per[r['thm']]['solved_by'].add(r['model'])

    rows = []
    for thm, d in per.items():
        f = formal.get(thm)
        if not f:
            continue
        solvers = d['solved_by']
        height = min((rank_of[m] for m in solvers), default=nmod)
        rows.append({
            'thm': thm, 'source': f.get('source'),
            'icl_frac': sum(d['frac']) / len(d['frac']),
            'ladder_height': height,
            'min_lines': f.get('min_lines'), 'genus': f.get('genus'),
            'mdl_bits': f.get('mdl_bits'), 'lemma_value': f.get('lemma_value'),
            'gen_lines': f.get('gen_lines'),
        })

    def col(k):
        return [r[k] for r in rows]
    print(f'\n{len(rows)} targets. Spearman vs ICL difficulty:')
    print(f'{"measure":14} {"rho(icl_frac)":>14} {"rho(ladder_ht)":>15}')
    # icl_frac: higher = easier, so negate to make "difficulty"
    icl_diff = [-x for x in col('icl_frac')]
    for meas in ['min_lines', 'genus', 'mdl_bits', 'lemma_value', 'gen_lines']:
        r1, n1 = spearman(col(meas), icl_diff)
        r2, n2 = spearman(col(meas), col('ladder_height'))
        s1 = f'{r1:+.3f}' if r1 is not None else '  n/a'
        s2 = f'{r2:+.3f}' if r2 is not None else '  n/a'
        print(f'{meas:14} {s1:>14} {s2:>15}  (n={n1})')

    # ladder height distribution
    from collections import Counter
    print('\nladder-height histogram (0=easiest model solves .. {}=none):'.format(nmod))
    print(' ', dict(sorted(Counter(col('ladder_height')).items())))
    # mean formal measure by ladder height (the "do harder theorems have higher
    # bits/genus" check)
    print('\nmean measures by ladder height:')
    byh = defaultdict(list)
    for r in rows:
        byh[r['ladder_height']].append(r)
    print(f'{"height":>6} {"n":>4} {"min_lines":>10} {"bits":>7} {"genus":>6} {"lemma":>6}')
    for h in sorted(byh):
        g = byh[h]
        def mean(k):
            vs = [x[k] for x in g if x.get(k) is not None]
            return sum(vs) / len(vs) if vs else float('nan')
        print(f'{h:>6} {len(g):>4} {mean("min_lines"):>10.1f} {mean("mdl_bits"):>7.0f} {mean("genus"):>6.1f} {mean("lemma_value"):>6.1f}')

    if args.out:
        with open(args.out, 'w') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')


if __name__ == '__main__':
    main()
