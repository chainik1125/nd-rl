#!/usr/bin/env python3
"""proof-state: tabulate validation-36 results from run_v36.sh outputs.

Per checkpoint: greedy solved, pass@32 (unbiased estimate from the k=256 draw) and pass@256, overall and for the
`<=6` / `>6` bins, plus the theorems each arm solves.  Writes results.md and results.json next to the outputs.
"""
import json, math, os, re, sys, collections

RUN = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else '~/runs/proof-state')
OUT = os.path.join(RUN, 'out')
IN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nd-takehome-dan/targets/validation_36.jsonl')
K = 256
bins = {json.loads(l)['name']: json.loads(l)['bin'] for l in open(IN) if l.strip()}
mlub = {json.loads(l)['name']: json.loads(l)['min_lines_ub'] for l in open(IN) if l.strip()}


def pass_at(n, c, k):
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def load(name, mode):
    fn = os.path.join(OUT, f'{name}_{mode}.jsonl')
    return {r['name']: r for r in map(json.loads, open(fn))} if os.path.exists(fn) else None


names = sorted({re.sub(r'_(greedy|k\d+)\.jsonl$', '', f) for f in os.listdir(OUT) if f.endswith('.jsonl') and 'dump' not in f})
order = lambda n: (['stage1', 'T1'].index(n.split('_')[0]), ['C0', 'S', 'SH', 'SN'].index(n.split('_')[1]), n)
res, solved_by = {}, collections.defaultdict(set)
for n in sorted(names, key=order):
    g, s = load(n, 'greedy'), load(n, f'k{K}')
    row = {}
    for part, keep in (('all', lambda t: True), ('<=6', lambda t: bins[t] == '<=6'), ('>6', lambda t: bins[t] == '>6')):
        ts = [t for t in bins if keep(t)]
        row[part] = {'n': len(ts),
                     'greedy': sum(g[t]['solved'] for t in ts) if g else None,
                     'pass32': round(sum(pass_at(s[t]['n_tried'], s[t]['n_ok'], 32) for t in ts), 2) if s else None,
                     f'pass{K}': sum(s[t]['solved'] for t in ts) if s else None}
    row['solved_k'] = sorted(t for t in bins if s and s[t]['solved'])
    row['solved_greedy'] = sorted(t for t in bins if g and g[t]['solved'])
    res[n] = row
    solved_by[n.rsplit('_s', 1)[0]] |= set(row['solved_k'])

json.dump(res, open(os.path.join(RUN, 'results.json'), 'w'), indent=1)
L = [f'# validation-36 (Lean 4.34.0 alone), greedy / pass@32 / pass@{K} at T 0.8\n',
     f'| checkpoint | all (36) | <=6 (12) | >6 (24) |', '|---|---|---|---|']
for n, r in res.items():
    f = lambda p: f"{r[p]['greedy']} / {r[p]['pass32']} / {r[p][f'pass{K}']}"
    L.append(f'| {n} | {f("all")} | {f("<=6")} | {f(">6")} |')
L.append(f'\n## >6-bin theorems solved at pass@{K} (union over seeds)\n')
for arm, ts in solved_by.items():
    big = sorted((t for t in ts if bins[t] == '>6'), key=lambda t: mlub[t])
    L.append(f'- **{arm}** ({len(big)}): ' + ', '.join(f'{t} ({mlub[t]})' for t in big))
open(os.path.join(RUN, 'results.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
