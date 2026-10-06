#!/usr/bin/env python3
"""Premise check for run noise-floor: are P1-P4 the same distribution?  Compares every shape-table quantity
across the four null sets and against the control's published shape table (data/dsg/README.md, C0 column),
plus assembly fill, depth-3 exclusion, evaluation-pool overlap, render check and pairwise renaming-class
overlap between the four sets.  Any pool that differs from the other three by more than sampling noise is an
ARM, not a replicate (pre-registration).   python3 nf_premise.py [--md]"""
import json, os, sys, glob, collections, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

POOLS = ('p1', 'p2', 'p3', 'p4')
D = 'artifacts/nf'
jl = lambda fn: json.load(open(fn)) if os.path.exists(fn) else None
sh = {p: list(jl(f'{D}/shape_{p}.json').values())[0] for p in POOLS}
asm = {p: jl(f'{D}/assemble_{p}.json') for p in POOLS}
ov = {p: jl(f'{D}/overlap_{p}.json') for p in POOLS}
rn = {p: jl(f'{D}/render_{p}.json') for p in POOLS}
N = 155000
# the control's published shape table (data/dsg/README.md, C0 = the lean-format a1 set, 155,000 proofs)
C0 = {'box0': 53.2, 'box1': 35.4, 'box2': 11.4, 'ORE': 1.46, 'mean_prem': 1.41, 'contra': 6.06, 'AS': 46.8,
      'IMPI': 36.4, 'IMPE': 40.2, 'DN': 8.3, 'NEGI': 9.9, 'NEGE': 12.0, 'ANDI': 22.9, 'R': 2.8}
RULES = ('AS', 'R', 'ANDI', 'ANDE1', 'ANDE2', 'ORI1', 'ORI2', 'ORE', 'IMPI', 'IMPE', 'NEGI', 'NEGE', 'DN', 'BOTE')

def row(name, vals, ctrl=None, fmt='{:.2f}'):
    v = [x for x in vals if x is not None]
    spread = (max(v) - min(v)) if v else 0.0
    # binomial sampling sd of a share (in the same units) at n = 155,000, for the "is this sampling noise?" test
    return (name, [fmt.format(x) for x in vals], fmt.format(spread),
            (fmt.format(ctrl) if ctrl is not None else '—'))

out = []
out.append(row('records', [sh[p]['n'] for p in POOLS], 155000, '{:.0f}'))
for i in range(2, 7):
    out.append(row(f'length {i}', [sh[p]['len_hist'][str(i)] for p in POOLS], 31000, '{:.0f}'))
for b in range(4):
    out.append(row(f'box depth {b} (%)', [100 * sh[p]['box_depth_hist'].get(str(b), 0) / N for p in POOLS],
                   C0.get(f'box{b}')))
for r in RULES:
    out.append(row(f'proofs with `{r}` (%)', [100 * sh[p]['rule_share'][r] for p in POOLS], C0.get(r)))
out.append(row('mean premises', [sh[p]['mean_n_prem'] for p in POOLS], C0['mean_prem']))
out.append(row('contradictory-premise theorems (%)', [100 * sh[p]['contra_prem_share'] for p in POOLS], C0['contra']))
out.append(row('reductio proofs', [sh[p]['pattern_counts'].get('reductio', 0) for p in POOLS], None, '{:.0f}'))
out.append(row('derived-`ORE` proofs', [sh[p]['pattern_counts'].get('derived_ore', 0) for p in POOLS], None, '{:.0f}'))
out.append(row('depth-3 proofs (excluded by design)', [sh[p]['pattern_counts'].get('depth3', 0) for p in POOLS], 0, '{:.0f}'))
out.append(row('mean ND tokens', [sh[p]['mean_nd_tokens'] for p in POOLS], None))
out.append(row('mean Lean tokens (`lean_seq`)', [sh[p]['mean_lean_tokens'] for p in POOLS], None))
for k in ('mean_term_size', 'mean_thm_size'):
    if k in sh['p1']:
        out.append(row(k.replace('_', ' '), [sh[p][k] for p in POOLS], None))
out.append(row('fill fraction (%)', [100 * asm[p]['fill_frac'] for p in POOLS], 0.0))

# pairwise renaming-class overlap between the four sets
sys.path.insert(0, '.')
from gen import canon_key
keys = {}
for p in POOLS:
    fn = f'data/nf/train_{p}.jsonl'
    keys[p] = {canon_key(json.loads(l)['thm']) for l in open(fn)} if os.path.exists(fn) else set()
pair = {}
for i, a in enumerate(POOLS):
    for b in POOLS[i + 1:]:
        if keys[a] and keys[b]:
            pair[f'{a}-{b}'] = len(keys[a] & keys[b]) / min(len(keys[a]), len(keys[b]))

res = {'shape': [{'quantity': q, 'p1': v[0], 'p2': v[1], 'p3': v[2], 'p4': v[3], 'spread': s, 'C0_published': c}
                 for q, v, s, c in out],
       'pairwise_class_overlap': pair,
       'distinct_classes_per_set': {p: len(keys[p]) for p in POOLS},
       'overlap_with_eval_pools': {p: ov[p] for p in POOLS},
       'render_check': {p: rn[p] for p in POOLS},
       'assemble': {p: asm[p] for p in POOLS}}
json.dump(res, open(f'{D}/premise.json', 'w'), indent=1)

print('| quantity | P1 | P2 | P3 | P4 | max − min | C0 published |\n|---|---|---|---|---|---|---|')
for q, v, s, c in out:
    print(f'| {q} | ' + ' | '.join(v) + f' | {s} | {c} |')
print('\npairwise renaming-class overlap (|A ∩ B| / min(|A|,|B|)):',
      ', '.join(f'{k} {100 * v:.1f} %' for k, v in pair.items()))
print('distinct classes per set:', {p: len(keys[p]) for p in POOLS})
for p in POOLS:
    o = ov[p]
    print(f'{p}: overlap with the nine evaluation / ladder pools ->', json.dumps(o)[:260] if o else None)
    print(f'{p}: render check ->', json.dumps(rn[p])[:220] if rn[p] else None)
