#!/usr/bin/env python3
"""Run 5: build a target / transfer pool of theorems that REQUIRE a pattern (necessity.py labels).

  python required_pool.py --nec data/p2/run5_reductio_nec.jsonl --pattern reductio --min_ub 7 --n_targets 300 --n_transfer 150 \
      --strata schema --out data/p2/targets_reductio_req.jsonl --out_transfer data/p2/transfer_reductio_req.jsonl --exclude ...
Keeps records with requires == True, min_lines_ub >= --min_ub (beyond the pretraining cap), oracle_ok, no timeouts; drops
classes in --exclude files and validation-36; allocates targets / transfer proportionally per stratum (schema or generating
length); n_lines := min_lines_ub (the length of a verifier-valid pattern proof found by minlen).
"""
import argparse, json, random, sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen import canon_key

ap = argparse.ArgumentParser()
ap.add_argument('--nec', nargs='+', required=True); ap.add_argument('--pattern', required=True)
ap.add_argument('--min_ub', type=int, default=7); ap.add_argument('--n_targets', type=int, default=300); ap.add_argument('--n_transfer', type=int, default=150)
ap.add_argument('--strata', default='schema', help="record field to stratify on ('schema' or 'gen_lines')")
ap.add_argument('--exclude', nargs='*', default=[]); ap.add_argument('--out', required=True); ap.add_argument('--out_transfer', required=True)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--prefix', default='req')
ap.add_argument('--mode', default='requires', choices=['requires', 'uses', 'gen'], help="requires: oracle-required; uses: the shortest found proof contains the pattern; gen: the generating proof contains it (pat2 field) and min_lines_ub >= min_ub")
ap.add_argument('--gen_pattern', default=None, help='pattern key in pat2 for --mode gen')
ap.add_argument('--gen_from', default=None, help='candidate jsonl carrying pat2 / proof (looked up by name) for --mode gen')
a = ap.parse_args()
rng = random.Random(a.seed)
excl = {canon_key(json.loads(l)['thm'].strip()) for l in open('targets/validation_36.jsonl')}
for fn in a.exclude:
    for l in open(fn):
        if l.strip():
            excl.add(json.loads(l)['key'])
recs, seen = [], set(); st = collections.Counter()
gen = {}
if a.gen_from:
    for l in open(a.gen_from):
        if l.strip():
            g = json.loads(l); gen[g['name']] = g
for fn in a.nec:
    for l in open(fn):
        r = json.loads(l); st['n'] += 1
        if a.mode == 'requires' and not r['requires']: st['not_required'] += 1; continue
        if a.mode == 'uses' and not r.get('uses'): st['not_uses'] += 1; continue
        if a.mode == 'gen':
            g = gen.get(r['name'], r); r['pat2'] = g.get('pat2'); r['proof_gen'] = g.get('proof'); r['gen_lines'] = r.get('gen_lines') or g.get('n_lines')
            if not (r.get('pat2') or {}).get(a.gen_pattern or a.pattern): st['not_gen'] += 1; continue
        if r['min_lines_ub'] is None or r['min_lines_ub'] < a.min_ub: st['too_short_or_unreachable'] += 1; continue
        if not r['oracle_ok'] or r['timeout'] or r['r_timeout']: st['oracle_flag'] += 1; continue
        key = r.get('key') or canon_key(r['thm'].strip())
        if key in excl: st['excluded'] += 1; continue
        if key in seen: st['dup'] += 1; continue
        seen.add(key); r['key'] = key; recs.append(r)
print('filter stats', dict(st), 'eligible', len(recs))
by = collections.defaultdict(list)
for r in recs:
    by[str(r.get(a.strata))].append(r)
tot = len(recs); targets, transfer = [], []
for s in sorted(by):
    rs = by[s]; rng.shuffle(rs)
    nt = round(a.n_targets * len(rs) / tot); nx = round(a.n_transfer * len(rs) / tot)
    targets += rs[:nt]; transfer += rs[nt:nt + nx]
used = {r['key'] for r in targets + transfer}
left = [r for r in recs if r['key'] not in used]; rng.shuffle(left)     # rounding: top up to the exact sizes
while len(targets) < a.n_targets and left: targets.append(left.pop())
while len(transfer) < a.n_transfer and left: transfer.append(left.pop())
rng.shuffle(targets); rng.shuffle(transfer)
targets, transfer = targets[:a.n_targets], transfer[:a.n_transfer]
for tag, rows, fn in (('targets', targets, a.out), ('transfer', transfer, a.out_transfer)):
    with open(fn, 'w') as f:
        for i, r in enumerate(rows):
            o = {'name': f'{tag}_{a.pattern}_{a.prefix}_{i}', 'thm': r['thm'], 'key': r['key'], 'prompt': r['prompt'], 'n_lines': r['min_lines_ub'],
                 'n_prem': r.get('n_prem', r['prompt'].split(' SEQ ')[0].count(' , ') + (0 if r['prompt'].startswith('THM SEQ') else 1)),
                 'schema': r.get('schema'), 'source': r.get('source', 'generator'), 'gen_lines': r.get('gen_lines'), 'min_lines_ub': r['min_lines_ub'],
                 'r_min_lines_ub': r['r_min_lines_ub'], 'requires': r['requires'], 'uses': r.get('uses'), 'mode': a.mode, 'oracle_proof': r['proof'], 'proof_pat': r['proof_pat'], 'gen_proof': r.get('proof_gen'),
                 'pat': {'derived_ore': a.pattern == 'derived_ore_strict', 'reductio': a.pattern == 'reductio', 'depth3': False, 'derived_ore_strict': a.pattern == 'derived_ore_strict'}}
            f.write(json.dumps(o) + '\n')
    print(tag, len(rows), 'min_lines_ub', dict(sorted(collections.Counter(r['min_lines_ub'] for r in rows).items())), 'strata', dict(sorted(collections.Counter(str(r.get(a.strata)) for r in rows).items())), '->', fn)
