#!/usr/bin/env python3
"""Build the bound-16 input from the current bound-10 labels: unresolved records not yet in labels/gen_b16.jsonl, ordered
(1) search exhausted at bound 10 (L_true >= 11 certain within the space) before 10 s timeouts, (2) n_prem >= 1 before 0,
(3) gen_lines >= 12 first, random (seed 24004) within."""
import json, random, os
D = '/scratch/cowifornia/nd-rl/ab2/data'
done = set()
if os.path.exists(f'{D}/labels/gen_b16.jsonl'):
    done = {json.loads(l)['name'] for l in open(f'{D}/labels/gen_b16.jsonl') if l.strip()}
U = {}
for l in open(f'{D}/labels/gen_b10.jsonl'):
    x = json.loads(l)
    if x['min_lines_ub'] is None and x['name'] not in done:
        U[x['name']] = x
recs = [json.loads(l) for l in open(f'{D}/raw/pool_gen.jsonl')]
recs = [r for r in recs if r['name'] in U]
random.Random(24004).shuffle(recs)
recs.sort(key=lambda r: (bool(U[r['name']].get('timeout')), r['n_prem'] == 0, r['n_lines'] < 12))
with open(f'{D}/raw/gen_u.jsonl', 'w') as f:
    for r in recs:
        f.write(json.dumps(r) + '\n')
print(len(recs), 'exhausted', sum(not U[r['name']].get('timeout') for r in recs))
