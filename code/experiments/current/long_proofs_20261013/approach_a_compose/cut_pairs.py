"""Cut-compose tiny theorems from the ladder output: (A |- B) + (B |- C) -> (A |- C), same atoms so the union of
subformulas stays small. Ranks by (sum of exact lengths, union subformula count). Writes candidate prompts."""
import json, sys, collections, itertools
sys.path.insert(0, '..')
import certify
from certify import truth_table, fstr
from minlen import parse_thm, subformulas
recs = {}
for p in sys.argv[2:]:
    for l in open(p):
        r = json.loads(l)
        if r.get('min_body') and r['n_prem'] == 1:
            recs[r['prompt']] = r
by_prem = collections.defaultdict(list)
parsed = {}
for pr, r in recs.items():
    prem, concl = parse_thm(pr); parsed[pr] = (prem[0], concl)
    by_prem[prem[0]].append(pr)
out = []
for p1, (a, b) in parsed.items():
    for p2 in by_prem.get(b, []):
        _, c = parsed[p2]
        if c == a or c == b:
            continue
        sat, valid = truth_table([a], c)
        if not valid or not sat:
            continue
        sub = set(); subformulas(a, sub); subformulas(c, sub)
        n_sub = len(sub)
        out.append((recs[p1]['min_body'] + recs[p2]['min_body'], n_sub, p1, p2, f'THM {fstr(a)} SEQ {fstr(c)} PRF'))
out.sort(key=lambda t: (-t[0], t[1]))
seen = set(); n = 0
with open(sys.argv[1], 'w') as fo:
    for s, ns, p1, p2, pr in out:
        if pr in seen: continue
        seen.add(pr); n += 1
        fo.write(json.dumps({'prompt': pr, 'sum_len': s, 'n_sub': ns, 'via': [p1, p2]}) + '\n')
print(n, 'cut pairs; top:')
for s, ns, p1, p2, pr in out[:25]:
    print(s, ns, pr, '  via', p1, '|', p2)
print('sum_len>=20 by n_sub:', collections.Counter(ns for s, ns, *_ in out if s >= 20))
