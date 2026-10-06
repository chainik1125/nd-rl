#!/usr/bin/env python3
"""Merge phase-2 collector parts into one pool + labels file per set (theorem order of the set file kept) and a
combined summary.   python merge_parts.py rr600 96b part1.jsonl part2.jsonl ..."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pool_lib as PL
s, k, parts = sys.argv[1], sys.argv[2], sys.argv[3:]
rows, labs, summ = {}, {}, []
for p in parts:
    for r in PL.read_jsonl(p):
        rows[r['name']] = r
    for r in PL.read_jsonl(p[:-6] + '.labels.jsonl'):
        labs[r['name']] = r
    summ.append(json.load(open(p[:-6] + '.summary.json')))
order = [t['name'] for t in PL.load_set(s) if t['name'] in rows]
PL.write_jsonl(PL.pool_path(s, k), [rows[n] for n in order])
PL.write_jsonl(PL.labels_path(s, k), [labs[n] for n in order])
att = sum(x['attempts'] for x in summ)
w = lambda key: sum(x[key] * x['attempts'] for x in summ) / att
S = dict(set=s, file=os.path.basename(PL.pool_path(s, k)), n_theorems=len(order), k=summ[0]['k'],
         seed=[x['seed'] for x in summ], attempts=att, finished_rate=w('finished_rate'), validity_rate=w('validity_rate'),
         p_valid_given_finished=sum(x['p_valid_given_finished'] * x['finished_rate'] * x['attempts'] for x in summ) / max(1, sum(x['finished_rate'] * x['attempts'] for x in summ)),
         pass_at_k=sum(any(labs[n]['valid']) for n in order) / len(order), wall_s=sum(x['wall_s'] for x in summ),
         sample_wall_s=sum(x['sample_wall_s'] for x in summ), lean_s=sum(x['lean_s'] for x in summ),
         gen_tokens=sum(x['gen_tokens'] for x in summ), prefill_tokens=sum(x['prefill_tokens'] for x in summ),
         ends={}, parts=[os.path.basename(p) for p in parts], note='phase-2 extra pool; wall_s summed over parallel parts')
for x in summ:
    for e, v in x['ends'].items():
        S['ends'][e] = S['ends'].get(e, 0) + v
json.dump(S, open(PL.pool_path(s, k)[:-6] + '.summary.json', 'w'), indent=1)
print(json.dumps({kk: S[kk] for kk in ('set', 'n_theorems', 'attempts', 'finished_rate', 'validity_rate', 'p_valid_given_finished')}))
