#!/usr/bin/env python3
"""Acceptance test 5, part 2 (pitfall 2) on REAL sampled data: hindsight relabels against theorems the gate never saw.

Reads $LEAN_GATE_DUMP from the test-5 round (every distinct checked sample with its Lean verdict), forms the hindsight
relabel of each one (the theorem it would prove if its last formula were the conclusion), keeps only those whose rewritten
theorem DIFFERS from the prompted one -- so no registered verdict can apply -- and judges them in one batched Lean run.
Writes artifacts/lj/t5_relabel.json with the counts and three accepted examples in Lean, for hand checking.
"""
import os, sys, json, collections
sys.path.insert(0, '/workspace/nd-takehome')
import lean_judge, nd2lean
_src = open('/workspace/nd-takehome/expert_iter.py').read()
_ns = {'judge_many': lean_judge.judge_many}
exec(_src[_src.index('def relabel_candidate'):_src.index('def relabel_batch')], _ns)
RELABEL = _ns['relabel_candidate']

src = sys.argv[1] if len(sys.argv) > 1 else 'artifacts/lj/t5_dump.jsonl'
recs = [json.loads(l) for l in open(src) if l.strip()]
pairs, cand = [], []
for r in recs:
    c = None
    try:
        c = RELABEL(r['prompt'], r['nd'])
    except Exception:
        c = None
    if c and c[0] != r['prompt']:
        pairs.append((r['prompt'], r['nd'])); cand.append(c)
res = lean_judge.judge_many([(c[0], nd) for c, (p, nd) in zip(cand, pairs)])
ok = [(c, p, nd, nl) for c, (p, nd), (o, rs, nl) in zip(cand, pairs, res) if o]
ex = []
for c, p, nd, nl in ok[:3]:
    ex.append({'prompted_theorem': p, 'relabelled_theorem': c[1], 'relabelled_prompt': c[0], 'nd_proof': nd,
               'n_lines': nl, 'lean_source': nd2lean.translate(c[0], nd, require_all_pr=False)})
out = {'dump': src, 'dump_records': len(recs), 'relabel_candidates_with_a_new_theorem': len(pairs),
       'accepted': len(ok), 'rejected': len(pairs) - len(ok), 'examples': ex,
       'judge_stats': lean_judge.stats()}
json.dump(out, open('artifacts/lj/t5_relabel.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'examples'}, indent=1))
if ex:
    print('--- one accepted hindsight relabel, in Lean ---')
    print(ex[0]['relabelled_theorem']); print(ex[0]['lean_source'])
