#!/usr/bin/env python3
"""Acceptance test 5, part 3: the OLD gate's count vs the NEW judge's count on exactly the same samples.

Reads the test-5 gate dump (every distinct checked (prompt, literal text, ND string, Lean verdict) of the round) and
recomputes, off the pod, what the OLD gate would have counted: Lean AND nd_verify.  The excess of the new judge must be
exactly the Lean-only class, and every excess sample is classified.  `nd_verify` is imported here to reproduce the OLD
count; it judges nothing.
"""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nd_verify import verify_text as ndv
from lj_regress import classify_leanonly

dump = sys.argv[1] if len(sys.argv) > 1 else 'artifacts/lj/t5_dump.jsonl'
out = sys.argv[2] if len(sys.argv) > 2 else 'artifacts/lj/t5_compare.json'
recs = [json.loads(l) for l in open(dump) if l.strip()]
tab = collections.Counter(); excess = []
for r in recs:
    nd_ok = bool(ndv(r['prompt'] + ' ' + r['nd'])[0])
    lean_ok = bool(r['lean_ok'])
    tab[(nd_ok, lean_ok)] += 1
    if lean_ok and not nd_ok:
        excess.append(r)
old = tab[(True, True)]
new = tab[(True, True)] + tab[(False, True)]
cls = collections.Counter(classify_leanonly(r['prompt'], r['nd']) for r in excess)
rec = {'dump': dump, 'distinct_checked': len(recs),
       'old_gate_counted_lean_and_nd_verify': old, 'new_judge_counted_lean_only': new,
       'excess': new - old, 'losses_old_counted_new_did_not': tab[(True, False)],
       'table': {f'nd={k[0]},lean={k[1]}': v for k, v in sorted(tab.items())},
       'excess_classes': dict(cls.most_common()),
       'excess_examples': [{'prompt': r['prompt'], 'nd': r['nd'], 'lean_text': r['lean_text'],
                            'class': classify_leanonly(r['prompt'], r['nd'])} for r in excess[:5]]}
json.dump(rec, open(out, 'w'), indent=1)
print(json.dumps({k: v for k, v in rec.items() if k != 'excess_examples'}, indent=1))
print('->', out)
