#!/usr/bin/env python3
"""The 1,195 records acceptance test 2 could not judge: run `efficiency`'s found files store the literal Lean text in
their `proof` field, not an ND proof, so the lean_seq/ND judge cannot translate them (`nd2lean: QED`).  They belong to
`lean_check`'s domain — the free-form-Lean judge — which is what run `efficiency` used.  Check them there, so that
"every sample the old judge counted is counted now" is closed for these too."""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean_check
from nd2lean import parse_prompt, lf

recs = [json.loads(l) for l in open('artifacts/lj/t2_judge.rejected.jsonl')]
srcs = []
for r in recs:
    prem, concl = parse_prompt(r['prompt'])
    hyps = ' '.join(f'(h{j+1} : {lf(p)})' for j, p in enumerate(prem))
    srcs.append(f'theorem t (P Q R S : Prop) {hyps} : {lf(concl)} := {r["nd"]}')
res, wall, cpu = lean_check.check(srcs)
ok = sum(1 for x in res if x['ok'])
sizes = sorted(x['size'] for x in res if x['ok'])
rec = {'n': len(recs), 'accepted_by_lean_check': ok, 'rejected': len(recs) - ok,
       'reject_reasons': dict(collections.Counter(x['reason'][:60] for x in res if not x['ok']).most_common(8)),
       'term_size_min_median_max': [sizes[0], sizes[len(sizes)//2], sizes[-1]] if sizes else None,
       'wall_s': wall, 'proc_s': cpu,
       'note': 'these records are literal Lean text stored in a found*.jsonl `proof` field by run efficiency; the '
               'lean_seq/ND judge cannot translate them, the free-form judge lean_check is the right one'}
json.dump(rec, open('artifacts/lj/t2_freeform.json', 'w'), indent=1)
print(json.dumps(rec, indent=1))
