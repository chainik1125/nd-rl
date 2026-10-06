#!/usr/bin/env python3
"""Checker of record for the held-out greedy proofs: a random sample per cell through the UNMODIFIED
nd2lean.py --check (official translation + Lean 4.34) and nd_verify.  The in-loop Lean gate already
requires both checkers for every sample, so this is an independent re-check of what was counted.
  python3 pod/nf/record_heldout.py [n_per_cell]   -> artifacts/nf/record_heldout.json"""
import sys, os, glob, json, random, subprocess, collections
n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 200
rng = random.Random(0)
src = 'artifacts/nf/heldout_all.counted.jsonl'
per_cell = {}
with open(src, 'w') as fo:
    for fn in sorted(glob.glob('artifacts/nf/heldout_p*_s*.jsonl')):
        cell = os.path.basename(fn)[len('heldout_'):-len('.jsonl')]
        rows = [json.loads(l) for l in open(fn)]
        got = [(r['prompt'], p) for r in rows if r['solved'] for p in r['proofs']]
        pick = rng.sample(got, min(n_per, len(got)))
        per_cell[cell] = {'solved_proofs': len(got), 'sampled': len(pick)}
        for pr, p in pick:
            fo.write(json.dumps({'name': cell, 'prompt': pr, 'proof': p}) + '\n')
rep = src.replace('.jsonl', '.record.jsonl')
r = subprocess.run(['python3', 'nd2lean.py', '--check', src, '--out', rep], capture_output=True, text=True)
rows = [json.loads(l) for l in open(rep)] if os.path.exists(rep) else []
c = collections.Counter((x['nd_ok'], x['lean_ok']) for x in rows)
out = {'n_cells': len(per_cell), 'n_per_cell': n_per, 'n_checked': len(rows),
       'both_accept': c[(True, True)], 'nd_ok_lean_rej': c[(True, False)],
       'nd_rej_lean_ok': c[(False, True)], 'both_reject': c[(False, False)],
       'per_cell': per_cell, 'stderr_tail': r.stderr[-300:]}
json.dump(out, open('artifacts/nf/record_heldout.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'per_cell'}))
