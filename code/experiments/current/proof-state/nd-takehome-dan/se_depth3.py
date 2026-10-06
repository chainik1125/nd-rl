#!/usr/bin/env python3
"""Held-out greedy on the depth-3 slice (`pat.depth3` of data/p2/heldout.jsonl), for every arm and for C0 (on file).
  python3 se_depth3.py > artifacts/se/heldout_depth3.json"""
import glob, json, os, subprocess, sys
pat = {}
for l in open('data/p2/heldout.jsonl'):
    r = json.loads(l); pat[r['name']] = bool((r.get('pat') or {}).get('depth3'))
def rows(lines):
    return [json.loads(l) for l in lines if l.strip()]
out = {}
srcs = [(os.path.basename(f)[8:-6], f, rows(open(f))) for f in sorted(glob.glob('artifacts/se/heldout_*.jsonl'))]
for s in (0, 1):
    p = f'artifacts/dsg/heldout2_c0_s{s}.jsonl'
    srcs.append((f'C0_s{s} (Lean AND nd_verify, on file)', p, rows(subprocess.check_output(['git', 'show', f'HEAD:{p}']).decode().splitlines())))
for name, fn, rs in srcs:
    d3 = [r for r in rs if pat.get(r['name'])]
    out[name] = {'source': fn, 'depth3_solved': sum(r['solved'] for r in d3), 'depth3_n': len(d3),
                 'depth3_rate': sum(r['solved'] for r in d3) / max(1, len(d3)),
                 'other_rate': sum(r['solved'] for r in rs if not pat.get(r['name'])) / max(1, sum(1 for r in rs if not pat.get(r['name'])))}
json.dump(out, sys.stdout, indent=1)
