#!/usr/bin/env python3
"""Checker of record for every counted proof of an arm: the UNMODIFIED nd2lean.py --check (official translation + Lean) and
nd_verify on (i) the dial's found_4.jsonl (EI and frozen), (ii) the ladder's found_8.jsonl / found_transfer_8.jsonl (T1 and frozen),
(iii) every distinct counted proof of the three coverage runs. Agreement counts -> artifacts/dsg/record_<arm>.json.
  python3 pod/dsg/record.py <c0|g1|g2> [seed]        (a seed restricts the pass to that seed; resume phase runs one seed per pod)"""
import sys, os, glob, json, subprocess, collections
arm = sys.argv[1]
seeds = (int(sys.argv[2]),) if len(sys.argv) > 2 else (0, 1)
sfx = f'_s{sys.argv[2]}' if len(sys.argv) > 2 else ''
out = {}
srcs = []
for s in seeds:
    for a in ('ei', 'frozen'):
        d = f'artifacts/dsg/{a}_d3_{arm}_s{s}'
        if os.path.isdir(d):
            srcs += [(f'{a}_d3_s{s}_found', f'{d}/found_4.jsonl'), (f'{a}_d3_s{s}_found_transfer', f'{d}/found_transfer_4.jsonl')]
    for a in ('la_T1', 'la_frozen'):
        d = f'artifacts/dsg/{a}_{arm}_s{s}'
        if os.path.isdir(d):
            srcs += [(f'{a}_s{s}_found', f'{d}/found_8.jsonl'), (f'{a}_s{s}_found_transfer', f'{d}/found_transfer_8.jsonl')]
    for pool in ('d3', 'req8', 'red'):
        fn = f'artifacts/dsg/cov_{pool}_{arm}_s{s}.s0.jsonl'
        if os.path.exists(fn):
            flat = f'artifacts/dsg/cov_{pool}_{arm}_s{s}.counted.jsonl'
            with open(flat, 'w') as f:
                for l in open(fn):
                    r = json.loads(l)
                    for p in r['proofs']:
                        f.write(json.dumps({'name': r['name'], 'prompt': r['prompt'], 'proof': p['proof'], 'count': p['count']}) + '\n')
            srcs.append((f'cov_{pool}_s{s}', flat))
for tag, src in srcs:
    if not os.path.exists(src):
        out[tag] = {'source': src, 'missing': True}; continue
    rep = src.replace('.jsonl', '.record.jsonl')
    p = subprocess.run(['python3', 'nd2lean.py', '--check', src, '--out', rep], capture_output=True, text=True)
    rows = [json.loads(l) for l in open(rep)] if os.path.exists(rep) else []
    c = collections.Counter((r['nd_ok'], r['lean_ok']) for r in rows)
    out[tag] = {'source': src, 'n': len(rows), 'both_accept': c[(True, True)], 'nd_ok_lean_rej': c[(True, False)], 'nd_rej_lean_ok': c[(False, True)],
                'both_reject': c[(False, False)], 'stderr_tail': p.stderr[-300:]}
    print(tag, len(rows), 'both accept', out[tag]['both_accept'], 'disagree', out[tag]['nd_ok_lean_rej'] + out[tag]['nd_rej_lean_ok'], flush=True)
json.dump(out, open(f'artifacts/dsg/record_{arm}{sfx}.json', 'w'), indent=1)
