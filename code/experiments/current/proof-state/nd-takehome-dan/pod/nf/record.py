#!/usr/bin/env python3
"""Checker of record for every counted proof on this pod: the UNMODIFIED nd2lean.py --check (official translation +
Lean 4.34) and nd_verify, over (i) every ladder run's found_8.jsonl / found_transfer_8.jsonl and (ii) every distinct
counted proof of every coverage run.  Agreement counts -> artifacts/nf/record_<host tag>.json.
  python3 pod/nf/record.py [tag]"""
import sys, os, glob, json, subprocess, collections
tag = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('NF_POD', 'pod')
out = {}
srcs = []
for d in sorted(glob.glob('artifacts/nf/la_*')):
    if not os.path.isdir(d):
        continue
    n = os.path.basename(d)
    for kind in ('found_8.jsonl', 'found_transfer_8.jsonl'):
        if os.path.exists(f'{d}/{kind}'):
            srcs.append((f'{n}_{kind[:-6]}', f'{d}/{kind}'))
for fn in sorted(glob.glob('artifacts/nf/cov_*.s0.jsonl')):
    n = os.path.basename(fn)[:-9]
    flat = f'artifacts/nf/{n}.counted.jsonl'
    with open(flat, 'w') as f:
        for l in open(fn):
            r = json.loads(l)
            for p in r['proofs']:
                f.write(json.dumps({'name': r['name'], 'prompt': r['prompt'], 'proof': p['proof'], 'count': p['count']}) + '\n')
    srcs.append((n, flat))
for name, src in srcs:
    rep = src.replace('.jsonl', '.record.jsonl')
    p = subprocess.run(['python3', 'nd2lean.py', '--check', src, '--out', rep], capture_output=True, text=True)
    rows = [json.loads(l) for l in open(rep)] if os.path.exists(rep) else []
    c = collections.Counter((r['nd_ok'], r['lean_ok']) for r in rows)
    out[name] = {'source': src, 'n': len(rows), 'both_accept': c[(True, True)], 'nd_ok_lean_rej': c[(True, False)],
                 'nd_rej_lean_ok': c[(False, True)], 'both_reject': c[(False, False)], 'stderr_tail': p.stderr[-300:]}
    print(name, len(rows), 'both accept', out[name]['both_accept'],
          'disagree', out[name]['nd_ok_lean_rej'] + out[name]['nd_rej_lean_ok'], flush=True)
# in-loop gate rate: every lean_gate log written on this pod (one json line per generate() call)
g = collections.Counter()
for fn in glob.glob('artifacts/nf/gate_*.jsonl'):
    if fn.endswith('.disagree.jsonl'):
        continue
    for l in open(fn):
        try:
            r = json.loads(l)
        except Exception:
            continue
        g['calls'] += 1
        for k in ('samples', 'parse_fail', 'distinct_checked', 'both_ok', 'nd_ok_lean_rej', 'nd_rej_lean_ok', 'both_rej'):
            g[k] += r.get(k, 0)
out['_gate'] = dict(g)
json.dump(out, open(f'artifacts/nf/record_{tag}.json', 'w'), indent=1)
print('gate', dict(g))
