#!/usr/bin/env python3
"""Compare two harness smoke runs (rl_C.py vs rl_repr.py on the same frozen copy and seed): found proofs, per-round
sample statistics, fine-tune losses and compute must agree (losses to 1e-4: bf16 backward kernels may not be
bit-deterministic across runs)."""
import json, math, sys

a, b = sys.argv[1], sys.argv[2]


def load(d):
    ms = [json.loads(l) for l in open(f'{d}/metrics.jsonl')]
    found = sorted(open(f'{d}/found_targets.jsonl').read().splitlines())
    return ms, found, json.load(open(f'{d}/summary.json'))


ma, fa, sa = load(a)
mb, fb, sb = load(b)
ok = True
if fa != fb:
    ok = False
    print('found_targets differ:', len(fa), len(fb))
sa_s = [m for m in ma if m['phase'] == 'sample']
sb_s = [m for m in mb if m['phase'] == 'sample']
for x, y in zip(sa_s, sb_s):
    for k in ('samples', 'accept_rate', 'new_proofs', 'targets_solved_cum', 'sample_flops'):
        if x.get(k) != y.get(k):
            ok = False
            print('sample', x.get('round'), k, x.get(k), y.get(k))
la = [m['loss'] for m in ma if m['phase'] == 'train' and m.get('loss') is not None]
lb = [m['loss'] for m in mb if m['phase'] == 'train' and m.get('loss') is not None]
if len(la) != len(lb) or any(abs(x - y) > 1e-4 for x, y in zip(la, lb)):
    ok = False
    print('losses differ', la[:6], lb[:6])
for k in ('used_flops', 'optimizer_steps'):
    if sa['compute'][k] != sb['compute'][k]:
        ok = False
        print('compute', k, sa['compute'][k], sb['compute'][k])
print('status', sa['status'], sb['status'], 'heldout', sa['heldout_greedy'], sb['heldout_greedy'])
print('EQUIVALENT' if ok and sa['status'] == sb['status'] == 'ok' else 'NOT EQUIVALENT')
sys.exit(0 if ok else 1)
