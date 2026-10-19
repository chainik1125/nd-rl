"""Tally all producer outputs: distinct theorems (by key), certified counts, length and premise distributions.

    python tally.py out/*.jsonl
"""
import collections
import json
import sys

recs = {}
for path in sys.argv[1:]:
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line being written right now
            old = recs.get(r['key'])
            # keep the strongest label: a certificate at a higher floor, else an exact minimum
            if old is None or (r.get('cert_ge') or 0) > (old.get('cert_ge') or 0):
                recs[r['key']] = r
rs = list(recs.values())
cert = [r for r in rs if r.get('cert_ge')]
print(f'{len(rs)} distinct theorems')
print('certified >= floor:', dict(sorted(collections.Counter(r['cert_ge'] for r in cert).items())))
print('exact minimal body:', dict(sorted(collections.Counter(r['min_body'] for r in rs if r.get('min_body') is not None).items())))
print('timeouts / memo-cap:', sum(bool(r.get('timeout')) for r in rs))
print('certified, by premise count:', dict(sorted(collections.Counter(r['n_prem'] for r in cert).items())))
print('all, by premise count:', dict(sorted(collections.Counter(r['n_prem'] for r in rs).items())))
