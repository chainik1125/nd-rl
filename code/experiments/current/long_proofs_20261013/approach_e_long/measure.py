"""Tightness measurements for gen_long.py records with certify_fast (exact minlen minimum, in C).

    python measure.py IN.jsonl OUT.jsonl --floor written|N [--secs 15 --procs 3 --mem 1024] [--limit 100]
--floor written: search below each record's own written body length. Exhausted -> the written proof is minimal
(`tight`); found -> `min_body` is the exact minimum (a shortcut of n_body - min_body lines).
--floor N: the usual certificate (`cert_ge` = N when nothing under N lines exists).
Prints a summary by written-length bucket.
"""
import argparse
import collections
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'certifier_exact'))
import certify  # noqa: E402
from certify_fast import certify_fast  # noqa: E402


def work(args):
    rec, floor, secs, mem = args
    c = certify.check(rec['text'])
    if c is None:
        return {'rejected': certify.check.last_reason}
    f = c['n_body'] if floor == 'written' else int(floor)
    r = certify_fast(c['prompt'], c['n_prem'], floor=f, secs=secs, mem_cap_mb=mem, want_proof=True)
    out = {**{k: rec[k] for k in rec if k != 'text'}, **c, **r}
    out['status'] = ('error' if r.get('error') else 'timeout' if r['timeout'] else
                     'short' if r['min_body'] is not None else ('tight' if floor == 'written' else 'cert'))
    return out


def summarize(rs, bucket=5):
    by = collections.defaultdict(collections.Counter)
    for r in rs:
        by[r['n_body'] // bucket * bucket][r['status']] += 1
    tot = collections.Counter(r['status'] for r in rs)
    print('total', len(rs), dict(tot))
    for b in sorted(by):
        print(f'  body {b}-{b + bucket - 1}:', dict(by[b]))
    gaps = collections.Counter(r['n_body'] - r['min_body'] for r in rs if r['status'] == 'short')
    if gaps:
        print('  shortcut sizes (written - minimal):', dict(sorted(gaps.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--floor', default='written')
    ap.add_argument('--secs', type=float, default=15)
    ap.add_argument('--procs', type=int, default=3)
    ap.add_argument('--mem', type=float, default=1024)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    recs = [json.loads(line) for line in open(a.src)]
    if a.limit:
        recs = recs[:a.limit]
    rs = []
    with ProcessPoolExecutor(a.procs) as ex, open(a.dst, 'w') as out:
        for r in ex.map(work, [(x, a.floor, a.secs, a.mem) for x in recs], chunksize=1):
            if 'rejected' in r:
                print('rejected', r['rejected'])
                continue
            rs.append(r)
            out.write(json.dumps(r) + '\n')
            out.flush()
    summarize(rs)


if __name__ == '__main__':
    main()
