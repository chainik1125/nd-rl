"""Certification cost probe: per record, cumulative seconds when each iterative-deepening level finished at floor F.

    python probe_cost.py IN.jsonl --floor 20 --secs 12 --procs 3 --limit 30
Prints median time of the last fully completed levels and the fraction exhausted, plus per-record
(n_body, S, n_prem, status, secs at level 16/17/18/19).
"""
import argparse
import json
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'certifier_exact'))
from certify_fast import run_fast  # noqa: E402


def work(args):
    r, floor, secs, mem = args
    pr = r['text'].split(' PRF ')[0] + ' PRF'
    x = run_fast(pr, floor=floor, secs=secs, mem_cap_mb=mem)
    lv = {b: t for b, t, _ in x['stats'].get('levels', [])}
    return {'n_body': r['n_body'], 'S': r.get('S'), 'n_prem': r['n_prem'], 'status': x['status'],
            'secs': round(x['secs'], 1), 'lv': {b: round(lv[b], 2) for b in (15, 16, 17, 18, 19) if b in lv},
            # every body budget up to last_level was exhausted: no proof with <= last_level body lines
            'last_level': max(lv, default=0), 'found_body': x.get('body')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--floor', type=int, default=20)
    ap.add_argument('--secs', type=float, default=12)
    ap.add_argument('--procs', type=int, default=3)
    ap.add_argument('--mem', type=float, default=1024)
    ap.add_argument('--limit', type=int, default=30)
    ap.add_argument('-v', action='store_true')
    a = ap.parse_args()
    recs = [json.loads(line) for line in open(a.src)][:a.limit]
    with ProcessPoolExecutor(a.procs) as ex:
        out = list(ex.map(work, [(r, a.floor, a.secs, a.mem) for r in recs]))
    if a.v:
        for o in out:
            print(o)
    for b in (16, 17, 18):
        # a level that did not finish counts as > secs
        ts = [o['lv'].get(b, 1e9) for o in out]
        print(f'level {b}: median {statistics.median(ts):.2f}s, finished {sum(t < 1e9 for t in ts)}/{len(ts)}')
    print('exhausted', sum(o['status'] == 'exhausted' for o in out), 'found', sum(o['status'] == 'found' for o in out),
          'of', len(out), '; median S', statistics.median(o['S'] or 0 for o in out),
          'median body', statistics.median(o['n_body'] for o in out))


if __name__ == '__main__':
    main()
