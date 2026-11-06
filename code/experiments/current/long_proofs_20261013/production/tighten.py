"""Stage 3: exact minimal lengths for certified records, by certifying at floor = written body length.

A record with cert_ge >= 20 and a written proof of n_body lines has minimum in [cert_ge, n_body]. Searching below
floor n_body either finds the exact minimum (< n_body) or exhausts, and then the written proof is minimal
(min_body = n_body, `tight`). Timeouts keep the interval. Cheapest (shortest) records first.

    python tighten.py OUT.jsonl IN.jsonl [more.jsonl ...] [--secs 20 --procs 3 --mem 1024]
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'certifier_exact'))
from certify_fast import certify_fast  # noqa: E402


def work(args):
    r, secs, mem = args
    if r['n_body'] <= r['cert_ge']:   # already certified at its own length
        return {'key': r['key'], 'n_body': r['n_body'], 'n_prem': r['n_prem'], 'cert_ge': r['cert_ge'],
                'min_body': r['n_body'], 'tight': True, 'timeout': False}
    lab = certify_fast(r['prompt'], r['n_prem'], floor=r['n_body'], secs=secs, mem_cap_mb=mem)
    out = {'key': r['key'], 'n_body': r['n_body'], 'n_prem': r['n_prem'], 'cert_ge': r['cert_ge'],
           'tighten_secs': lab['cert_secs'], 'timeout': lab['timeout']}
    if lab.get('min_body') is not None:
        out['min_body'] = lab['min_body']
    elif lab.get('cert_ge'):
        out['min_body'] = r['n_body']
        out['tight'] = True
    if lab.get('error'):
        out['error'] = lab['error']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--secs', type=float, default=20.0)
    ap.add_argument('--procs', type=int, default=3)
    ap.add_argument('--mem', type=float, default=1024.0)
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(line)['key'] for line in open(a.out)}
    recs = {}
    for p in a.paths:
        for line in open(p):
            r = json.loads(line)
            if (r.get('cert_ge') or 0) >= 20 and r['key'] not in done:
                recs.setdefault(r['key'], r)
    todo = sorted(recs.values(), key=lambda r: r['n_body'])
    print(f'{len(todo)} to tighten', flush=True)
    with ProcessPoolExecutor(a.procs) as ex, open(a.out, 'a') as f:
        for r in ex.map(work, [(r, a.secs, a.mem) for r in todo], chunksize=2):
            f.write(json.dumps(r) + '\n')
            f.flush()


if __name__ == '__main__':
    main()
