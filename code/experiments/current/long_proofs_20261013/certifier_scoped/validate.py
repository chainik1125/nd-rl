"""Validate a lower-bound function against every record with a known proof length.

For each record with a prompt: lb must be <= n_body (the written proof, verified by nd_verify) and <= min_body
(minlen's exact minimum, a verified proof) when present. Exact minimum for tightness: min_body, or n_body when the
record was certified at floor = written length (cert_ge == n_body).

    python validate.py OUT.jsonl IN.jsonl [IN2.jsonl ...] [--fn sig|scoped] [--procs 3]
"""
import argparse
import collections
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _work(args):
    fn, rec = args
    import sigbound  # noqa: F401  (sets sys.path, widens atoms)
    from minlen import parse_thm
    prem, concl = parse_thm(rec['prompt'])
    if fn == 'sig':
        from sigbound import sig_bound
        b = sig_bound(prem, concl)
    else:
        from scoped import scoped_bound
        b = scoped_bound(prem, concl)
    exact = rec.get('min_body')
    if exact is None and rec.get('cert_ge') is not None and rec['cert_ge'] == rec.get('n_body'):
        exact = rec['n_body']
    ub = min(x for x in [rec.get('n_body'), rec.get('min_body')] if x is not None)
    return {'src': rec['_src'], 'prompt': rec['prompt'], 'n_body': rec.get('n_body'), 'exact': exact, 'ub': ub,
            'cert_ge': rec.get('cert_ge'), **b}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--fn', default='sig')
    ap.add_argument('--procs', type=int, default=3)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    recs, seen = [], set()
    for p in a.paths:
        n = 0
        for line in open(p):
            r = json.loads(line)
            if 'prompt' not in r and 'text' in r:
                r['prompt'] = r['text'].split(' PRF ', 1)[0] + ' PRF'
            if 'prompt' not in r or r['prompt'] in seen:
                continue
            if 'n_body' not in r:
                continue
            seen.add(r['prompt'])
            r['_src'] = os.path.basename(p)
            recs.append(r)
            n += 1
            if a.limit and n >= a.limit:
                break
    viol = 0
    stats = collections.defaultdict(list)
    with ProcessPoolExecutor(a.procs) as ex, open(a.out, 'w') as out:
        for r in ex.map(_work, [(a.fn, r) for r in recs], chunksize=16):
            out.write(json.dumps(r) + '\n')
            if r['lb'] > r['ub']:
                viol += 1
                print('VIOLATION', r['lb'], r['ub'], r['prompt'], flush=True)
            if r['exact'] is not None:
                stats[r['src']].append(r['lb'] / r['exact'])
            stats['written:' + r['src']].append(r['lb'] / r['n_body'])
    print(f'{len(recs)} records, violations {viol}')
    for k, v in sorted(stats.items()):
        v.sort()
        print(f'{k:40s} n={len(v):6d} mean lb/ref={sum(v)/len(v):.3f} median={v[len(v)//2]:.3f} min={v[0]:.3f}')


if __name__ == '__main__':
    main()
