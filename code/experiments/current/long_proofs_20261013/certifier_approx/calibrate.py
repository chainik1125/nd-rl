"""Run hunter variants over the exact ground truth and log what each finds.

    python calibrate.py OUT.jsonl VARIANT [VARIANT ...] [--secs 15] [--hi 19] [--only-exact]

Ground truth = approach_d_theorem_first/{r2_cert20,r1_cert20,t1_cert20,t1_rest_cert16,hunt_cert18}.jsonl,
deduplicated on `key`, longest-min first. Each output line: {key, prompt, n_prem, true_min (or None),
cert_ge, variant, found, exh, stop, secs, memo}.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
from hunter import hunt  # noqa: E402

GT_FILES = ['r2_cert20', 'r1_cert20', 't1_cert20', 't1_rest_cert16', 'hunt_cert18']

VARIANTS = {
    'none': dict(lem='none'),
    'none_d2': dict(lem='none', max_depth=2),
    'lit': dict(lem='lit'),
    'sub': dict(lem='sub'),
    'full_cap1m': dict(lem='all', memo_cap=1_000_000),
    'rand300k_s1': dict(lem='all', memo_cap=300_000, seed=1),
    'rand300k_s2': dict(lem='all', memo_cap=300_000, seed=2),
    'lit_cap1m': dict(lem='lit', memo_cap=1_000_000),
    'none_cap1m': dict(lem='none', memo_cap=1_000_000),
}


def ground_truth(extra=()):
    seen, rs = {}, []
    for name in GT_FILES:
        p = os.path.join(HERE, '..', 'approach_d_theorem_first', name + '.jsonl')
        for line in open(p):
            r = json.loads(line)
            if r['key'] in seen:
                continue
            seen[r['key']] = 1
            rs.append({'key': r['key'], 'prompt': r['prompt'], 'n_prem': r['n_prem'], 'true_min': r.get('min_body'),
                       'cert_ge': r.get('cert_ge'), 'src': name})
    for p in extra:
        for line in open(p):
            r = json.loads(line)
            if r.get('key') in seen or r.get('min_body') is None:
                continue
            seen[r['key']] = 1
            rs.append({'key': r['key'], 'prompt': r['prompt'], 'n_prem': r['n_prem'], 'true_min': r['min_body'],
                       'cert_ge': None, 'src': os.path.basename(p)})
    rs.sort(key=lambda r: -(r['true_min'] if r['true_min'] is not None else 99))
    return rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('variants', nargs='+')
    ap.add_argument('--secs', type=float, default=15)
    ap.add_argument('--hi', type=int, default=19)
    ap.add_argument('--only-exact', action='store_true')
    ap.add_argument('--extra', nargs='*', default=[])
    a = ap.parse_args()
    rs = ground_truth(a.extra)
    if a.only_exact:
        rs = [r for r in rs if r['true_min'] is not None]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            d = json.loads(line)
            done.add((d['key'], d['variant']))
    t0 = time.time()
    with open(a.out, 'a') as fo:
        for i, r in enumerate(rs):
            for v in a.variants:
                if (r['key'], v) in done:
                    continue
                h = hunt(r['prompt'], 1, a.hi, secs=a.secs, **VARIANTS[v])
                row = {**r, 'variant': v, 'found': h['found'], 'exh': h['exh'], 'stop': h['stop'], 'secs': h['secs'],
                       'memo': h['memo'], 'S': h['S'], 'per_level': h['per_level']}
                if h['found'] is not None and r['true_min'] is not None and h['found'] < r['true_min']:
                    row['ANOMALY'] = 'found shorter than exact label'
                fo.write(json.dumps(row) + '\n')
                fo.flush()
            if (i + 1) % 20 == 0:
                print(f'{i+1}/{len(rs)} {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
