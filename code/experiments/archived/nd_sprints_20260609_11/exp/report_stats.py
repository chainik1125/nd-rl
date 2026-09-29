"""Compact stats digest from results files, for the writeup.

Usage (on pod): python -m exp.report_stats --res /root/nd/results
"""
import argparse
import json
import os
from collections import Counter, defaultdict


def load_jsonl(p):
    out = []
    with open(p) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def binrate(rs, binw=2, lo=2, pred=None):
    g = defaultdict(lambda: [0, 0])
    for r in rs:
        if r.get('n_lines') is None or (pred and not pred(r)):
            continue
        b = (r['n_lines'] - lo) // binw
        g[b][0] += r['solved']
        g[b][1] += 1
    return {f'{lo+b*binw}-{lo+b*binw+binw-1}': (s, n)
            for b, (s, n) in sorted(g.items()) if n >= 10}


def fmt(d):
    return '  '.join(f'{k}:{s}/{n}={s/n:.2f}' for k, (s, n) in d.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', default='/root/nd/results')
    args = ap.parse_args()
    R = args.res

    for cond in ['L8', 'L8_2xtrue', 'L12', 'L16']:
        for metric in ['greedy', 'pass8']:
            rs = []
            for ev in ['eval_iid', 'eval_ood']:
                p = os.path.join(R, f'cliff_{cond}_{ev}_{metric}.jsonl')
                if os.path.exists(p):
                    rs += load_jsonl(p)
            if rs:
                print(f'== {cond} {metric} (synthetic mixed) ==')
                print('  ', fmt(binrate(rs)))
            p = os.path.join(R, f'cliff_{cond}_eval_local_{metric}.jsonl')
            if os.path.exists(p):
                rs = load_jsonl(p)
                print(f'== {cond} {metric} (LOCAL-ONLY) ==')
                print('  ', fmt(binrate(rs)))

    # solved-proof length vs target generating length (difficulty proxy check)
    p = os.path.join(R, 'cliff_L8_eval_ood_greedy.jsonl')
    if os.path.exists(p):
        rs = [r for r in load_jsonl(p) if r['solved']]
        shorter = sum(1 for r in rs if r['best_len'] <= 8)
        print(f'L8 greedy ood solves: {len(rs)}; with proof <=8 lines: {shorter}')
        c = Counter()
        for r in rs:
            c[r['n_lines'] - r['best_len']] += 1
        print('  (gen_len - found_len) hist:', dict(sorted(c.items())))

    # failure reasons at the cliff
    p = os.path.join(R, 'cliff_L8_eval_ood_greedy.jsonl')
    if os.path.exists(p):
        agg = Counter()
        for r in load_jsonl(p):
            if not r['solved']:
                for k, v in (r.get('fail_reasons') or {}).items():
                    agg[k] += v
        print('L8 ood greedy failure reasons:', dict(agg.most_common(8)))

    # base pass@64 probe
    p = os.path.join(R, 'base_L8_transfer_pass64.jsonl')
    if os.path.exists(p):
        rs = load_jsonl(p)
        print(f'base L8 pass@64 on transfer: '
              f'{sum(r["solved"] for r in rs)}/{len(rs)}')
        print('   by len:', fmt(binrate(rs)))


if __name__ == '__main__':
    main()
