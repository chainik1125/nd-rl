"""Build cap-L training sets and just-past-cap RL target sets for the
overhang sweep. Run on a pod that already has pool_short.jsonl and the
standard splits (whose reserved theorems we must exclude).

Usage: python -m exp.make_caps --data /root/nd/data --cap 4
Produces: train_L{cap}.jsonl (<=cap lines), targets_L{cap}.jsonl
(cap+1 .. 3*cap lines, 1200 theorems).
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--cap', type=int, required=True)
    ap.add_argument('--n-train', type=int, default=120000)
    ap.add_argument('--n-targets', type=int, default=1200)
    args = ap.parse_args()
    d = args.data
    rng = random.Random(13)

    reserved = set()
    for name in ['eval_iid', 'eval_ood', 'rl_targets', 'eval_transfer',
                 'oracle_long', 'targets_textbook']:
        p = os.path.join(d, f'{name}.jsonl')
        if os.path.exists(p):
            for r in load(p):
                reserved.add(r['thm'])

    pool = []
    for name in ['pool_short.jsonl', 'pool_long.jsonl']:
        p = os.path.join(d, name)
        if os.path.exists(p):
            pool += load(p)
    pool = [r for r in pool if r['thm'] not in reserved]
    rng.shuffle(pool)

    lo, hi = args.cap + 1, 3 * args.cap
    targets = [r for r in pool if lo <= r['n_lines'] <= hi][:args.n_targets]
    target_ids = {r['thm'] for r in targets}
    train = [r for r in pool
             if r['n_lines'] <= args.cap and r['thm'] not in target_ids][:args.n_train]

    tp = os.path.join(d, f'train_L{args.cap}.jsonl')
    gp = os.path.join(d, f'targets_L{args.cap}.jsonl')
    with open(tp, 'w') as f:
        for r in train:
            f.write(json.dumps(r) + '\n')
    with open(gp, 'w') as f:
        for r in targets:
            f.write(json.dumps(r) + '\n')
    from collections import Counter
    print(f'cap {args.cap}: train {len(train)} (<= {args.cap} lines), '
          f'targets {len(targets)} ({lo}-{hi} lines)')
    print('target len hist:', dict(sorted(Counter(r["n_lines"] for r in targets).items())))


if __name__ == '__main__':
    main()
