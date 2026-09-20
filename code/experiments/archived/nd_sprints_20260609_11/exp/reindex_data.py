"""Re-index proof texts with random line-number offsets.

Every line-index token N<k> in a record's proof becomes N<k+o>, with o drawn
uniformly per record from [0, 64 - n_lines]. The theorem, the proof structure,
and everything else are unchanged — only the (arbitrary) starting line number
shifts. After this transformation all index tokens N1..N64 appear in the
training distribution, removing the novel-token barrier to longer proofs.

Usage:
  python -m exp.reindex_data --in data/train_L8.jsonl --out data/train_L8off.jsonl --seed 11
"""
import argparse
import json
import random
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nd.verify import verify_text

NTOK = re.compile(r'^N(\d+)$')


def reindex_text(text, o):
    toks = text.split()
    out = []
    for t in toks:
        m = NTOK.match(t)
        out.append(f'N{int(m.group(1)) + o}' if m else t)
    return ' '.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--max-idx', type=int, default=64)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    n = 0
    checked = 0
    with open(args.inp) as fi, open(args.out, 'w') as fo:
        for line in fi:
            if not line.strip():
                continue
            r = json.loads(line)
            o = rng.randint(0, max(0, args.max_idx - r['n_lines']))
            r['text'] = reindex_text(r['text'], o)
            # spot-verify 1 in 200
            if n % 200 == 0:
                ok, reason, _ = verify_text(r['text'])
                assert ok, f'reindexed proof failed verify: {reason}'
                checked += 1
            fo.write(json.dumps(r) + '\n')
            n += 1
    print(f'reindexed {n} records ({checked} spot-verified) -> {args.out}')


if __name__ == '__main__':
    main()
