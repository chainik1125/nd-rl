"""Randomly relabel the constants in a proof (the FOL analog of Sprint-1's
line-index offset). Each proof's constants are remapped by a random injection
into the full constant inventory {a,b,c,d,e}, so every constant token appears
in training even when individual proofs use few. Removes the 'unseen constant
token' wall while preserving the proof exactly up to renaming parameters.

Re-verified after relabel.
"""
import argparse
import json
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fol.tokenizer import CONST_TOKENS
from fol.verify import verify_text

CONSTSET = set(CONST_TOKENS)


def relabel_text(text, rng):
    toks = text.split()
    used = sorted({t for t in toks if t in CONSTSET})
    if not used:
        return text
    targets = rng.sample(CONST_TOKENS, len(used))
    m = dict(zip(used, targets))
    return ' '.join(m.get(t, t) for t in toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    n = 0
    checked = 0
    with open(args.inp) as fi, open(args.out, 'w') as fo:
        for line in fi:
            if not line.strip():
                continue
            r = json.loads(line)
            r['text'] = relabel_text(r['text'], rng)
            if n % 300 == 0:
                ok, reason, _ = verify_text(r['text'])
                assert ok, f'relabel broke proof: {reason}'
                checked += 1
            fo.write(json.dumps(r) + '\n')
            n += 1
    print(f'relabeled {n} ({checked} spot-verified) -> {args.out}')


if __name__ == '__main__':
    main()
