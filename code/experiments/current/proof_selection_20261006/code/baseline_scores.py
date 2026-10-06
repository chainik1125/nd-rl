#!/usr/bin/env python3
"""Baseline scorers (schema §3) from pool rows only (never labels).
random: uniform noise (seeded by pool stem + theorem name); first: -i; completion: done ? 1 : 0;
shortest: completion first, then fewer steps (done*1000 - steps).
  python baseline_scores.py --sets dev1108 textbook72 ...
"""
import argparse, hashlib, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pool_lib as PL


def scores_for(row, stem):
    att = row['attempts']
    seed = int(hashlib.sha256(f'{stem}|{row["name"]}'.encode()).hexdigest()[:12], 16)
    rng = random.Random(seed)
    return {
        'random': [rng.random() for _ in att],
        'first': [-float(a['i']) for a in att],
        'completion': [1.0 if a['done'] else 0.0 for a in att],
        'shortest': [(1000.0 if a['done'] else 0.0) - a['steps'] for a in att],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', nargs='+', default=list(PL.SET_FILES))
    a = ap.parse_args()
    os.makedirs(PL.SCORES, exist_ok=True)
    for s in a.sets:
        if not os.path.exists(PL.pool_path(s)):
            print('missing pool', s); continue
        stem = PL.pool_stem(s)
        rows = PL.load_pool(s)
        out = {}
        for r in rows:
            for sc, v in scores_for(r, stem).items():
                out.setdefault(sc, []).append(dict(name=r['name'], scores=v))
        for sc, rs in out.items():
            PL.write_jsonl(os.path.join(PL.SCORES, f'{stem}__{sc}.jsonl'), rs)
        print('wrote', stem, sorted(out))


if __name__ == '__main__':
    main()
