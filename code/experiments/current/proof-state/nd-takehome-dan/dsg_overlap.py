#!/usr/bin/env python3
"""Overlap table of training sets against every evaluation pool (proposal 9 protocol): counts of set records whose theorem
matches a pool theorem (a) order-sensitive (`thm` string), (b) by renaming class (`key`, atoms relabelled by first appearance),
(c) premise-order-insensitive (renaming class of the theorem with premises sorted).

  python3 dsg_overlap.py --sets data/p2/train_depth3_f0_a1.jsonl data/dsg/train_g1.jsonl --pools data/p2/heldout.jsonl ... --out artifacts/dsg/overlap.json"""
import argparse, json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen import canon_key


def pkey(thm):
    thm = thm.strip()
    prem, concl = thm.split('|-')
    ps = sorted(p.strip() for p in prem.split(' , ') if p.strip()) if prem.strip() else []
    return canon_key((' , '.join(ps) + ' |- ' + concl.strip()) if ps else '|- ' + concl.strip())


def keys_of(fn):
    T, K, P = set(), set(), set()
    for l in open(fn):
        if not l.strip():
            continue
        r = json.loads(l); thm = r['thm'].strip()
        T.add(thm); K.add(r.get('key') or canon_key(thm)); P.add(pkey(thm))
    return T, K, P


ap = argparse.ArgumentParser()
ap.add_argument('--sets', nargs='+', required=True); ap.add_argument('--pools', nargs='+', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
pools = {p: keys_of(p) for p in a.pools}
out = {}
for s in a.sets:
    T, K, P = keys_of(s)
    row = {}
    for p, (pT, pK, pP) in pools.items():
        row[p] = {'pool_n': len(pT), 'thm': len(T & pT), 'class': len(K & pK), 'premise_order_insensitive': len(P & pP)}
    out[s] = {'set_n_classes': len(K), 'vs': row}
    print(f'== {s} ({len(K)} classes)')
    for p, v in row.items():
        print(f'   {p:55s} pool {v["pool_n"]:6d}  thm {v["thm"]:5d}  class {v["class"]:5d}  premise-order-insensitive {v["premise_order_insensitive"]:5d}')
os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
json.dump(out, open(a.out, 'w'), indent=1)
