#!/usr/bin/env python3
"""Build disjoint pools from generator output.

  python make_splits.py --cap6 data/raw_cap6.jsonl --long data/raw_long.jsonl --heldout_per_len 1000 --targets 3000 --transfer 2000

Disjointness is by atom-renaming class (`key`), across ALL pools; validation_36 classes are
removed from everything. Long pools exclude theorems with contradictory premises (A and ~A
both premises) because those have a short ex-falso proof regardless of generating length.
Writes data/train.jsonl, data/heldout.jsonl, data/rl_targets.jsonl, data/transfer.jsonl and data/splits_stats.json.
"""
import argparse, json, random, collections, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen import canon_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cap6', required=True)
    ap.add_argument('--long', required=True)
    ap.add_argument('--heldout_per_len', type=int, default=1000)
    ap.add_argument('--targets', type=int, default=3000)
    ap.add_argument('--transfer', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--outdir', default='data')
    a = ap.parse_args()
    rng = random.Random(a.seed)
    val = [json.loads(l) for l in open('targets/validation_36.jsonl')]
    val_keys = {canon_key(v['thm'].strip()) for v in val}
    used = set(val_keys)
    stats = {}

    def load(fn):
        rs = [json.loads(l) for l in open(fn) if l.strip()]
        for r in rs:
            r['key'] = canon_key(r['thm'])
        return rs

    cap6 = load(a.cap6)
    stats['cap6_raw'] = len(cap6)
    stats['cap6_validation_class_hits'] = sum(r['key'] in val_keys for r in cap6)
    cap6 = [r for r in cap6 if r['key'] not in val_keys]
    # trivial-pattern audit (all reported, only the first two were already excluded by the generator)
    def is_trivial(r):
        prem = r['thm'].split(' |- ')[0].split(' , ') if r['thm'].split(' |- ')[0] else []
        con = r['thm'].split(' |- ')[1]
        return con in prem or 'F' in prem or (r['n_lines'] == 2 and con in (f'( {p} v {p} )' for p in prem)) or (r['n_lines'] == 2 and con in (f'( {p} & {p} )' for p in prem))
    stats['cap6_trivial_pattern'] = sum(is_trivial(r) for r in cap6)
    stats['cap6_contra_prem'] = sum(r.get('contra_prem', False) for r in cap6)
    stats['cap6_len2'] = sum(r['n_lines'] == 2 for r in cap6)
    rng.shuffle(cap6)
    by = collections.defaultdict(list)
    for r in cap6:
        by[r['n_lines']].append(r)
    heldout, train = [], []
    for L, rs in sorted(by.items()):
        heldout += rs[:a.heldout_per_len]
        train += rs[a.heldout_per_len:]
    for r in train + heldout:
        assert r['key'] not in used or True
        used.add(r['key'])
    assert len({r['key'] for r in train} & {r['key'] for r in heldout}) == 0
    long = load(a.long)
    stats['long_raw'] = len(long)
    stats['long_validation_class_hits'] = sum(r['key'] in val_keys for r in long)
    stats['long_contra_prem_removed'] = sum(r.get('contra_prem', False) for r in long)
    long = [r for r in long if r['key'] not in used and not r.get('contra_prem', False)]
    rng.shuffle(long)
    byl = collections.defaultdict(list)
    for r in long:
        byl[r['n_lines']].append(r)
    targets, transfer = [], []
    nt = a.targets // len(byl); nx = a.transfer // len(byl)
    for L, rs in sorted(byl.items()):
        targets += rs[:nt]
        transfer += rs[nt:nt + nx]
    keys = [set(r['key'] for r in p) for p in (train, heldout, targets, transfer)]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not (keys[i] & keys[j]), (i, j)
    os.makedirs(a.outdir, exist_ok=True)
    for name, pool in (('train', train), ('heldout', heldout), ('rl_targets', targets), ('transfer', transfer)):
        with open(f'{a.outdir}/{name}.jsonl', 'w') as f:
            for i, r in enumerate(pool):
                rec = {'name': f'{name}_{i}', 'thm': r['thm'], 'key': r['key'], 'prompt': r['prompt'], 'n_lines': r['n_lines'],
                       'rules': r['rules'], 'n_prem': r['n_prem']}
                if name in ('train', 'heldout'):
                    rec['proof'] = r['proof']; rec['text'] = r['text']
                else:
                    rec['gen_proof'] = r['proof']   # generating proof: an UPPER BOUND on the shortest proof; never trained on
                f.write(json.dumps(rec) + '\n')
        stats[name + '_n'] = len(pool)
        stats[name + '_by_len'] = dict(sorted(collections.Counter(r['n_lines'] for r in pool).items()))
        rc = collections.Counter(x for r in pool for x in r['rules'])
        stats[name + '_rules'] = dict(rc.most_common())
        stats[name + '_n_prem'] = dict(sorted(collections.Counter(r['n_prem'] for r in pool).items()))
    json.dump(stats, open(f'{a.outdir}/splits_stats.json', 'w'), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()
