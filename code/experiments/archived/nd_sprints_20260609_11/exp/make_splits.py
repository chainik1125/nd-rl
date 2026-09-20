"""Build train/eval/RL-target splits from the generated pools.

All splits are disjoint by theorem id (pools are already deduped per file;
we dedup across files here, first file wins).

Splits produced (in data dir):
  train_L8.jsonl      150k, n_lines 2..8
  train_L8_2x.jsonl   300k, n_lines 2..8 (superset of train_L8)
  train_L12.jsonl     150k, n_lines 2..12
  train_L16.jsonl     150k, n_lines 2..16
  eval_iid.jsonl      1000, n_lines 2..8
  eval_ood.jsonl      ~150/bin for n_lines 9..30 (2-line bins)
  rl_targets.jsonl    3000, n_lines 9..24
  eval_transfer.jsonl 1000, n_lines 9..24 (disjoint from rl_targets)
  oracle_long.jsonl   8000, n_lines 9..24 (disjoint; control data WITH proofs)

Eval/target/oracle sets are reserved FIRST, then training sets are filled from
the remainder, so training never contains an eval/target theorem.
"""
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


def dump(path, recs):
    with open(path, 'w') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')
    print(f'{path}: {len(recs)}')


def main(data_dir, textbook_path=None):
    rng = random.Random(42)
    pools = []
    for name in ['pool_short.jsonl', 'pool_long.jsonl']:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            pools.append(load(p))
    seen = set()
    recs = []
    for pool in pools:
        for r in pool:
            if r['thm'] not in seen:
                seen.add(r['thm'])
                recs.append(r)
    print(f'{len(recs)} unique theorems across pools')

    # exclude textbook target theorems from everything
    excluded = set()
    if textbook_path and os.path.exists(textbook_path):
        for r in load(textbook_path):
            excluded.add(r['thm'])
    recs = [r for r in recs if r['thm'] not in excluded]

    rng.shuffle(recs)
    short = [r for r in recs if 2 <= r['n_lines'] <= 8]
    mid = [r for r in recs if 9 <= r['n_lines'] <= 24]
    longr = [r for r in recs if 9 <= r['n_lines'] <= 30]

    # reserve eval/targets first
    eval_iid = short[:1000]
    short_rest = short[1000:]

    # OOD eval: up to 150 per 2-line bin over 9..30
    by_bin = {}
    eval_ood = []
    ood_ids = set()
    for r in longr:
        b = (r['n_lines'] - 9) // 2
        if by_bin.get(b, 0) < 150:
            by_bin[b] = by_bin.get(b, 0) + 1
            eval_ood.append(r)
            ood_ids.add(r['thm'])

    mid_rest = [r for r in mid if r['thm'] not in ood_ids]
    rl_targets = mid_rest[:3000]
    eval_transfer = mid_rest[3000:4000]
    oracle_long = mid_rest[4000:12000]

    d = data_dir
    dump(os.path.join(d, 'eval_iid.jsonl'), eval_iid)
    dump(os.path.join(d, 'eval_ood.jsonl'), eval_ood)
    dump(os.path.join(d, 'rl_targets.jsonl'), rl_targets)
    dump(os.path.join(d, 'eval_transfer.jsonl'), eval_transfer)
    dump(os.path.join(d, 'oracle_long.jsonl'), oracle_long)

    dump(os.path.join(d, 'train_L8.jsonl'), short_rest[:150000])
    dump(os.path.join(d, 'train_L8_2x.jsonl'), short_rest[:300000])
    reserved = set(x['thm'] for x in eval_iid) | ood_ids | \
        set(x['thm'] for x in rl_targets) | \
        set(x['thm'] for x in eval_transfer) | \
        set(x['thm'] for x in oracle_long)
    pool_ok = [r for r in recs if r['thm'] not in reserved]
    dump(os.path.join(d, 'train_L12.jsonl'),
         [r for r in pool_ok if r['n_lines'] <= 12][:150000])
    dump(os.path.join(d, 'train_L16.jsonl'),
         [r for r in pool_ok if r['n_lines'] <= 16][:150000])

    # report length histograms
    from collections import Counter
    for name in ['train_L8', 'eval_ood', 'rl_targets']:
        rs = load(os.path.join(d, name + '.jsonl'))
        c = Counter(r['n_lines'] for r in rs)
        print(name, 'discharge frac',
              round(sum(r['discharge'] for r in rs) / max(1, len(rs)), 3),
              dict(sorted(c.items())))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'data',
         sys.argv[2] if len(sys.argv) > 2 else None)
