#!/usr/bin/env python3
"""Size and proof-length histogram of every pool the harness uses, for one fork commit.

  python3 pools.py <fork-checkout>      # writes pools/<fork-commit>.json (commit read from remote.sh)

Run once per fork commit; params.py refuses to write params.json without the matching file. The
split logic (dev = sha1 parity of key, heldout = strided 2000, targets = seeded sample of N_TARGETS)
copies harness.py so the numbers describe exactly what the harness will see.
"""
import collections, hashlib, json, os, random, sys

from params import HERE, fork_commit, harness_rules


def read(fn):
    return [json.loads(l) for l in open(fn) if l.strip()]


def hist(recs):
    c = collections.Counter(r['n_lines'] for r in recs)
    return {'n': len(recs), 'by_lines': {str(k): c[k] for k in sorted(c)}}


def main():
    fork, commit, H = sys.argv[1], fork_commit(), harness_rules()
    train = read(f'{fork}/data/train.jsonl')
    heldout = read(f'{fork}/data/heldout.jsonl')
    heldout = heldout[::len(heldout) // 2000][:2000]
    targets_all = read(f'{fork}/data/ladder/rl_targets.jsonl')
    transfer = read(f'{fork}/data/ladder/transfer.jsonl')
    parity = lambda r: int(hashlib.sha1(r['key'].encode()).hexdigest(), 16) % 2
    out = {'fork_commit': commit, 'seed': 0,
           'pretrain': {'file': 'data/train.jsonl', **hist(train)},
           'pretrain_heldout': {'file': 'data/heldout.jsonl (strided to 2000)', **hist(heldout)},
           'rl_targets': {'file': 'data/ladder/rl_targets.jsonl', 'pool': hist(targets_all),
                          'sampled': hist(random.Random(0).sample(targets_all, min(H["N_TARGETS"], len(targets_all))))},
           'transfer_dev': {'file': 'data/ladder/transfer.jsonl (sha1 parity 0)', **hist([r for r in transfer if parity(r) == 0])},
           'transfer_holdout': {'file': 'data/ladder/transfer.jsonl (sha1 parity 1)', **hist([r for r in transfer if parity(r) == 1])}}
    fn = f'{HERE}/pools/{commit[:7]}.json'
    json.dump(out, open(fn, 'w'), indent=1)
    print(fn, {k: v.get('n') or v.get('sampled', {}).get('n') for k, v in out.items() if isinstance(v, dict)})


if __name__ == '__main__':
    main()
