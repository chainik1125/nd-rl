#!/usr/bin/env python3
"""Overlap audit between the training pools and the provided evaluation files, by atom-renaming
class and by exact sequent string. Prints and writes counts only (no per-theorem output), so the
test set is not inspected."""
import json, gzip, re, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen import canon_key


def rd(fn):
    op = gzip.open if fn.endswith('.gz') else open
    return [json.loads(l) for l in op(fn, 'rt') if l.strip()]


def thm(prompt):
    m = re.match(r'THM\s*(.*?)\s*SEQ (.*) PRF', prompt)
    pre = m.group(1).strip()
    return f'{pre} |- {m.group(2).strip()}' if pre else f' |- {m.group(2).strip()}'


def main():
    pools = {'train': 'data/train.jsonl.gz' if os.path.exists('data/train.jsonl.gz') else 'data/train.jsonl',
             'heldout': 'data/heldout.jsonl', 'rl_targets': 'data/rl_targets.jsonl', 'transfer': 'data/transfer.jsonl'}
    keys = {p: {canon_key(r['thm']) for r in rd(fn)} for p, fn in pools.items()}
    strs = {p: {r['thm'].strip() for r in rd(fn)} for p, fn in pools.items()}
    out = {}
    lines = ['| evaluation file | n | renaming class in train | in held-out | in RL targets | in transfer | exact string in train |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for fn in ('targets/validation_36.jsonl', 'targets/test_short_prompts.jsonl', 'targets/test_long_prompts.jsonl'):
        rs = rd(fn)
        ks = [canon_key(thm(r['prompt'])) for r in rs]; ss = [thm(r['prompt']).strip() for r in rs]
        row = {p: sum(k in keys[p] for k in ks) for p in pools}
        row['exact_train'] = sum(s in strs['train'] for s in ss); row['n'] = len(rs)
        out[os.path.basename(fn)] = row
        lines.append(f"| {os.path.basename(fn)} | {len(rs)} | {row['train']} ({100*row['train']/len(rs):.0f}%) | {row['heldout']} | {row['rl_targets']} | {row['transfer']} | {row['exact_train']} |")
    json.dump(out, open('artifacts/test_overlap.json', 'w'), indent=1)
    open('artifacts/test_overlap.md', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
