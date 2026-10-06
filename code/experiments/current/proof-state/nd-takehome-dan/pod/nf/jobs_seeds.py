#!/usr/bin/env python3
"""Pre-registration addendum 2: 40 more Stage-1 seeds (3-12) on P1-P4, held-out greedy only.
Four chains of ten models, so the pod still runs exactly four concurrent GPU jobs.
  python3 pod/nf/jobs_seeds.py <pool> [<pool> ...]   -> '<jobname>\t<deps>\t<command>' lines
Each pod runs the pools whose 155,000-record set it generated, so no set has to be copied between pods."""
import sys
D = 'data/p2'
POOLS = tuple(sys.argv[1:]) or ('p1', 'p2', 'p3', 'p4')
SEEDS = range(3, 13)
NCHAIN = 4
chains = [[] for _ in range(NCHAIN)]
for i, p in enumerate(POOLS):
    for s in SEEDS:
        SET = f'data/nf/train_{p}.jsonl'
        ck = f'ckpts/nf/stage1_{p}_s{s}.pt'
        chains[(i * len(SEEDS) + (s - 3)) % NCHAIN] += [
            (f'stage1_{p}_s{s}', f'python3 train.py --data {SET} --heldout {D}/heldout.jsonl --mode lean_seq --steps 6000 --bs 128 --out {ck} --cap 6 --seed {s}'),
            (f'heldout_{p}_s{s}', f'python3 eval_set.py --ckpt {ck} --in {D}/heldout.jsonl --out artifacts/nf/heldout_{p}_s{s}.jsonl --k 1 --temperature 0 --batch 512 --summary artifacts/nf/heldout_{p}_s{s}.json')]
for c in chains:
    prev = '-'
    for name, cmd in c:
        print(f'{name}\t{prev}\t{cmd}')
        prev = name
