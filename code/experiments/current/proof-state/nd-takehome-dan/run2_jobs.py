#!/usr/bin/env python3
"""Emit run-2 job lines (pod/runq.sh format). python run2_jobs.py ei|frozen|cov --patterns depth4,impe_chain4 --seeds 0,1"""
import argparse
SET = {'depth4': 'struct', 'impe_chain4': 'struct', 'nested_ore': 'struct', 'impi_ore': 'impi_ore_f0', 'negi_ande_hyp': 'negi_ande_hyp_f0', 'ori_ore': 'ori_ore_f0'}
ap = argparse.ArgumentParser(); ap.add_argument('kind'); ap.add_argument('--patterns', required=True); ap.add_argument('--seeds', default='0,1'); ap.add_argument('--k', type=int, default=2000)
a = ap.parse_args()
for pat in a.patterns.split(','):
    st = SET[pat]
    for s in a.seeds.split(','):
        ck = f'ckpts/r2/stage1_r2_{st}_s{s}.pt'; T = f'data/r2/targets_{pat}.jsonl'; X = f'data/r2/transfer_{pat}.jsonl'
        common = f'--targets {T} --transfer {X} --heldout data/p2/heldout.jsonl --train data/r2/train_r2_{st}.jsonl --rounds 8 --k 32 --temperature 0.8 --seed {s} --batch 768'
        if a.kind == 'ei':
            print(f'ei_r2_{pat}_s{s} python3 expert_iter.py --init {ck} --name r2/ei_{pat}_s{s} {common}')
        elif a.kind == 'frozen':
            print(f'frozen_r2_{pat}_s{s} python3 expert_iter.py --init {ck} --name r2/frozen_{pat}_s{s} {common} --no_train')
        elif a.kind == 'cov':
            print(f'cov_r2_{pat}_s{s} python3 coverage.py --ckpt {ck} --in {T} --out artifacts/r2/cov_{pat}_s{s} --k {a.k} --temperature 0.8 --limit 300 --batch 1024 --seed 0 --procs 3')
