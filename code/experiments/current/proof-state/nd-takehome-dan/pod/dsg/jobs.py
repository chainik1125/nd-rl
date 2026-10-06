#!/usr/bin/env python3
"""Job list of one arm as lines '<jobname>\t<dep1,dep2|->\t<command>'.  python3 pod/dsg/jobs.py <c0|g1|g2>
c0 : the a1 set and its bucket checkpoints (no Stage-1, dial on file): held-out greedy, coverage (3 pools), ladder T1 + frozen, record.
g1 / g2 : generation + assembly (CPU), Stage-1 x2, held-out greedy, dial EI + frozen (4 rounds), coverage, ladder T1 + frozen, record.
GPU concurrency by construction <= 4 jobs (CUDA_MEM_FRACTION 0.21 each)."""
import sys
arm = sys.argv[1]
D = 'data/p2'
SET = f'{D}/train_depth3_f0_a1.jsonl' if arm == 'c0' else f'data/dsg/train_{arm}.jsonl'
CK = (lambda s: f'ckpts/lf/stage1_a1_seq_s{s}.pt') if arm == 'c0' else (lambda s: f'ckpts/dsg/stage1_{arm}_s{s}.pt')
J = []
if arm != 'c0':
    J.append((f'gen_{arm}', '-', f'bash pod/dsg/gen.sh {arm}'))
    for s in (0, 1):
        J.append((f'stage1_{arm}_s{s}', f'gen_{arm}', f'python3 train.py --data {SET} --heldout {D}/heldout.jsonl --mode lean_seq --steps 6000 --bs 128 --out {CK(s)} --cap 6 --seed {s}'))
dial = f'--targets {D}/targets_depth3.jsonl --transfer {D}/transfer_depth3.jsonl --heldout {D}/heldout.jsonl --train {SET} --rounds 4 --k 32 --temperature 0.8 --batch 768'
for s in (0, 1):
    st = f'stage1_{arm}_s{s}' if arm != 'c0' else '-'
    both = f'stage1_{arm}_s0,stage1_{arm}_s1' if arm != 'c0' else '-'
    J.append((f'heldout_{arm}_s{s}', st, f'python3 eval_set.py --ckpt {CK(s)} --in {D}/heldout.jsonl --out artifacts/dsg/heldout_{arm}_s{s}.jsonl --k 1 --temperature 0 --batch 512 --summary artifacts/dsg/heldout_{arm}_s{s}.json'))
    if arm != 'c0':
        J.append((f'ei_d3_{arm}_s{s}', both, f'python3 expert_iter.py --init {CK(s)} --name dsg/ei_d3_{arm}_s{s} {dial} --seed {s}'))
        J.append((f'frozen_d3_{arm}_s{s}', both, f'python3 expert_iter.py --init {CK(s)} --name dsg/frozen_d3_{arm}_s{s} {dial} --seed {s} --no_train'))
    cov = lambda pool, fn: f'python3 coverage.py --ckpt {CK(s)} --in {fn} --out artifacts/dsg/cov_{pool}_{arm}_s{s} --k 2000 --temperature 0.8 --seed 0 --batch 1000 --procs 4'
    J.append((f'cov_d3_{arm}_s{s}', f'ei_d3_{arm}_s{s}' if arm != 'c0' else f'heldout_{arm}_s{s}', cov('d3', f'{D}/targets_depth3.jsonl')))
    J.append((f'cov_req8_{arm}_s{s}', f'cov_d3_{arm}_s{s}', cov('req8', 'data/r3_1/depth3_req.jsonl')))
    J.append((f'cov_red_{arm}_s{s}', f'cov_req8_{arm}_s{s}', cov('red', f'{D}/targets_reductio_req.jsonl')))
ladder_deps = f'cov_red_{arm}_s0,cov_red_{arm}_s1' + (f',frozen_d3_{arm}_s0,frozen_d3_{arm}_s1' if arm != 'c0' else '')
for s in (0, 1):
    la = f'python3 ladder_ei.py --init {CK(s)} --outdir artifacts/dsg --seed {s} --batch 512 --train {SET} --heldout {D}/heldout.jsonl'
    J.append((f'la_T1_{arm}_s{s}', ladder_deps, f'{la} --name la_T1_{arm}_s{s}'))
    J.append((f'la_frozen_{arm}_s{s}', ladder_deps, f'{la} --name la_frozen_{arm}_s{s} --no_train'))
J.append((f'record_{arm}', f'la_T1_{arm}_s0,la_T1_{arm}_s1,la_frozen_{arm}_s0,la_frozen_{arm}_s1', f'python3 pod/dsg/record.py {arm}'))
for n, d, c in J:
    print(f'{n}\t{d}\t{c}')
