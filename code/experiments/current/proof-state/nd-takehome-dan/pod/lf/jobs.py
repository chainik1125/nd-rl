#!/usr/bin/env python3
"""Emit the job list of one pod role as lines '<jobname>\t<command>'.  python3 pod/lf/jobs.py <d3|ladder> <rand|seq>
d3     : Stage-1 on the depth-3 f=0 set a1 (seeds 0, 1), EI + frozen (8 rounds, k 32), one frozen token-format round for timing
ladder : Stage-1 on the full cap-6 set, held-out greedy + transfer pass@16 (mechanism test), ladder T1 + frozen (EI seeds 0, 1),
         one frozen token-format ladder round for timing; for seq also the --no_shift ablation Stage-1 and its pass@16."""
import sys
role, sch = sys.argv[1], sys.argv[2]
W = lambda f: f'until [ -f {f} ]; do sleep 30; done; '
J = []
if role == 'd3':
    D = 'data/p2'; common = f'--targets {D}/targets_depth3.jsonl --transfer {D}/transfer_depth3.jsonl --heldout {D}/heldout.jsonl --train {D}/train_depth3_f0_a1.jsonl --rounds 8 --k 32 --temperature 0.8 --batch 768'
    for s in (0, 1):
        ck = f'ckpts/lf/stage1_a1_{sch}_s{s}.pt'
        J.append((f'stage1_a1_{sch}_s{s}', f'python3 train.py --data {D}/train_depth3_f0_a1.jsonl --heldout {D}/heldout.jsonl --mode lean_{sch} --steps 6000 --bs 128 --out {ck} --cap 6 --seed {s}'))
        J.append((f'ei_d3_{sch}_s{s}', W(ck) + W(f'ckpts/lf/stage1_a1_{sch}_s1.pt') + f'python3 expert_iter.py --init {ck} --name lf/ei_d3_{sch}_s{s} {common} --seed {s}'))
        J.append((f'frozen_d3_{sch}_s{s}', W(ck) + W(f'ckpts/lf/stage1_a1_{sch}_s1.pt') + f'python3 expert_iter.py --init {ck} --name lf/frozen_d3_{sch}_s{s} {common} --seed {s} --no_train'))
    c1 = common.replace('--rounds 8', '--rounds 1')
    J.append((f'timing_d3_{sch}', f'python3 expert_iter.py --init ckpts/lf/token_stage1_a1.pt --name lf/timing_d3_token_on_{sch}pod {c1} --seed 0 --no_train && python3 expert_iter.py --init ckpts/lf/stage1_a1_{sch}_s0.pt --name lf/timing_d3_{sch} {c1} --seed 0 --no_train'))
else:
    ck = f'ckpts/lf/stage1_full_{sch}_s0.pt'
    J.append((f'stage1_full_{sch}_s0', f'python3 train.py --data data/train.jsonl --heldout data/heldout.jsonl --mode lean_{sch} --steps 6000 --bs 128 --out {ck} --cap 6 --seed 0'))
    ev = lambda c, tag: (f'python3 eval_set.py --ckpt {c} --in data/heldout.jsonl --out artifacts/lf/{tag}_heldout_greedy.jsonl --temperature 0 --batch 512 --summary artifacts/lf/{tag}_heldout_greedy.json && '
                         f'python3 eval_set.py --ckpt {c} --in data/transfer.jsonl --out artifacts/lf/{tag}_transfer2_k16.jsonl --k 16 --temperature 0.8 --seed 0 --batch 512 --summary artifacts/lf/{tag}_transfer2_k16.json')
    J.append((f'mech_full_{sch}', W(ck) + ev(ck, f'stage1_full_{sch}')))
    if sch == 'seq':
        ckf = 'ckpts/lf/stage1_full_seqfixed_s0.pt'
        J.append(('stage1_full_seqfixed_s0', f'python3 train.py --data data/train.jsonl --heldout data/heldout.jsonl --mode lean_seq --no_shift --steps 6000 --bs 128 --out {ckf} --cap 6 --seed 0'))
        J.append(('mech_full_seqfixed', W(ckf) + ev(ckf, 'stage1_full_seqfixed')))
    for s in (0, 1):
        J.append((f'la_T1_{sch}_s{s}', W(ck) + f'python3 ladder_ei.py --init {ck} --name la_T1_{sch}_s{s} --outdir artifacts/lf --seed {s} --batch 512'))
        J.append((f'la_frozen_{sch}_s{s}', W(ck) + f'python3 ladder_ei.py --init {ck} --name la_frozen_{sch}_s{s} --outdir artifacts/lf --seed {s} --batch 512 --no_train'))
    J.append((f'timing_la_{sch}', f'python3 ladder_ei.py --init ckpts/stage1_abs.pt --name timing_la_token_on_{sch}pod --outdir artifacts/lf --seed 0 --batch 512 --no_train --rounds 1 && python3 ladder_ei.py --init {ck} --name timing_la_{sch} --outdir artifacts/lf --seed 0 --batch 512 --no_train --rounds 1'))
for n, c in J:
    print(f'{n}\t{c}')
