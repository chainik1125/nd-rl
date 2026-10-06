#!/usr/bin/env python3
"""Job list of one noise-floor pod (Stage-1 seeds 0, 1 and 2 — seed 2 per pre-registration addendum 1) as lines '<jobname>\t<dep1,dep2|->\t<command>'.  python3 pod/nf/jobs.py <1|2>

Pod 1: null pools P1 (generator seed 21000) and P2 (22000); gap-closer = ds-composition's C0 ladder, T1 + frozen,
       both Stage-1 seeds.
Pod 2: null pools P3 (23000) and P4 (24000); gap-closers = ds-composition's A1 ladder (T1 + frozen, both seeds)
       and ds-generator's missing la_frozen_g1_s0.

Four chains per pod, one per (pool, Stage-1 seed) cell, each strictly sequential:
    stage1 -> heldout greedy -> FROZEN LADDER -> cov red -> cov req8 -> cov d3 -> [one gap-closer job]
so exactly four GPU jobs run at a time (CUDA_MEM_FRACTION 0.21) and the quantities are produced in
decreasing order of importance: if the run is cut, the frozen ladder (the headline) is already in hand.
"""
import sys
pod = int(sys.argv[1])
POOLS = {1: [('p1', 21000), ('p2', 22000)], 2: [('p3', 23000), ('p4', 24000)]}[pod]
D = 'data/p2'
B = 'hf://buckets/dan-pandori/nd-rl'
J = []

# ---- fetch: bucket checkpoints and training sets for the gap-closers (no retraining).
# The `hf` CLI is NOT installed on these pods and HF_TOKEN is not exported into the ssh command, so the
# bucket assets are downloaded on the VPS (which is logged in) and pushed with pod/nf/sync.sh push; this
# job only asserts they arrived.  (2026-09-24, disclosed in log.md.) ----
need = ({0: ['ckpts/lf/stage1_a1_seq_s0.pt', 'ckpts/lf/stage1_a1_seq_s1.pt', f'{D}/train_depth3_f0_a1.jsonl']}
        if pod == 1 else
        {0: ['ckpts/dsc/stage1_a1_s0.pt', 'ckpts/dsc/stage1_a1_s1.pt', 'ckpts/dsg/stage1_g1_s0.pt',
             'data/dsc/train_a1.jsonl', 'data/dsg/train_g1.jsonl']})[0]
J.append(('fetch', '-', 'ls -la ' + ' '.join(need)))

# ---- generation (CPU, all cores; the two pools of this pod run one after the other) ----
prev_gen = '-'
for p, gseed in POOLS:
    J.append((f'gen_{p}', prev_gen, f'bash pod/nf/gen.sh {p} {gseed}'))
    prev_gen = f'gen_{p}'

LA = ('python3 ladder_ei.py --outdir artifacts/nf --rounds 8 --k 32 --temperature 0.8 '
      '--batch 512 --max_new 512 --heldout {D}/heldout.jsonl'.format(D=D))
COV = ('python3 coverage.py --k 2000 --temperature 0.8 --seed 0 --batch 1000 --procs 2')

# ---- gap-closer jobs, one appended to the tail of each chain (most valuable first) ----
# Gap-closers, cut to what is not already on file with an identical command (2026-09-24 17:45, log.md):
#   - la_frozen_dsg_g1_s0: ds-generator's missing cell, and a fifth independent draw of the control's
#     distribution (different generator flags: --ore_steps 3 --ore_boxes).
#   - ds-composition's A1 ladder at Stage-1 seed 1 (T1 and frozen). C0's seed-1 ladder is NOT re-run:
#     ds-generator already measured the same checkpoint (ckpts/lf/stage1_a1_seq_s1.pt) with the same
#     ladder_ei command, the same pools and the same --batch 512 -> T1 965 / L* 11, frozen 114 / L* 9.
#   - A1's and C0's seed-0 ladders stay inherited from ds-composition (T1 965 / 856, frozen 205 / 158).
if pod == 1:
    CKA = lambda s: f'ckpts/dsc/stage1_a1_s{s}.pt'
    SETA = 'data/dsc/train_a1.jsonl'
    GAP = [('la_T1_dsc_a1_s1', f'{LA} --init {CKA(1)} --train {SETA} --seed 1 --name la_T1_dsc_a1_s1'),
           ('la_frozen_dsc_a1_s1', f'{LA} --init {CKA(1)} --train {SETA} --seed 1 --name la_frozen_dsc_a1_s1 --no_train')]
else:
    GAP = [('la_frozen_dsg_g1_s0', f'{LA} --init ckpts/dsg/stage1_g1_s0.pt --train data/dsg/train_g1.jsonl --seed 0 --name la_frozen_dsg_g1_s0 --no_train')]

# Scope cut 2026-09-24 17:45 (log.md, QUESTIONS.md): the measured cost of a frozen ladder on these
# pods is ~2.1 pod-hours, not the brief's ~1.25, and of a pass@2,000 coverage run ~0.7-2.4 h, not
# ~0.6.  Within the ceiling: seed 2 keeps Stage-1 + held-out greedy only (the quantity whose floor is
# bimodal and whose 12th cell is nearly free), its frozen ladder and coverage are dropped, and
# `targets_depth3` coverage - the stop rule's first drop - is dropped for every cell.
chains, tails = [], []
for p, _ in POOLS:
    for s in (0, 1, 2):
        SET = f'data/nf/train_{p}.jsonl'
        ck = f'ckpts/nf/stage1_{p}_s{s}.pt'
        c = [(f'stage1_{p}_s{s}', f'python3 train.py --data {SET} --heldout {D}/heldout.jsonl --mode lean_seq --steps 6000 --bs 128 --out {ck} --cap 6 --seed {s}'),
             (f'heldout_{p}_s{s}', f'python3 eval_set.py --ckpt {ck} --in {D}/heldout.jsonl --out artifacts/nf/heldout_{p}_s{s}.jsonl --k 1 --temperature 0 --batch 512 --summary artifacts/nf/heldout_{p}_s{s}.json')]
        if s == 2:
            tails.append(c)
            continue
        c += [(f'la_frozen_{p}_s{s}', f'{LA} --init {ck} --train {SET} --seed {s} --name la_frozen_{p}_s{s} --no_train'),
              (f'cov_red_{p}_s{s}', f'{COV} --ckpt {ck} --in {D}/targets_reductio_req.jsonl --out artifacts/nf/cov_red_{p}_s{s}'),
              (f'cov_req8_{p}_s{s}', f'{COV} --ckpt {ck} --in data/r3_1/depth3_req.jsonl --out artifacts/nf/cov_req8_{p}_s{s}')]
        chains.append(c)

for i, t in enumerate(tails):          # seed 2's Stage-1 + held-out, appended to the four chains
    chains[i % len(chains)] += t
for i, g in enumerate(GAP):            # then the gap-closers
    chains[i % len(chains)].append(g)

for i, c in enumerate(chains):
    p = POOLS[i // 2][0]
    prev = f'gen_{p},fetch'
    for name, cmd in c:
        J.append((name, prev, cmd))
        prev = name

J.append((f'record_{pod}', ','.join(c[-1][0] for c in chains), f'python3 pod/nf/record.py'))
for n, d, c in J:
    print(f'{n}\t{d}\t{c}')
