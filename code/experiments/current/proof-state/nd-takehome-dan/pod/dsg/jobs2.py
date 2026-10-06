#!/usr/bin/env python3
"""Resume phase (2026-09-23) job list for ONE SEED as lines '<jobname>\t<dep1,dep2|->\t<command>'.
  python3 pod/dsg/jobs2.py <0|1>

The 2026-09-22 session was cut off before any ladder job finished and the host cleanup then deleted ckpts/.
This phase: (regress) check the adopted fast decode path is a no-op against this branch's base path; (fetch,
stage1) re-fetch G1 / G2's training sets from the bucket and retrain this seed's two Stage-1 checkpoints, with a
held-out greedy re-measurement as the reproduction check against the committed heldout_<arm>_s<seed>.json;
(ladder) T1 and frozen, 8 rounds x k 32, for C0, G2, G1 in that order; (record) the checker of record
(nd2lean.py --check, unmodified) over every counted proof of this seed, dial and coverage included.

**One seed per pod, all three arms on it** (RTX A6000 and RTX 4090 were what RunPod had; 3090 / A40 are out of
stock). Every arm-vs-arm comparison therefore happens on one GPU; the seed axis is confounded with the pod, which
is harmless because a difference is only claimed when both seeds agree in sign.

Sampler: `dsg_regress.py` on the pod found the adopted fast path token-identical to this branch's base path
EXCEPT with compaction on, which flipped 1 row of 128 at decode step 162 (a bf16 reduction-order change once the
live batch shrinks -- an RNG re-draw, not a wrong accept).  Per the pre-registration's fallback the run therefore
sets ND_SAMPLE_COMPACT=0, which reproduces the base path exactly, and keeps the originally pre-registered ladder
settings --batch 512 / --max_new 512 (the default).  What is retained from run `efficiency` is the memoised RoPE
table (identical values, no per-step host sync).  One sampling job per GPU at a time: these pods have a 7.65-CPU
quota, so a second job would contend with the Lean gate rather than the GPU.  C0 first, then G2 (the two arms
carrying the pre-registered falsifier), then G1 (pre-registered as a null manipulation) so G1 is the droppable tail."""
import sys
s = int(sys.argv[1])
D = 'data/p2'
SET = {'c0': f'{D}/train_depth3_f0_a1.jsonl', 'g1': 'data/dsg/train_g1.jsonl', 'g2': 'data/dsg/train_g2.jsonl'}
CK = {'c0': f'ckpts/lf/stage1_a1_seq_s{s}.pt', 'g1': f'ckpts/dsg/stage1_g1_s{s}.pt', 'g2': f'ckpts/dsg/stage1_g2_s{s}.pt'}
BUCKET = 'hf://buckets/dan-pandori/nd-rl/ds-generator'
J = [('regress', '-', f'python3 dsg_regress.py {CK["c0"]}'),
     ('fetch', '-', ' && '.join(f'hf buckets cp {BUCKET}/data/dsg/train_{a}.jsonl {SET[a]}' for a in ('g1', 'g2'))
      + f' && wc -l {SET["g1"]} {SET["g2"]}')]
prev = 'regress'


def ladder(arm):
    global prev
    la = (f'python3 ladder_ei.py --init {CK[arm]} --outdir artifacts/dsg --seed {s} --batch 512 '
          f'--train {SET[arm]} --heldout {D}/heldout.jsonl')
    for kind, extra in (('la_T1', ''), ('la_frozen', ' --no_train')):
        name = f'{kind}_{arm}_s{s}'
        J.append((name, prev, f'{la} --name {name}{extra}'))
        prev = name


ladder('c0')
for arm in ('g2', 'g1'):
    J.append((f'stage1_{arm}_s{s}', f'fetch,{prev}',
              f'python3 train.py --data {SET[arm]} --heldout {D}/heldout.jsonl --mode lean_seq --steps 6000 '
              f'--bs 128 --out {CK[arm]} --cap 6 --seed {s}'))
    prev = f'stage1_{arm}_s{s}'
    J.append((f'heldout2_{arm}_s{s}', prev,
              f'python3 eval_set.py --ckpt {CK[arm]} --in {D}/heldout.jsonl --out artifacts/dsg/heldout2_{arm}_s{s}.jsonl '
              f'--k 1 --temperature 0 --batch 512 --summary artifacts/dsg/heldout2_{arm}_s{s}.json'))
    prev = f'heldout2_{arm}_s{s}'
    ladder(arm)
for arm in ('c0', 'g2', 'g1'):
    J.append((f'record_{arm}_s{s}', prev, f'python3 pod/dsg/record.py {arm} {s}'))
    prev = f'record_{arm}_s{s}'
for n, d, c in J:
    print(f'{n}\t{d}\t{c}')
