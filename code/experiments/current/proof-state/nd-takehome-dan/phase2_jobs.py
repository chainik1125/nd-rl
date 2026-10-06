#!/usr/bin/env python3
"""Emit job lines for pod/runq.sh.  python phase2_jobs.py stage1|ei|frozen --sets TAG[,TAG..] --seeds 0,1 [--rounds 8]
TAG = <pattern>_f<freq> (a file data/p2/train_<TAG>.jsonl); pattern = TAG.rsplit('_f',1)[0]."""
import argparse, os
ap = argparse.ArgumentParser()
ap.add_argument('kind'); ap.add_argument('--sets', required=True); ap.add_argument('--seeds', default='0,1'); ap.add_argument('--rounds', type=int, default=8)
ap.add_argument('--batch', type=int, default=768); ap.add_argument('--k', type=int, default=32)
ap.add_argument('--targets', default=None, help='target pool tag (default: the pattern), e.g. reductio2 -> data/<dir>/targets_reductio2.jsonl')
ap.add_argument('--dir', default='p2', help='data/ckpts/artifacts subdirectory')
ap.add_argument('--suffix', default='', help='suffix for the EI/frozen/cov/nov arm names (e.g. _t2 for a second target pool)')
ap.add_argument('--cap', type=int, default=6)
ap.add_argument('--heldout', default=None)
ap.add_argument('--stage1_dir', default=None, help='where the Stage-1 checkpoints live (default ckpts/<dir>)')
a = ap.parse_args()
D = a.dir; S1 = a.stage1_dir or f'ckpts/{D}'; H = a.heldout or f'data/{D}/heldout.jsonl'
for tag in a.sets.split(','):
    pat = tag.rsplit('_f', 1)[0]
    tp = a.targets or pat
    for s in a.seeds.split(','):
        if a.kind == 'stage1':
            print(f'stage1_{tag}_s{s} python3 train.py --data data/{D}/train_{tag}.jsonl --heldout {H} --mode abs --steps 6000 --bs 128 --out {S1}/stage1_{tag}_s{s}.pt --cap {a.cap} --seed {s}')
        elif a.kind == 'ei':
            print(f'ei_{tag}_s{s}{a.suffix} until [ -f {S1}/stage1_{tag}_s{s}.pt ]; do sleep 30; done; python3 expert_iter.py --init {S1}/stage1_{tag}_s{s}.pt --name {D}/ei_{tag}_s{s}{a.suffix} --targets data/{D}/targets_{tp}.jsonl --transfer data/{D}/transfer_{tp}.jsonl --heldout {H} --train data/{D}/train_{tag}.jsonl --rounds {a.rounds} --k {a.k} --temperature 0.8 --seed {s} --batch {a.batch}')
        elif a.kind == 'cov':   # base-model pass@1e4 on the pattern targets (Phase-1 method), f=0 arms
            print(f'cov_{tag}_s{s}{a.suffix} python3 coverage.py --ckpt {S1}/stage1_{tag}_s{s}.pt --in data/{D}/targets_{tp}.jsonl --k {a.k} --temperature 0.8 --out artifacts/{D}/cov_{tag}_s{s}{a.suffix}_targets --batch 4096 --seed {s} --limit 300')
        elif a.kind == 'nov':   # base log-probs + surprisal loci of every proof the EI arm found (targets + transfer)
            print(f'nov_{tag}_s{s}{a.suffix} python3 novelty.py --ckpts base={S1}/stage1_{tag}_s{s}.pt final=ckpts/{D}/ei_{tag}_s{s}{a.suffix}_r{a.rounds - 1}.pt --src ei_targets=artifacts/{D}/ei_{tag}_s{s}{a.suffix}/found_{a.rounds}.jsonl ei_transfer=artifacts/{D}/ei_{tag}_s{s}{a.suffix}/found_transfer_{a.rounds}.jsonl --out artifacts/{D}/novelty_{tag}_s{s}{a.suffix}')
        elif a.kind == 'frozen':
            print(f'frozen_{tag}_s{s}{a.suffix} until [ -f {S1}/stage1_{tag}_s{s}.pt ]; do sleep 30; done; python3 expert_iter.py --init {S1}/stage1_{tag}_s{s}.pt --name {D}/frozen_{tag}_s{s}{a.suffix} --targets data/{D}/targets_{tp}.jsonl --transfer data/{D}/transfer_{tp}.jsonl --heldout {H} --train data/{D}/train_{tag}.jsonl --rounds {a.rounds} --k {a.k} --temperature 0.8 --seed {s} --batch {a.batch} --no_train')
