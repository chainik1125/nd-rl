#!/usr/bin/env python3
"""Intervention (c) 'transfer from a sibling' for the ignition study: build the training mix for ONE fine-tuning step
identical to an expert-iteration round's training (each theorem's proofs capped at 4, repeated 4x, + 20k retained
Stage-1 records), from the arm's own found_4.jsonl PLUS the sibling arm's found_4.jsonl (outside data), then train.py
from the arm's round-4 checkpoint.  Usage:
  python ignition_transfer_mix.py --own artifacts/p2/<arm> --sibling artifacts/p2/<sib> --train data/p2/train_X.jsonl \
      --init ckpts/p2/<arm>_r4.pt --out ckpts/p2/<arm>_ivS_r4t.pt --seed S
"""
import argparse, json, os, random, subprocess, collections


def load(fn):
    d = collections.defaultdict(list)
    for l in open(fn):
        x = json.loads(l); d[x['name']].append(x)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--own', required=True); ap.add_argument('--sibling', required=True)
    ap.add_argument('--train', required=True); ap.add_argument('--init', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, required=True); ap.add_argument('--round', type=int, default=4)
    ap.add_argument('--retain', type=int, default=20000); ap.add_argument('--max_per_thm', type=int, default=4); ap.add_argument('--rl_weight', type=int, default=4)
    a = ap.parse_args()
    rng = random.Random(a.seed * 1000 + a.round + 77)
    own = load(f'{a.own}/found_{a.round}.jsonl'); sib = load(f'{a.sibling}/found_{a.round}.jsonl')
    train = [json.loads(l) for l in open(a.train) if l.strip()]
    mix = f'{a.own}_ivS_mix.jsonl'
    n_own = n_sib = 0
    with open(mix, 'w') as f:
        for src, tag in ((own, 'own'), (sib, 'sibling')):
            for name, fs in src.items():
                fs = list(fs); rng.shuffle(fs)
                for x in fs[:a.max_per_thm]:
                    for _ in range(a.rl_weight):
                        f.write(json.dumps({'prompt': x['prompt'], 'proof': x['proof'], 'n_lines': x['written'], 'src': tag}) + '\n')
                    if tag == 'own': n_own += 1
                    else: n_sib += 1
        for x in rng.sample(train, min(a.retain, len(train))):
            f.write(json.dumps({'prompt': x['prompt'], 'proof': x['proof'], 'n_lines': x['n_lines']}) + '\n')
    print(f'mix {mix}: own proofs {n_own} ({len(own)} theorems), sibling proofs {n_sib} ({len(sib)} theorems), retained {min(a.retain, len(train))}', flush=True)
    json.dump({'own': a.own, 'sibling': a.sibling, 'own_proofs': n_own, 'sibling_proofs': n_sib, 'init': a.init, 'out': a.out}, open(f'{a.own}_ivS_transfer.json', 'w'), indent=1)
    cmd = ['python3', 'train.py', '--data', mix, '--init', a.init, '--steps', '600', '--lr', '3e-4', '--min_lr', '3e-5', '--warmup', '50', '--cap', '0', '--out', a.out, '--seed', str(a.seed * 1000 + a.round), '--log_every', '200']
    print(' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
