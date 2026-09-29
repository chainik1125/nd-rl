"""No-retraining control for the discovery-accumulation curve.

Protocol identical to expert_iter rl mode (k rollouts per unsolved target per
round, same temperature, same target list) but the model is NEVER retrained.
Any growth in cumulative discoveries here is pure sampling lottery; the gap
between expert_iter's discovery curve and this one is the effect of retraining
on discovery power.

Usage:
  python -m exp.static_accum --ckpt runs/L8/ckpt.pt --data data --rounds 5 \
      --out results/static_accum.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from train.evalgen import load_model, run_eval


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', default='/root/nd/data')
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--n-rollouts', type=int, default=32)
    ap.add_argument('--n-targets', type=int, default=1200)
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--bs', type=int, default=1536)
    ap.add_argument('--torch-seed', type=int, default=123)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    torch.manual_seed(args.torch_seed)  # different stream than expert_iter's rollouts
    device = 'cuda'
    model, _ = load_model(args.ckpt, device)
    targets = load_jsonl(os.path.join(args.data, 'rl_targets.jsonl'))[:args.n_targets]
    solved = {}
    history = []
    for rnd in range(1, args.rounds + 1):
        unsolved = [t for t in targets if t['thm'] not in solved]
        rr = run_eval(model, unsolved, args.n_rollouts, args.temp, 420,
                      args.bs, device)
        new = 0
        for r in rr:
            if r['solved']:
                solved[r['thm']] = r['best_len']
                new += 1
        from collections import Counter
        lens_new = Counter(r['best_len'] for r in rr if r['solved'])
        history.append({'round': rnd, 'new': new, 'cum': len(solved),
                        'new_len_hist': dict(sorted(lens_new.items())),
                        'cum_len_hist': dict(sorted(Counter(solved.values()).items()))})
        print(json.dumps(history[-1]), flush=True)
        with open(args.out, 'w') as f:
            json.dump(history, f, indent=1)


if __name__ == '__main__':
    main()
