"""SFT trainer for ND proof models.

Trains a small GPT on JSONL proof datasets (records with 'text'). Loss is
computed only on the completion (after 'PRF') — the model learns to write
proofs given theorems. Each sequence gets a random position offset so all
absolute positions are visited during training (see model.py).

Usage:
  python -m train.train --train data/train_L8.jsonl --out runs/L8 \
      --iters 6000 --bs 128
Multiple --train files may be given as path:repeat (repeat = integer dup factor).
"""
import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from nd.tokenizer import STOI, PAD, VOCAB_SIZE
from train.model import GPT


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def encode_examples(records, block_size=512):
    data = []
    for r in records:
        toks = r['text'].split()
        if len(toks) > block_size:
            continue
        ids = [STOI[t] for t in toks]
        plen = toks.index('PRF') + 1
        data.append((ids, plen))
    return data


def make_batch(data, idxs, block_size, device, rng):
    rows = [data[i] for i in idxs]
    L = max(len(ids) for ids, _ in rows)
    B = len(rows)
    x = torch.full((B, L), PAD, dtype=torch.long)
    y = torch.full((B, L), PAD, dtype=torch.long)
    mask = torch.zeros((B, L))
    pos = torch.zeros((B, L), dtype=torch.long)
    for b, (ids, plen) in enumerate(rows):
        n = len(ids)
        x[b, :n - 1] = torch.tensor(ids[:-1])
        y[b, :n - 1] = torch.tensor(ids[1:])
        mask[b, plen - 1:n - 1] = 1.0  # predict tokens after PRF through QED
        off = rng.randint(0, block_size - n) if block_size > n else 0
        pos[b, :n - 1] = torch.arange(off, off + n - 1)
    return (x.to(device), y.to(device), mask.to(device), pos.to(device))


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    records = []
    for spec in args.train:
        if ':' in spec:
            path, rep = spec.rsplit(':', 1)
            rep = int(rep)
        else:
            path, rep = spec, 1
        rs = load_jsonl(path)
        records += rs * rep
        print(f'loaded {len(rs)} x{rep} from {path}')
    data = encode_examples(records, args.block)
    print(f'{len(data)} training sequences')

    # small held-out slice for val loss
    rng.shuffle(data)
    n_val = min(2000, len(data) // 20)
    val, data = data[:n_val], data[n_val:]

    model = GPT(VOCAB_SIZE, block_size=args.block, n_layer=args.layers,
                n_head=args.heads, d_model=args.dim, dropout=args.dropout).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f'{nparam/1e6:.2f}M params')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                            betas=(0.9, 0.95))

    def lr_at(it):
        if it < args.warmup:
            return args.lr * it / args.warmup
        p = (it - args.warmup) / max(1, args.iters - args.warmup)
        return args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * p)))

    metrics = []
    t0 = time.time()
    model.train()
    for it in range(1, args.iters + 1):
        for g in opt.param_groups:
            g['lr'] = lr_at(it)
        idxs = [rng.randrange(len(data)) for _ in range(args.bs)]
        x, y, m, pos = make_batch(data, idxs, args.block, device, rng)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=(device == 'cuda')):
            _, loss = model(x, pos_ids=pos, targets=y, loss_mask=m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % args.log_every == 0 or it == args.iters:
            model.eval()
            with torch.no_grad():
                vidx = [rng.randrange(len(val)) for _ in range(min(256, len(val)))]
                x, y, m, pos = make_batch(val, vidx, args.block, device, rng)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                    enabled=(device == 'cuda')):
                    _, vloss = model(x, pos_ids=pos, targets=y, loss_mask=m)
            model.train()
            rec = {'it': it, 'loss': float(loss.item()),
                   'val_loss': float(vloss.item()),
                   'lr': lr_at(it), 't': time.time() - t0}
            metrics.append(rec)
            print(json.dumps(rec))

    os.makedirs(args.out, exist_ok=True)
    cfg = {'layers': args.layers, 'heads': args.heads, 'dim': args.dim,
           'block': args.block, 'vocab': VOCAB_SIZE, 'nparam': nparam,
           'train': args.train, 'iters': args.iters, 'bs': args.bs,
           'lr': args.lr, 'seed': args.seed}
    torch.save({'model': model.state_dict(), 'config': cfg},
               os.path.join(args.out, 'ckpt.pt'))
    with open(os.path.join(args.out, 'metrics.json'), 'w') as f:
        json.dump({'config': cfg, 'log': metrics}, f, indent=1)
    print(f'saved to {args.out} ({time.time()-t0:.0f}s)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--iters', type=int, default=6000)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=6e-4)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--dropout', type=float, default=0.0)
    ap.add_argument('--block', type=int, default=512)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--log-every', type=int, default=250)
    train(ap.parse_args())


if __name__ == '__main__':
    main()
