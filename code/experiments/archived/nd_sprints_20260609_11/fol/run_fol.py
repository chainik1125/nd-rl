"""Train a small GPT on FOL ND proofs and evaluate solve rate via sampling +
the independent FOL verifier. Self-contained (uses train.model.GPT + the FOL
tokenizer/verifier).

Usage:
  python -m fol.run_fol train --train data/fol_train.jsonl --out runs/folA --iters 6000
  python -m fol.run_fol eval  --ckpt runs/folA/ckpt.pt --targets data/fol_eval.jsonl --k 8
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
from fol.tokenizer import STOI, ITOS, PAD, VOCAB_SIZE
from fol.verify import verify_text
from train.model import GPT

QED = STOI['QED']


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def encode_examples(recs, block=512):
    data = []
    for r in recs:
        toks = r['text'].split()
        if len(toks) > block:
            continue
        ids = [STOI[t] for t in toks]
        data.append((ids, toks.index('PRF') + 1))
    return data


def make_batch(data, idxs, block, device, rng):
    rows = [data[i] for i in idxs]
    L = max(len(ids) for ids, _ in rows)
    B = len(rows)
    x = torch.full((B, L), PAD, dtype=torch.long)
    y = torch.full((B, L), PAD, dtype=torch.long)
    mask = torch.zeros((B, L))
    pos = torch.zeros((B, L), dtype=torch.long)
    for b, (ids, pl) in enumerate(rows):
        n = len(ids)
        x[b, :n - 1] = torch.tensor(ids[:-1])
        y[b, :n - 1] = torch.tensor(ids[1:])
        mask[b, pl - 1:n - 1] = 1.0
        off = rng.randint(0, block - n) if block > n else 0
        pos[b, :n - 1] = torch.arange(off, off + n - 1)
    return x.to(device), y.to(device), mask.to(device), pos.to(device)


def do_train(a):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    recs = []
    for spec in a.train:
        path, rep = (spec.split(':') + ['1'])[:2]
        recs += load(path) * int(rep)
    data = encode_examples(recs, a.block)
    print(f'{len(data)} sequences', flush=True)
    rng.shuffle(data)
    val, data = data[:2000], data[2000:]
    model = GPT(VOCAB_SIZE, block_size=a.block, n_layer=a.layers, n_head=a.heads,
                d_model=a.dim).to(device)
    print(f'{sum(p.numel() for p in model.parameters())/1e6:.2f}M params', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1, betas=(0.9, 0.95))

    def lr_at(it):
        if it < a.warmup:
            return a.lr * it / a.warmup
        p = (it - a.warmup) / max(1, a.iters - a.warmup)
        return a.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * p)))
    t0 = time.time()
    model.train()
    for it in range(1, a.iters + 1):
        for g in opt.param_groups:
            g['lr'] = lr_at(it)
        idxs = [rng.randrange(len(data)) for _ in range(a.bs)]
        x, y, m, pos = make_batch(data, idxs, a.block, device, rng)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
            _, loss = model(x, pos_ids=pos, targets=y, loss_mask=m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % 500 == 0 or it == a.iters:
            print(json.dumps({'it': it, 'loss': float(loss), 't': time.time() - t0}), flush=True)
    os.makedirs(a.out, exist_ok=True)
    torch.save({'model': model.state_dict(),
                'config': {'layers': a.layers, 'heads': a.heads, 'dim': a.dim,
                           'block': a.block, 'vocab': VOCAB_SIZE}},
               os.path.join(a.out, 'ckpt.pt'))
    print('saved', a.out, flush=True)


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    c = ck['config']
    m = GPT(c['vocab'], block_size=c['block'], n_layer=c['layers'], n_head=c['heads'], d_model=c['dim']).to(device)
    m.load_state_dict(ck['model'])
    m.eval()
    return m


def do_eval(a):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    model = load_model(a.ckpt, device)
    recs = load(a.targets)
    if a.limit:
        recs = recs[:a.limit]
    jobs = []
    for i, r in enumerate(recs):
        toks = r['text'].split()
        prompt = toks[:toks.index('PRF') + 1]
        pids = [STOI[t] for t in prompt]
        for _ in range(a.k):
            jobs.append((i, pids))
    order = sorted(range(len(jobs)), key=lambda j: len(jobs[j][1]))
    res = [{'soka': 0, 'best': None} for _ in recs]
    for s in range(0, len(order), a.bs):
        chunk = [jobs[j] for j in order[s:s + a.bs]]
        prompts = [c[1] for c in chunk]
        outs = model.sample(prompts, max_new=a.max_new, temperature=a.temp,
                            top_k=None, eos_id=QED, pad_id=PAD, device=device)
        for (ri, pid), o in zip(chunk, outs):
            full = ' '.join(ITOS[t] for t in pid + o)
            ok, _, nl = verify_text(full)
            if ok:
                res[ri]['soka'] += 1
                if res[ri]['best'] is None or nl < res[ri]['best']:
                    res[ri]['best'] = nl
    solved = sum(r['soka'] > 0 for r in res)
    out = [{'thm': rc['thm'], 'n_lines': rc.get('n_lines'), 'n_consts': rc.get('n_consts'),
            'quant': rc.get('quant'), 'solved': r['soka'] > 0, 'best_len': r['best']}
           for rc, r in zip(recs, res)]
    with open(a.out, 'w') as f:
        for r in out:
            f.write(json.dumps(r) + '\n')
    print(f'SOLVED {solved}/{len(recs)} = {solved/len(recs):.3f}')
    # by n_consts bucket
    from collections import defaultdict
    g = defaultdict(lambda: [0, 0])
    for r in out:
        g[r['n_consts']][0] += r['solved']
        g[r['n_consts']][1] += 1
    print('by n_consts:', {k: f'{v[0]}/{v[1]}' for k, v in sorted(g.items())})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd')
    t = sub.add_parser('train')
    t.add_argument('--train', nargs='+', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--iters', type=int, default=6000)
    t.add_argument('--bs', type=int, default=128)
    t.add_argument('--lr', type=float, default=6e-4)
    t.add_argument('--warmup', type=int, default=200)
    t.add_argument('--layers', type=int, default=4)
    t.add_argument('--heads', type=int, default=8)
    t.add_argument('--dim', type=int, default=256)
    t.add_argument('--block', type=int, default=512)
    t.add_argument('--seed', type=int, default=0)
    e = sub.add_parser('eval')
    e.add_argument('--ckpt', required=True)
    e.add_argument('--targets', required=True)
    e.add_argument('--k', type=int, default=8)
    e.add_argument('--temp', type=float, default=0.8)
    e.add_argument('--max-new', type=int, default=480)
    e.add_argument('--bs', type=int, default=1024)
    e.add_argument('--limit', type=int, default=None)
    e.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.cmd == 'train':
        do_train(a)
    elif a.cmd == 'eval':
        do_eval(a)


if __name__ == '__main__':
    main()
