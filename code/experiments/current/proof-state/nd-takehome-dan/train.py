#!/usr/bin/env python3
"""Supervised training / fine-tuning on (prompt, proof) records.

  python train.py --data data/train.jsonl --heldout data/heldout.jsonl --mode rel --steps 6000 --out ckpts/stage1_rel.pt --cap 6
  python train.py --data mix.jsonl --init ckpts/stage1.pt --steps 800 --lr 3e-4 --out ckpts/r1.pt --cap 0

--cap N  : assert every record's verifier length n_lines <= N (N=0 disables; Stage 1 must use 6).
Loss is next-token cross-entropy on proof tokens only (prompt tokens masked out).
"""
import argparse, json, math, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
from tokenizer import make_tokenizer
from model import GPT, save_ckpt, load_ckpt
from nd_verify import verify_text


def load(fn, tok, cap, check_verify=False):
    out = []
    for l in open(fn):
        if not l.strip():
            continue
        r = json.loads(l)
        if cap:
            assert r['n_lines'] <= cap, f'record exceeds cap {cap}: {r}'
            if check_verify:
                ok, reason, nl = verify_text(r['prompt'] + ' ' + r['proof'])
                assert ok and nl == r['n_lines'] and nl <= cap, (reason, nl, r)
        ids = tok.encode_proof(r['proof'])
        if cap and hasattr(tok, 'statement'):      # Lean surface form: the rendered record must denote exactly the cap-checked ND proof
            assert tok.decode(ids) == r['proof'], r
        out.append((tok.encode_prompt(r['prompt']), ids))
    return out


def batch(data, idxs, tok, rng, dev):
    seqs, masks = [], []
    for i in idxs:
        p, q = data[i]
        q = tok.shift_abs(q, rng)
        seqs.append(p + q)
        masks.append([0] * len(p) + [1] * len(q))
    T = max(len(s) for s in seqs)
    x = torch.full((len(seqs), T), tok.pad, dtype=torch.long)
    m = torch.zeros((len(seqs), T), dtype=torch.bool)
    for i, (s, mk) in enumerate(zip(seqs, masks)):
        x[i, :len(s)] = torch.tensor(s)
        m[i, :len(s)] = torch.tensor(mk, dtype=torch.bool)
    return x.to(dev), m.to(dev)


def loss_on(model, x, m):
    logits = model(x[:, :-1])
    tgt = x[:, 1:]
    mk = m[:, 1:]
    l = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1), reduction='none')
    return (l * mk.reshape(-1)).sum() / mk.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--heldout', default=None)
    ap.add_argument('--mode', default='rel')
    ap.add_argument('--init', default=None)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--min_lr', type=float, default=1e-4)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--wd', type=float, default=0.1)
    ap.add_argument('--n_layer', type=int, default=4)
    ap.add_argument('--d', type=int, default=256)
    ap.add_argument('--n_head', type=int, default=8)
    ap.add_argument('--cap', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--log_every', type=int, default=200)
    ap.add_argument('--no_shift', action='store_true', help='abs mode ablation: no random start-index offset (N1..N6 only ever seen)')
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if a.init:
        model, tok, _ = load_ckpt(a.init, dev)
        a.mode = tok.mode
    else:
        tok = make_tokenizer(a.mode)
        model = GPT(tok.vocab_size, a.n_layer, a.d, a.n_head).to(dev)
    tok.shift = not a.no_shift
    print('params', model.n_params(), 'mode', tok.mode, 'shift', tok.shift, flush=True)
    data = load(a.data, tok, a.cap, check_verify=(a.cap > 0))
    held = load(a.heldout, tok, 0)[:2000] if a.heldout else None
    print('train records', len(data), 'max len', max(len(p) + len(q) for p, q in data), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95))
    sched = lambda s: a.lr * s / a.warmup if s < a.warmup else a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * (s - a.warmup) / max(1, a.steps - a.warmup)))
    model.train()
    t0 = time.time()
    perm = []
    for step in range(1, a.steps + 1):
        if len(perm) < a.bs:
            perm = list(range(len(data)))
            rng.shuffle(perm)
        idxs = [perm.pop() for _ in range(a.bs)]
        x, m = batch(data, idxs, tok, rng, dev)
        for g in opt.param_groups:
            g['lr'] = sched(step)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
            loss = loss_on(model, x, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % a.log_every == 0 or step == a.steps:
            msg = f'step {step} loss {loss.item():.4f} lr {sched(step):.2e} {time.time()-t0:.0f}s'
            if held:
                model.eval()
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
                    vl = sum(loss_on(model, *batch(held, range(s, min(s + 256, len(held))), tok, rng, dev)).item() * min(256, len(held) - s)
                             for s in range(0, len(held), 256)) / len(held)
                msg += f' val {vl:.4f}'
                model.train()
            print(msg, flush=True)
    save_ckpt(a.out, model, tok.mode, extra={'args': vars(a), 'n_params': model.n_params(), 'secs': time.time() - t0})
    print('saved', a.out, flush=True)


if __name__ == '__main__':
    main()
