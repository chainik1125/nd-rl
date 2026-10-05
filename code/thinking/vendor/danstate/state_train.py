#!/usr/bin/env python3
"""Supervised training on (state, action) pairs (run `state-env`, arms S and SH).

  python3 state_train.py --data data/p2/train_depth3_f0_a1.jsonl --heldout data/p2/heldout.jsonl \
      --mode lean_state --steps 6000 --recs 128 --out ckpts/se/stage1_S_s0.pt --cap 6

Every record is decomposed by `state_env.decompose` into its (state, action) pairs.  A step draws **--recs whole
proofs** and trains on *all* their pairs, so the model sees the same proofs the same number of times as the
whole-proof control (`train.py --bs 128`, 6,000 steps, 155,000 records); the pair batch is therefore
--recs x (actions per proof) = 128 x 5.0 = 640 on the control's set.  Loss is next-token cross-entropy on the
**action** tokens only: the state is an observation, not something the policy writes.  One random `lean_seq` name
offset per proof per presentation, applied to state and action together.
"""
import argparse, collections, json, math, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
from tokenizer import make_tokenizer
from model import GPT, save_ckpt, load_ckpt
from state_env import decompose
from lean_tok import ParseFail


def load(fn, tok, cap, limit=0):
    """-> (list of per-record [(state ids, action ids)], n_skipped)"""
    out, skipped = [], 0
    for l in open(fn):
        if not l.strip():
            continue
        r = json.loads(l)
        if cap:
            assert r['n_lines'] <= cap, f'record exceeds cap {cap}: {r.get("name")}'
        try:
            steps, toks, env = decompose(r['prompt'], r['proof'], canon=getattr(tok, 'canon', False))
        except (ParseFail, ValueError, AssertionError, IndexError) as e:
            skipped += 1
            continue
        pairs = []
        for stt, act, hs in steps:
            pre = tok.encode_toks((hs + stt) if tok.with_history else stt)
            pairs.append((bytes(pre), bytes(tok.encode_toks(act) + [tok.eos])))
        out.append(pairs)
        if limit and len(out) >= limit:
            break
    return out, skipped


def batch(data, idxs, tok, rng, dev):
    seqs, masks = [], []
    for i in idxs:
        for p, q in data[i]:
            p, q = tok.shift_pair(list(p), list(q), rng)
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
    l = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), x[:, 1:].reshape(-1), reduction='none')
    return (l * m[:, 1:].reshape(-1)).sum() / m[:, 1:].sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--heldout', default=None)
    ap.add_argument('--mode', default='lean_state')
    ap.add_argument('--init', default=None)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--recs', type=int, default=128, help='whole proofs per step (the control trains on 128 proofs/step)')
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
    ap.add_argument('--max_tokens', type=int, default=200000, help='padded tokens per micro-batch (same gradient, less memory)')
    ap.add_argument('--recipe', default='control', choices=['control', 'best'],
                    help="best: Robbie's 6x384 Peri-LN/ALiBi network, Muon + AdamW, MTP 0.3, token-budget batches (state_train_best.py)")
    ap.add_argument('--budget_secs', type=float, default=300, help='best: wall-clock budget of the cosine schedule')
    ap.add_argument('--best_steps', type=int, default=0, help='best: schedule over this many steps instead of --budget_secs')
    ap.add_argument('--tok_budget', type=int, default=128 * 144, help='best: padded tokens per step')
    ap.add_argument('--mtp', type=float, default=0.3, help='best: MTP loss weight')
    ap.add_argument('--curve_every', type=float, default=30, help='best: seconds between curve / val points')
    ap.add_argument('--no_compile', action='store_true')
    ap.add_argument('--best_dims', default=None, help='best: "layers,d,heads,d_ff" override (CPU tests only; the recipe is 6,384,8,1280)')
    a = ap.parse_args()
    if a.recipe == 'best':
        assert not a.init, '--recipe best is Stage-1 pretraining; fine-tunes of a best checkpoint use the default loop'
        import best_model as BM
        a.n_layer, a.d, a.n_head, a.d_ff = (BM.N_LAYER, BM.D, BM.N_HEAD, BM.D_FF) if not a.best_dims else \
            tuple(int(x) for x in a.best_dims.split(','))
    import record    # results registry (REGISTRY.md)
    record.save_config(vars(a), a.out, role='finetune' if a.init else 'stage1')
    record.preflight()    # ND_RUN_ID + hf CLI present, or ND_OFFLINE=1: checked before training, not at the first save
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if a.init:
        model, tok, _ = load_ckpt(a.init, dev)
        a.mode = tok.mode
    else:
        tok = make_tokenizer(a.mode)
        model = GPT(tok.vocab_size, a.n_layer, a.d, a.n_head).to(dev)
    assert getattr(tok, 'state_mode', False), 'state_train needs a lean_state* tokenizer'
    print('params', model.n_params(), 'mode', tok.mode, flush=True)
    t0 = time.time()
    data, skipped = load(a.data, tok, a.cap)
    npairs = sum(len(x) for x in data)
    print(f'train records {len(data)} (skipped {skipped}); pairs {npairs}; mean {npairs/max(1,len(data)):.3f} per proof; '
          f'max pair len {max(len(p)+len(q) for x in data for p, q in x)}; load {time.time()-t0:.0f}s', flush=True)
    held = None
    if a.heldout:
        held, _ = load(a.heldout, tok, 0, limit=2000)
    if a.recipe == 'best':
        import state_train_best
        del model
        model, extra = state_train_best.train(a, tok, data, held, dev, record)
        save_ckpt(a.out, model, tok.mode, extra={'args': vars(a), 'n_params': model.n_params(), 'pairs': npairs,
                                                 'records': len(data), **extra})
        lab = dict(ckpt=a.out, data=a.data, source=a.out, n_params=model.n_params(), pairs=npairs, recipe='best')
        record.record('train_loss', extra['last_train_loss'], **lab)
        if held:
            record.record('val_loss', extra['final_val'], n=len(held), **lab)
        print('saved', a.out, flush=True)
        return
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95))
    sched = lambda s: a.lr * s / a.warmup if s < a.warmup else a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * (s - a.warmup) / max(1, a.steps - a.warmup)))
    model.train()
    t0 = time.time()
    perm = []
    npb = []
    for step in range(1, a.steps + 1):
        if len(perm) < a.recs:
            perm = list(range(len(data)))
            rng.shuffle(perm)
        idxs = [perm.pop() for _ in range(a.recs)]
        x, m = batch(data, idxs, tok, rng, dev)
        npb.append(x.shape[0])
        record.count(train_steps=1, train_tokens=sum(len(p) + len(q) for i in idxs for p, q in data[i]))   # non-pad tokens
        for g in opt.param_groups:
            g['lr'] = sched(step)
        opt.zero_grad(set_to_none=True)
        # micro-batches of <= --max_tokens padded tokens, each weighted by its share of the action tokens, so the
        # gradient is the same token-mean as one full batch (arm SH's history makes pairs long enough to OOM a 24 GB card)
        per = max(1, a.max_tokens // x.shape[1])
        ntok = m[:, 1:].sum()
        loss_sum = 0.0
        for s0 in range(0, x.shape[0], per):
            xb, mb = x[s0:s0 + per], m[s0:s0 + per]
            kb = mb[:, 1:].sum()
            if int(kb) == 0:
                continue
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
                lb = loss_on(model, xb, mb) * (kb / ntok)
            lb.backward()
            loss_sum += float(lb)
        loss = torch.tensor(loss_sum)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % a.log_every == 0 or step == a.steps:
            msg = f'step {step} loss {loss.item():.4f} lr {sched(step):.2e} pairs/batch {sum(npb)/len(npb):.0f} {time.time()-t0:.0f}s'
            npb = []
            if held:
                model.eval()
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
                    tot, n = 0.0, 0
                    for s in range(0, len(held), 64):
                        xb, mb = batch(held, range(s, min(s + 64, len(held))), tok, rng, dev)
                        k = int(mb[:, 1:].sum())
                        tot += loss_on(model, xb, mb).item() * k; n += k
                msg += f' val {tot/n:.4f}'
                model.train()
            print(msg, flush=True)
    save_ckpt(a.out, model, tok.mode, extra={'args': vars(a), 'n_params': model.n_params(), 'secs': time.time() - t0,
                                             'pairs': npairs, 'records': len(data)})
    lab = dict(ckpt=a.out, data=a.data, source=a.out, n_params=model.n_params(), pairs=npairs)
    record.record('train_loss', loss.item(), **lab)
    if held:
        record.record('val_loss', tot / n, n=len(held), **lab)
    print('saved', a.out, flush=True)


if __name__ == '__main__':
    main()
