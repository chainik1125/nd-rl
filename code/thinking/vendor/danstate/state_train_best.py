"""`state_train.py --recipe best`: Robbie's pretraining recipe on (state, action) pairs (run `best-state`).

The loop is `pretrain()` of Robbie's `pretrain_best.py` (nd-rl `robbie-experiments`,
`code/experiments/current/factorial_20260929/`; network and optimizer pieces in `best_model.py`), with the state
trainer's data and loss:

  - a training example is one (state, action) pair from `state_train.load` (all pairs of all records), and the loss is
    next-token CE on the action tokens only (plus MTP at 0.3 on the same kind of positions);
  - one random `lean_seq` name offset per pair per presentation, on state and action together (`tok.shift_pair`,
    vectorised as in his `gpu_batch`);
  - token-budget batches (his TOK_BUDGET = 128 x 144 padded tokens): pairs sorted by length within a random
    permutation, cut into runs with count x max_len <= budget, runs shuffled;
  - schedule: warmup 200 steps, cosine to 0.3 x lr over the wall-clock budget (`--budget_secs`, his 300 s), or over
    `--best_steps` steps if given.  The clock starts after the data is on the device (his started before
    tokenisation; our decomposition into pairs is the format's cost, not the recipe's) and includes model
    construction, compilation and validation;
  - validation loss: the state trainer's held-out pairs (`--heldout`, first 2,000 records), action tokens, no offset.
Muon on the block matrices, the output head and the MTP matrices; AdamW (lr 1e-3, wd 0.1, betas 0.9/0.95) on the rest.
The checkpoint is `model.save_ckpt`'s format with cfg['arch'] = 'best', so every loader rebuilds it via `load_ckpt`.
"""
import math, random, time
import torch
import best_model as BM
from lean_tok import MAXN

TOK_BUDGET = 128 * 144
MIN_LR_FRAC, WARMUP = 0.3, 200


def to_device(data, dev):
    """list of per-record [(state bytes, action bytes)] -> (ids (N, T) uint8 on dev, plen on dev, lens on the host, lens on dev)."""
    pairs = [(p, q) for rec in data for p, q in rec]
    lens = torch.tensor([len(p) + len(q) for p, q in pairs], dtype=torch.long)
    plen = torch.tensor([len(p) for p, q in pairs], dtype=torch.long)
    N, T = len(pairs), int(lens.max())
    flat = torch.frombuffer(bytearray(b''.join(p + q for p, q in pairs)), dtype=torch.uint8)
    row = torch.repeat_interleave(torch.arange(N), lens)
    col = torch.arange(len(flat)) - torch.repeat_interleave(lens.cumsum(0) - lens, lens)
    ids = torch.zeros((N, T), dtype=torch.uint8)    # pad = 0
    ids[row, col] = flat
    return ids.to(dev), plen.to(dev), lens, lens.to(dev)


def token_batches(lens_cpu, gen_cpu, rng, budget):
    """One epoch of token-budget batches (Robbie's token_batches, on the host).  Returns (order, cuts)."""
    n = len(lens_cpu)
    assert int(lens_cpu.max()) <= budget, f'a pair of {int(lens_cpu.max())} tokens exceeds the token budget {budget}'
    perm = torch.randperm(n, generator=gen_cpu)
    order = perm[torch.argsort(lens_cpu[perm], stable=True)]
    sl = lens_cpu[order]
    cs = torch.cat([torch.zeros(1, dtype=torch.long), sl.cumsum(0)])
    cuts, s = [], 0
    while s < n:
        window = sl[s:s + budget // int(sl[s])]
        e = s + int((torch.arange(1, len(window) + 1) * window <= budget).sum())
        cuts.append((s, e, int(sl[e - 1]), int(cs[e] - cs[s])))    # (start, end, padded len, real tokens)
        s = e
    rng.shuffle(cuts)
    return order, cuts


def make_batch(ids, plen, lens, idxs, T, tok, gen, shift=True):
    x = ids[idxs, :T].long()
    ar = torch.arange(T, device=x.device)[None]
    m = (ar >= plen[idxs, None]) & (ar < lens[idxs, None])
    if shift and tok.shift:
        isref = x >= tok.ref0
        mx = torch.where(isref, x - tok.ref0 + 1, torch.zeros_like(x)).amax(1)
        room = MAXN - mx
        s = (torch.rand(len(idxs), device=x.device, generator=gen) * (room + 1)).long().clamp(max=room)
        x = torch.where(isref, x + s[:, None], x)
    return x, m


def val_loss(model, v, tok, autocast, bs=256):
    ids, plen, lens_cpu, lens = v
    model.eval()
    tot = n = 0.0
    with torch.no_grad(), autocast:
        for i in range(0, len(lens_cpu), bs):
            idxs = torch.arange(i, min(i + bs, len(lens_cpu)), device=ids.device)
            x, m = make_batch(ids, plen, lens, idxs, int(lens_cpu[i:i + bs].max()), tok, None, shift=False)
            k = float(m[:, 1:].sum())
            main = BM.mtp_loss(model, None, x, m, weight=0)[1]
            tot += float(main) * k; n += k
    model.train()
    return tot / n


def train(a, tok, data, held, dev, record):
    """Returns (model, extra) after training; the caller saves the checkpoint."""
    cuda = dev == 'cuda'
    t_data = time.time()
    tr = to_device(data, dev)
    ids, plen, lens_cpu, lens = tr
    vd = to_device(held, dev) if held else None
    print(f'best: {len(lens_cpu):,} train pairs (T {ids.shape[1]}), {0 if not vd else len(vd[2]):,} val pairs, '
          f'on {dev} in {time.time() - t_data:.0f}s', flush=True)
    t0 = time.time()
    torch.manual_seed(a.seed)
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    gen_cpu = torch.Generator(); gen_cpu.manual_seed(a.seed)
    rng = random.Random(a.seed)
    model = BM.ALiBiGPT(tok.vocab_size, a.n_layer, a.d, a.n_head, d_ff=a.d_ff).to(dev)
    mtp = BM.MTPHead(a.d, a.n_head, a.d_ff)
    mtp.apply(model._init)
    mtp = mtp.to(dev)
    compiled = []
    if cuda and not a.no_compile:
        torch._inductor.config.triton.coalesce_tiling_analysis = False
        compiled = list(model.blocks) + [mtp.block]
        for b in compiled:
            b.compile(dynamic=True)
    BM.orthogonal_init(list(model.blocks.parameters()) + list(mtp.parameters()))
    print(f'best: {model.n_params():,} params (+{sum(p.numel() for p in mtp.parameters()):,} MTP, not saved)', flush=True)
    trainable = list(model.parameters()) + list(mtp.parameters())
    matrices = [p for p in model.blocks.parameters() if p.ndim == 2] + [model.head.weight]
    matrices += [p for p in mtp.parameters() if p.ndim == 2]
    mids = {id(p) for p in matrices}
    aux_params = [p for p in trainable if id(p) not in mids]
    bufs = [torch.zeros_like(p) for p in matrices]
    muon_wd = 10 * a.wd * a.lr / BM.MUON_LR
    opt = torch.optim.AdamW(aux_params, lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95), fused=cuda)
    autocast = torch.autocast('cuda', dtype=torch.bfloat16, enabled=cuda)
    muon_lr = torch.zeros((), device=dev)
    muon_graph = None
    model.train()
    step, seen, curve, recent, recent_aux = 0, 0, [], [], []
    next_point = a.curve_every
    cuts, order, epochs_cuts = [], None, None
    while True:
        el = time.time() - t0
        frac = step / a.best_steps if a.best_steps else el / a.budget_secs
        if frac >= 1.0:
            break
        step += 1
        lr = a.lr * step / WARMUP if step < WARMUP else \
            a.lr * MIN_LR_FRAC + 0.5 * a.lr * (1 - MIN_LR_FRAC) * (1 + math.cos(math.pi * frac))
        for g in opt.param_groups:
            g['lr'] = lr
        if not cuts:
            order, cuts = token_batches(lens_cpu, gen_cpu, rng, a.tok_budget)
            epochs_cuts = len(cuts)
        s, e, T, ntok = cuts.pop()
        idxs = order[s:e].to(dev, non_blocking=True)
        seen += e - s
        record.count(train_steps=1, train_tokens=ntok)
        x, m = make_batch(ids, plen, lens, idxs, T, tok, gen)
        with autocast:
            loss, main, aux = BM.mtp_loss(model, mtp, x, m, weight=a.mtp)
        model.zero_grad(set_to_none=False)
        mtp.zero_grad(set_to_none=False)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        muon_lr.fill_(BM.MUON_LR * lr / a.lr)
        if not cuda:
            BM.muon_step(matrices, bufs, muon_lr, a.d, muon_wd)
        else:
            if muon_graph is None:    # capture once (Robbie's): warm-up, capture, then undo both updates
                state = matrices + bufs
                saved = [t.clone() for t in state]
                side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    BM.muon_step(matrices, bufs, muon_lr, a.d, muon_wd)
                torch.cuda.current_stream().wait_stream(side)
                muon_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(muon_graph):
                    BM.muon_step(matrices, bufs, muon_lr, a.d, muon_wd)
                with torch.no_grad():
                    for t, v in zip(state, saved, strict=True):
                        t.copy_(v)
            muon_graph.replay()
        if step % 20 == 0 or not cuda:
            recent.append(float(main)); recent_aux.append(float(aux))
        el = time.time() - t0
        if el >= next_point or (a.best_steps and step == a.best_steps):
            if not recent:
                recent.append(float(main)); recent_aux.append(float(aux))
            pt = {'secs': round(el, 1), 'step': step, 'train_loss': sum(recent) / max(1, len(recent)),
                  'mtp_loss': sum(recent_aux) / max(1, len(recent_aux)), 'epochs': seen / len(lens_cpu)}
            if vd:
                pt['val_loss'] = val_loss(model, vd, tok, autocast)
            curve.append(pt)
            print(f"step {step} loss {pt['train_loss']:.4f} mtp {pt['mtp_loss']:.4f} lr {lr:.2e} "
                  f"epochs {pt['epochs']:.2f} {el:.0f}s" + (f" val {pt['val_loss']:.4f}" if vd else ''), flush=True)
            recent, recent_aux = [], []
            next_point += a.curve_every
    secs = time.time() - t0
    for b in compiled:
        b._compiled_call_impl = None
    final_val = val_loss(model, vd, tok, autocast) if vd else None
    print(f'best: done {step} steps, {seen / len(lens_cpu):.2f} epochs of pairs, {secs:.0f}s, final val {final_val}', flush=True)
    model.eval()
    extra = {'recipe': 'best', 'steps': step, 'secs': secs, 'pairs_seen': seen, 'epochs': seen / len(lens_cpu),
             'batches_per_epoch': epochs_cuts, 'tok_budget': a.tok_budget, 'curve': curve, 'final_val': final_val,
             'last_train_loss': curve[-1]['train_loss'] if curve else float(main),
             'peak_mem_gb': torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else None}
    return model, extra
