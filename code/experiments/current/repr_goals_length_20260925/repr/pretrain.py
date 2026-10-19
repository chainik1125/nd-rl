#!/usr/bin/env python3
"""Fixed-step pretraining with the pretraining team's run 039 recipe (039_muon-adamw-finish-5pct) for one
representation variant, plus the reproduction metrics.

    python pretrain.py --variant abs --seed 0 [--steps 2911] [--out pretrain/abs_s0] [--gpu auto|env|N]
    python pretrain.py --variant abs --seed 0 --steps 60 --out /tmp/x --gpu env --smoke   # plumbing check
    python pretrain.py --ref 0 --out pretrain/ref039_s0      # metrics of the released 039 checkpoint s0 only

The recipe is 039's pretrain.py (source/pretrain.py of 039_muon-adamw-finish-5pct_s0) line for line, with one
change the monitor asked for: the schedule runs on a fixed step count instead of the 300 s wall clock. 039 set
frac = elapsed / budget at the top of each step; here frac = (steps done) / steps. Everything else as 039:
batch 256 records, lr 1e-3 (warmup 200 steps, cosine over frac to 0.1x), AdamW (0.9, 0.95), wd 0.1 on the
embeddings / layer norms / biases / head; Muon (lr 0.01 x schedule, momentum 0.95, 5 Newton-Schulz steps, QKV
orthogonalised per role, decoupled wd 0.01) on the 2-D block matrices; at frac >= 0.95 the block matrices join
AdamW with fresh moments; grad clip 1.0; bf16 autocast; data = frozen/data/train_cap6.jsonl (byte-identical to
039's data/train.jsonl), the first 1,024 records held out for val loss; abs index shift per record (abs mode);
seed -> torch.manual_seed (model init) and a CUDA generator (data order, shifts).

Variant additions (nothing else changes): pos_train='rand' (posrand) draws each row's positions as a sorted
random subset of range(1024) from its own generator (seed + 1000), so the data order and shifts equal abs's;
pe='line' (posline) computes line positions from the tokens inside the model; rel has no index shift (039's
gpu_batch shifts in abs mode only).

Metrics written to <out>/summary.json (the reproduction check compares them with the released checkpoints):
  val_loss            039's val_loss: mean proof-token loss on train records [0, 1024), no shift, positions 0..
  val_loss_randpos    posrand only: the same with training-style random positions
  heldout_greedy_039  039 harness greedy_rate: heldout_cap6[::2][:2000], greedy, max_new 512 (its reported metric)
  heldout_greedy      frozen/harness.py heldout_greedy: heldout_cap6[::5][:1000], greedy, max_new 512 (what the RL
                      harness reports at the end of every run)
  heldout_greedy_randpos  posrand only: the 039 slice with random positions (see evaluate.py's rand positions)
"""
import argparse, json, math, os, random, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common_repr import (HELDOUT_CAP6, PRETRAIN_DIR, PRETRAIN_STEPS, REF039, TRAIN_CAP6, VARIANTS, Lease, build_model,
                         label, lib_path, load_model, log, read_jsonl, utc, write_json)

# ---- 039 knobs (source/pretrain.py)
BS, LR, MIN_LR_FRAC, WARMUP, WD = 256, 1e-3, 0.1, 200, 0.1
MUON_LR, MUON_MOMENTUM, MUON_STEPS = 0.01, 0.95, 5
MUON_WD = WD * LR / MUON_LR
ADAMW_FINISH_FRAC = 0.95
VAL_N = 1024
CURVE_EVERY = 150          # steps between curve points (039: every 15 s, ~145 steps on its A40)
MAXN = 64


# Muon update adapted from KellerJordan/Muon, commit f98f1cacc0263b04290753e32be8d498c1efc806 (muon.py), MIT
# License, Copyright (c) 2024 Keller Jordan; copied unchanged from 039's pretrain.py.
def muon_step(params, buffers, lr):
    import torch
    with torch.no_grad():
        for p, momentum in zip(params, buffers, strict=True):
            if p.grad is None:
                continue
            momentum.lerp_(p.grad, 1 - MUON_MOMENTUM)
            update = p.grad.lerp(momentum, MUON_MOMENTUM).bfloat16()
            if update.size(0) == 3 * update.size(1):
                update = update.reshape(3, update.size(1), update.size(1))
            transpose = update.size(-2) > update.size(-1)
            if transpose:
                update = update.mT
            update = update / (update.norm(dim=(-2, -1), keepdim=True) + 1e-7)
            for _ in range(MUON_STEPS):
                gram = update @ update.mT
                polynomial = -4.7750 * gram + 2.0315 * gram @ gram
                update = 3.4445 * update + polynomial @ update
            if transpose:
                update = update.mT
            update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
            p.mul_(1 - lr * MUON_WD)
            p.add_(update.reshape_as(p), alpha=-lr)


def loss_on(model, x, m, pos=None):
    """The fork's train.loss_on, with optional explicit positions (B, T) for x."""
    import torch.nn.functional as F
    logits = model(x[:, :-1], pos=None if pos is None else pos[:, :-1])
    tgt, mk = x[:, 1:], m[:, 1:]
    l = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1), reduction='none')
    return (l * mk.reshape(-1)).sum() / mk.sum()


def to_gpu(data, tok, device):
    import torch
    N, T = len(data), max(len(p) + len(q) for p, q in data)
    ids = torch.full((N, T), tok.pad, dtype=torch.long)
    mask = torch.zeros((N, T), dtype=torch.bool)
    ref_max = torch.zeros(N, dtype=torch.long)
    for i, (p, q) in enumerate(data):
        ids[i, :len(p) + len(q)] = torch.tensor(p + q)
        mask[i, len(p):len(p) + len(q)] = True
        ref_max[i] = max((x - tok.ref0 + 1 for x in q if x >= tok.ref0), default=0)
    return ids.to(device), mask.to(device), ref_max.to(device), (ids != tok.pad).sum(1).to(device)


def gpu_batch(ids, mask, ref_max, lens, idxs, tok, gen):
    import torch
    x, m = ids[idxs], mask[idxs]
    T = int(lens[idxs].max())
    x, m = x[:, :T].clone(), m[:, :T]
    if tok.mode == 'abs':
        room = MAXN - ref_max[idxs]
        shift = (torch.rand(len(idxs), device=x.device, generator=gen) * (room + 1)).long().clamp(max=room)
        is_ref = (x >= tok.ref0) & m
        x = torch.where(is_ref, x + shift[:, None], x)
    return x, m, lens[idxs]


def val_loss(model, ids, mask, lens, randpos_gen=None):
    """039's val_loss (first VAL_N records, no shift); randpos_gen: posrand's random positions instead of 0.."""
    import torch
    from posenc import batch_positions
    was = model.training
    model.eval()
    tot = 0.0
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=ids.is_cuda):
        for i in range(0, VAL_N, BS):
            idxs = torch.arange(i, min(i + BS, VAL_N), device=ids.device)
            T = int(lens[idxs].max())
            x, m = ids[idxs][:, :T], mask[idxs][:, :T]
            pos = batch_positions(model, x, lens[idxs], randpos_gen) if randpos_gen is not None else None
            tot += loss_on(model, x, m, pos).item() * len(idxs)
    model.train(was)
    return tot / VAL_N


def heldout_metrics(model, tok, device, randpos=False, seed=0):
    """(039 greedy rate on heldout[::2][:2000], frozen-harness greedy rate on heldout[::5][:1000]) and, with
    randpos, the 039 slice under random positions (sampler.py pos_tables, rand_tables of evaluate.py)."""
    lib_path()
    from nd_verify import verify_text
    from sample import generate
    held = read_jsonl(HELDOUT_CAP6)
    s039 = held[::len(held) // 2000][:2000]
    sfz = held[::5][:1000]
    model.eval()
    out = {}
    for name, recs, batch in (('heldout_greedy_039', s039, 1024), ('heldout_greedy', sfz, 384)):
        t0 = time.time()
        outs = generate(model, tok, [r['prompt'] for r in recs], greedy=True, max_new=512, batch=batch)
        out[name] = sum(verify_text(r['prompt'] + ' ' + p)[0] for r, p in zip(recs, outs)) / len(recs)
        out[name + '_secs'] = time.time() - t0
    if randpos:
        from evaluate import rand_tables
        from sampler import sample_ids
        prompts = [r['prompt'] for r in s039]
        tabs = rand_tables(model, tok, prompts, [r['n_lines'] for r in s039], 512, seed)
        ids = sample_ids(model, tok, prompts, max_new=512, batch=512, seed=seed, greedy=True, pos_tables=tabs)
        out['heldout_greedy_randpos'] = sum(verify_text(r['prompt'] + ' ' + tok.decode(o))[0] for r, o in zip(s039, ids)) / len(s039)
    return out


def pretrain(variant, seed, steps, device, smoke=False):
    import torch
    torch.manual_seed(seed)
    random.seed(seed)
    recs = read_jsonl(TRAIN_CAP6)
    assert all(r['n_lines'] <= 6 for r in recs), 'pretrain file violates cap 6'
    t0 = time.time()
    model, tok = build_model(variant, 'cpu')          # 039: GPT(...).to(device) after the harness's manual_seed
    model = model.to(device)
    from posenc import batch_positions, needs_positions
    data = [(tok.encode_prompt(r['prompt']), tok.encode_proof(r['proof'])) for r in recs]
    ids, mask, ref_max, lens = to_gpu(data, tok, device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    pgen = None
    if needs_positions(model):
        pgen = torch.Generator(device=device)
        pgen.manual_seed(seed + 1000)
    log(f'pretrain {variant} s{seed}: {model.n_params():,} params, cfg {model.cfg}, tok {tok.mode}, {len(data):,} records, '
        f'{steps} steps, max seq {int(lens.max())} tokens')
    matrices = [p for p in model.blocks.parameters() if p.ndim == 2]
    matrix_ids = {id(p) for p in matrices}
    auxiliary = [p for p in model.parameters() if id(p) not in matrix_ids]
    momentum_buffers = [torch.zeros_like(p) for p in matrices]
    opt = torch.optim.AdamW(auxiliary, lr=LR, weight_decay=WD, betas=(0.9, 0.95))
    adamw_finish = False
    autocast = torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda'))
    model.train()
    perm, step, last_loss = torch.empty(0, dtype=torch.long, device=device), 0, float('nan')
    curve, recent = [], []
    n_train = len(data) - VAL_N
    finish_step = None
    while True:
        frac = step / steps              # 039: (time.time() - t0) / budget_secs
        if frac >= 1.0:
            break
        if not adamw_finish and frac >= ADAMW_FINISH_FRAC:
            opt.add_param_group({'params': matrices})
            adamw_finish = True
            finish_step = step + 1
        step += 1
        lr = LR * step / WARMUP if step < WARMUP else LR * MIN_LR_FRAC + 0.5 * LR * (1 - MIN_LR_FRAC) * (1 + math.cos(math.pi * frac))
        for g in opt.param_groups:
            g['lr'] = lr
        if len(perm) < BS:
            perm = torch.randperm(n_train, device=device, generator=gen) + VAL_N
        idxs, perm = perm[:BS], perm[BS:]
        x, m, bl = gpu_batch(ids, mask, ref_max, lens, idxs, tok, gen)
        pos = batch_positions(model, x, bl, pgen) if pgen is not None else None
        with autocast:
            loss = loss_on(model, x, m, pos)
        model.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if not adamw_finish:
            muon_step(matrices, momentum_buffers, MUON_LR * lr / LR)
        if step % 20 == 0:
            last_loss = loss.item()
            recent.append(last_loss)
        if step % CURVE_EVERY == 0 or step == steps:
            curve.append({'secs': round(time.time() - t0), 'step': step,
                          'train_loss': sum(recent) / len(recent) if recent else last_loss,
                          'val_loss': val_loss(model, ids, mask, lens)})
            log(f"pretrain step {step} train {curve[-1]['train_loss']:.4f} val {curve[-1]['val_loss']:.4f} lr {lr:.2e} "
                f"{curve[-1]['secs']}s")
            recent = []
    if device == 'cuda':
        torch.cuda.synchronize()
    secs = time.time() - t0
    model.eval()
    stats = {'steps': step, 'records_seen': step * BS, 'epochs': step * BS / n_train, 'last_loss': last_loss,
             'adamw_from_step': finish_step, 'train_secs': secs, 'steps_per_sec': step / secs, 'curve': curve,
             'val_loss': val_loss(model, ids, mask, lens)}
    if pgen is not None:
        vg = torch.Generator(device=device)
        vg.manual_seed(12345)
        stats['val_loss_randpos'] = val_loss(model, ids, mask, lens, vg)
    return model, tok, stats


def ref_metrics(seed, device, out):
    """The released 039 checkpoint of `seed`, measured with the same code on this hardware."""
    import torch
    path = os.path.join(REF039.format(seed=seed), 'pretrain.pt')
    model, tok, ck = load_model(path, device)
    recs = read_jsonl(TRAIN_CAP6)
    data = [(tok.encode_prompt(r['prompt']), tok.encode_proof(r['proof'])) for r in recs[:VAL_N]]
    ids, mask, ref_max, lens = to_gpu(data, tok, device)
    summ = json.load(open(os.path.join(REF039.format(seed=seed), 'summary.json')))
    res = {'ckpt': path, 'seed': seed, 'val_loss': val_loss(model, ids, mask, lens), **heldout_metrics(model, tok, device),
           'released_summary': {'steps': summ['pretrain']['steps'], 'secs': summ['pretrain'].get('secs'),
                                'gpu': summ.get('gpu'), 'heldout_greedy': summ['pretrain'].get('heldout_greedy'),
                                'final_val_loss': summ['pretrain']['curve'][-1]['val_loss'],
                                'n_params': summ['pretrain'].get('n_params')}}
    write_json(os.path.join(out, 'summary.json'), dict(res, status='ok', finished_utc=utc()))
    log(f"ref039 s{seed}: val {res['val_loss']:.4f} heldout_greedy_039 {res['heldout_greedy_039']:.4f} "
        f"heldout_greedy {res['heldout_greedy']:.4f} (released: {res['released_summary']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=sorted(VARIANTS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=PRETRAIN_STEPS)
    ap.add_argument('--out', default=None)
    ap.add_argument('--ref', type=int, default=None, help='measure the released 039 checkpoint of this seed instead')
    ap.add_argument('--gpu', default='auto')
    ap.add_argument('--gpus', default='0,1,2,3,4')
    ap.add_argument('--gpu-wait', type=float, default=12 * 3600)
    ap.add_argument('--smoke', action='store_true', help='allow a non-standard step count (tests)')
    a = ap.parse_args()
    if a.ref is None and (a.variant is None or (a.steps != PRETRAIN_STEPS and not a.smoke)):
        ap.error('--variant is required, and --steps may differ from PRETRAIN_STEPS only with --smoke')
    os.environ.setdefault('AB_WORKER', 'repr')
    out = a.out or os.path.join(PRETRAIN_DIR, f'ref039_s{a.ref}' if a.ref is not None else label(a.variant, a.seed))
    os.makedirs(out, exist_ok=True)
    lease = Lease(f'repr_pretrain_{os.path.basename(out)}')
    try:
        lease.take(a.gpu, a.gpus, a.gpu_wait)
        import torch
        dev = 'cuda'
        gpu = f"{os.environ.get('CUDA_VISIBLE_DEVICES', '')}:{torch.cuda.get_device_name(0)}"
        if a.ref is not None:
            return ref_metrics(a.ref, dev, out)
        started = utc()
        model, tok, stats = pretrain(a.variant, a.seed, a.steps, dev, smoke=a.smoke)
        lib_path()
        from model import save_ckpt
        extra = {'recipe': '039_muon-adamw-finish-5pct, fixed steps', 'variant': a.variant, 'seed': a.seed,
                 'steps': stats['steps'], 'n_params': model.n_params()}
        tmp = os.path.join(out, 'pretrain.pt.tmp')
        save_ckpt(tmp, model, tok.mode, extra=extra)
        os.replace(tmp, os.path.join(out, 'pretrain.pt'))
        torch.cuda.empty_cache()             # training memory back before the batch-1024 greedy check
        th = time.time()
        hm = heldout_metrics(model, tok, dev, randpos=(model.pos_train == 'rand'), seed=a.seed)
        summ = {'status': 'ok', 'variant': a.variant, 'seed': a.seed, 'cfg': model.cfg, 'tok_mode': tok.mode,
                'n_params': model.n_params(), 'gpu': gpu, 'started_utc': started, 'finished_utc': utc(),
                'heldout_secs': time.time() - th, **{k: v for k, v in stats.items()}, **hm, 'smoke': a.smoke}
        write_json(os.path.join(out, 'summary.json'), summ)
        log(f"SUMMARY {a.variant} s{a.seed}: {stats['steps']} steps in {stats['train_secs']:.0f}s "
            f"({stats['steps_per_sec']:.2f}/s), val {stats['val_loss']:.4f}, heldout_greedy_039 {hm['heldout_greedy_039']:.4f}, "
            f"heldout_greedy {hm['heldout_greedy']:.4f}" + (f", randpos val {stats['val_loss_randpos']:.4f} greedy "
                                                          f"{hm['heldout_greedy_randpos']:.4f}" if 'val_loss_randpos' in stats else ''))
    finally:
        lease.drop()
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
