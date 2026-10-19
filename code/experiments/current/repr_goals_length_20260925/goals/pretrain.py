#!/usr/bin/env python3
"""Fixed-step pretraining of the goals variant with the 039 recipe: a copy of repr/pretrain.py (sha256 040fe71ff4ab...,
itself 039's source/pretrain.py line for line with frac = step / steps) with these edits only (DESIGN.md 4):
  1. data: data/train_cap6_goals.jsonl (frozen/data/train_cap6.jsonl, same order, proofs canonically annotated)
  2. tokenizer: lib/gtokenizer.GoalTokenizer (abs + GOAL, vocab 100); the model is the frozen lib GPT
  3. the abs index shift of 039's gpu_batch also applies in mode abs_goals (GOAL is below ref0: never shifted)
  4. held-out greedy checks strip annotations before verifying, and use max_new 768 (frozen / repr: 512)
plus one extra metric, val_loss_lines (the val loss on proof-line tokens and QED only, annotation tokens excluded).

    python pretrain.py --seed 0 [--steps 2911] [--out pretrain/goals_s0] [--gpu auto|env|N]
    python pretrain.py --seed 0 --steps 60 --out /tmp/x --gpu auto --smoke      # plumbing check

039 recipe: batch 256 records, lr 1e-3 (warmup 200 steps, cosine over frac to 0.1x), AdamW (0.9, 0.95), wd 0.1 on the
embeddings / layer norms / biases / head; Muon (lr 0.01 x schedule, momentum 0.95, 5 Newton-Schulz steps, QKV
orthogonalised per role, decoupled wd 0.01) on the 2-D block matrices; at frac >= 0.95 the block matrices join AdamW
with fresh moments; grad clip 1.0; bf16 autocast; the first 1,024 records held out for val loss; abs index shift per
record; seed -> torch.manual_seed (model init) and a CUDA generator (data order, shifts).
"""
import argparse, json, math, os, random, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common_goals import (GREEDY_MAX_NEW, HELDOUT_CAP6, PRETRAIN_DIR, PRETRAIN_STEPS, TRAIN_CAP6, Lease, build_model,
                          label, lib_path, log, read_jsonl, sha256_file, utc, write_json)

# ---- 039 knobs (source/pretrain.py)
BS, LR, MIN_LR_FRAC, WARMUP, WD = 256, 1e-3, 0.1, 200, 0.1
MUON_LR, MUON_MOMENTUM, MUON_STEPS = 0.01, 0.95, 5
MUON_WD = WD * LR / MUON_LR
ADAMW_FINISH_FRAC = 0.95
VAL_N = 1024
CURVE_EVERY = 150          # steps between curve points (as repr)
MAXN = 64
SHIFT_MODES = ('abs', 'abs_goals')   # edit 3


# Muon update adapted from KellerJordan/Muon, commit f98f1cacc0263b04290753e32be8d498c1efc806 (muon.py), MIT
# License, Copyright (c) 2024 Keller Jordan; copied unchanged from 039's pretrain.py (via repr/pretrain.py).
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


def loss_on(model, x, m):
    """The fork's train.loss_on."""
    import torch.nn.functional as F
    logits = model(x[:, :-1])
    tgt, mk = x[:, 1:], m[:, 1:]
    l = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1), reduction='none')
    return (l * mk.reshape(-1)).sum() / mk.sum()


def annotation_mask(q, tok):
    """1 for the tokens of annotation groups (GOAL and its formula) of an encoded canonical proof, else 0."""
    out, inside = [], False
    for x in q:
        if x == tok.goal:
            inside = True
        elif x >= tok.ref0 or x == tok.eos:
            inside = False
        out.append(1 if inside else 0)
    return out


def to_gpu(data, tok, device):
    import torch
    N, T = len(data), max(len(p) + len(q) for p, q in data)
    ids = torch.full((N, T), tok.pad, dtype=torch.long)
    mask = torch.zeros((N, T), dtype=torch.bool)
    ann = torch.zeros((N, T), dtype=torch.bool)
    ref_max = torch.zeros(N, dtype=torch.long)
    for i, (p, q) in enumerate(data):
        ids[i, :len(p) + len(q)] = torch.tensor(p + q)
        mask[i, len(p):len(p) + len(q)] = True
        ann[i, len(p):len(p) + len(q)] = torch.tensor(annotation_mask(q, tok), dtype=torch.bool)
        ref_max[i] = max((x - tok.ref0 + 1 for x in q if x >= tok.ref0), default=0)
    return ids.to(device), mask.to(device), ann.to(device), ref_max.to(device), (ids != tok.pad).sum(1).to(device)


def gpu_batch(ids, mask, ref_max, lens, idxs, tok, gen):
    import torch
    x, m = ids[idxs], mask[idxs]
    T = int(lens[idxs].max())
    x, m = x[:, :T].clone(), m[:, :T]
    if tok.mode in SHIFT_MODES:
        room = MAXN - ref_max[idxs]
        shift = (torch.rand(len(idxs), device=x.device, generator=gen) * (room + 1)).long().clamp(max=room)
        is_ref = (x >= tok.ref0) & m
        x = torch.where(is_ref, x + shift[:, None], x)
    return x, m, lens[idxs]


def val_loss(model, ids, mask, lens, ann=None):
    """039's val_loss (first VAL_N records, no shift); with ann: only proof tokens that are not annotation tokens."""
    import torch
    was = model.training
    model.eval()
    tot = 0.0
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=ids.is_cuda):
        for i in range(0, VAL_N, BS):
            idxs = torch.arange(i, min(i + BS, VAL_N), device=ids.device)
            T = int(lens[idxs].max())
            x, m = ids[idxs][:, :T], mask[idxs][:, :T]
            if ann is not None:
                m = m & ~ann[idxs][:, :T]
            tot += loss_on(model, x, m).item() * len(idxs)
    model.train(was)
    return tot / VAL_N


def heldout_metrics(model, tok, device):
    """(039 greedy rate on heldout[::2][:2000], frozen-harness greedy rate on heldout[::5][:1000]), verified after
    stripping annotations (edit 4), plus annotation statistics of the greedy proofs."""
    lib_path()
    from nd_verify import verify_text
    from sample import generate
    import goals as GL
    held = read_jsonl(HELDOUT_CAP6)
    s039 = held[::len(held) // 2000][:2000]
    sfz = held[::5][:1000]
    model.eval()
    out = {}
    for name, recs, batch in (('heldout_greedy_039', s039, 1024), ('heldout_greedy', sfz, 384)):
        t0 = time.time()
        outs = generate(model, tok, [r['prompt'] for r in recs], greedy=True, max_new=GREEDY_MAX_NEW, batch=batch)
        oks, cons, groups, bad = 0, [], 0, 0
        for r, p in zip(recs, outs):
            plain, ann, nb = GL.read_annotations(p)
            groups += sum(len(v) for v in ann.values())
            bad += nb
            ok = verify_text(r['prompt'] + ' ' + plain)[0]
            oks += ok
            if ok:
                c = GL.consistency(r['prompt'], p)
                cons.append(c['exact'])
        out[name] = oks / len(recs)
        out[name + '_secs'] = time.time() - t0
        out[name + '_annotation'] = {'groups_per_proof': groups / len(recs), 'malformed': bad,
                                      'exact_canonical_of_correct': sum(cons) / max(1, len(cons))}
    return out


def pretrain(seed, steps, device, smoke=False):
    import torch
    torch.manual_seed(seed)
    random.seed(seed)
    recs = read_jsonl(TRAIN_CAP6)
    assert all(r['n_lines'] <= 6 for r in recs), 'pretrain file violates cap 6'
    t0 = time.time()
    model, tok = build_model('cpu')          # 039: GPT(...).to(device) after the harness's manual_seed
    model = model.to(device)
    data = [(tok.encode_prompt(r['prompt']), tok.encode_proof(r['proof'])) for r in recs]
    ids, mask, ann, ref_max, lens = to_gpu(data, tok, device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    log(f'pretrain goals s{seed}: {model.n_params():,} params, cfg {model.cfg}, tok {tok.mode}, {len(data):,} records, '
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
    seen_tokens = seen_loss_tokens = seen_ann_tokens = seen_rows = 0
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
        with autocast:
            loss = loss_on(model, x, m)
        model.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if not adamw_finish:
            muon_step(matrices, momentum_buffers, MUON_LR * lr / LR)
        if step % 20 == 0:
            last_loss = loss.item()
            recent.append(last_loss)
            seen_tokens += int(bl.sum()) * 20                   # sampled every 20 steps (no per-step host sync)
            seen_loss_tokens += int(m[:, 1:].sum()) * 20
            seen_ann_tokens += int(ann[idxs][:, 1:x.size(1)].sum()) * 20
            seen_rows += x.size(0) * (x.size(1) - 1) * 20
        if step % CURVE_EVERY == 0 or step == steps:
            curve.append({'secs': round(time.time() - t0), 'step': step,
                          'train_loss': sum(recent) / len(recent) if recent else last_loss,
                          'val_loss': val_loss(model, ids, mask, lens),
                          'val_loss_lines': val_loss(model, ids, mask, lens, ann)})
            log(f"pretrain step {step} train {curve[-1]['train_loss']:.4f} val {curve[-1]['val_loss']:.4f} "
                f"val_lines {curve[-1]['val_loss_lines']:.4f} lr {lr:.2e} {curve[-1]['secs']}s")
            recent = []
    if device == 'cuda':
        torch.cuda.synchronize()
    secs = time.time() - t0
    model.eval()
    stats = {'steps': step, 'records_seen': step * BS, 'epochs': step * BS / n_train, 'last_loss': last_loss,
             'adamw_from_step': finish_step, 'train_secs': secs, 'steps_per_sec': step / secs, 'curve': curve,
             'val_loss': val_loss(model, ids, mask, lens), 'val_loss_lines': val_loss(model, ids, mask, lens, ann),
             'tokens_seen_est': seen_tokens, 'loss_tokens_seen_est': seen_loss_tokens,
             'annotation_loss_tokens_seen_est': seen_ann_tokens, 'padded_positions_est': seen_rows,
             'tokens_seen_note': 'sampled every 20th step x 20 (estimates)'}
    return model, tok, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=PRETRAIN_STEPS)
    ap.add_argument('--out', default=None)
    ap.add_argument('--gpu', default='auto')
    ap.add_argument('--gpus', default='0,1,2,3,4')
    ap.add_argument('--gpu-wait', type=float, default=12 * 3600)
    ap.add_argument('--smoke', action='store_true', help='allow a non-standard step count (tests)')
    a = ap.parse_args()
    if a.steps != PRETRAIN_STEPS and not a.smoke:
        ap.error('--steps may differ from PRETRAIN_STEPS only with --smoke')
    os.environ.setdefault('AB_WORKER', 'goals')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')   # allocator only (no numerical effect)
    out = a.out or os.path.join(PRETRAIN_DIR, label(a.seed))
    os.makedirs(out, exist_ok=True)
    lease = Lease(f'goals_pretrain_{os.path.basename(out)}')
    try:
        lease.take(a.gpu, a.gpus, a.gpu_wait)
        import torch
        dev = 'cuda'
        gpu = f"{os.environ.get('CUDA_VISIBLE_DEVICES', '')}:{torch.cuda.get_device_name(0)}"
        started = utc()
        model, tok, stats = pretrain(a.seed, a.steps, dev, smoke=a.smoke)
        lib_path()
        from model import save_ckpt
        extra = {'recipe': '039_muon-adamw-finish-5pct, fixed steps (goals copy of repr/pretrain.py)', 'variant': 'goals',
                 'seed': a.seed, 'steps': stats['steps'], 'n_params': model.n_params(), 'data': TRAIN_CAP6,
                 'data_sha256': sha256_file(TRAIN_CAP6)}
        tmp = os.path.join(out, 'pretrain.pt.tmp')
        save_ckpt(tmp, model, tok.mode, extra=extra)
        os.replace(tmp, os.path.join(out, 'pretrain.pt'))
        torch.cuda.empty_cache()
        th = time.time()
        hm = heldout_metrics(model, tok, dev)
        summ = {'status': 'ok', 'variant': 'goals', 'seed': a.seed, 'cfg': model.cfg, 'tok_mode': tok.mode,
                'n_params': model.n_params(), 'gpu': gpu, 'started_utc': started, 'finished_utc': utc(),
                'heldout_secs': time.time() - th, **stats, **hm, 'smoke': a.smoke,
                'source_sha256': sha256_file(os.path.abspath(__file__)), 'data_sha256': extra['data_sha256']}
        write_json(os.path.join(out, 'summary.json'), summ)
        log(f"SUMMARY goals s{a.seed}: {stats['steps']} steps in {stats['train_secs']:.0f}s ({stats['steps_per_sec']:.2f}/s), "
            f"val {stats['val_loss']:.4f} val_lines {stats['val_loss_lines']:.4f}, heldout_greedy_039 {hm['heldout_greedy_039']:.4f}, "
            f"heldout_greedy {hm['heldout_greedy']:.4f}")
    finally:
        lease.drop()
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
