# imports the fork (model, tokenizer, train, ...), which is not resolvable here
# pyright: reportMissingImports=false
"""AGENT-EDITABLE: the pretraining stage. This is the only file the autoresearch agent changes.

Contract with harness.py (frozen):

  pretrain(train_recs, budget_secs, device, seed, log) -> (model, tok, stats)

  train_recs   list of dicts {prompt, proof, n_lines<=6, key, ...}; the frozen cap-6 Stage-1 data
  budget_secs  wall-clock you are charged for, from call to return, INCLUDING model construction
               and torch.compile. Overrun by more than the harness grace -> the run is a timeout.
  model        nn.Module; <= PARAM_CAP unique params. The harness fine-tunes it (AdamW on
               model.parameters()) and samples from it with the fork's sample.generate, so it must:
                 - forward(idx) -> logits (B,T,V)  and  forward(idx, pos=, mask=, caches=) with a
                   per-block KV cache dict, exactly like the fork's GPT (model.blocks must exist)
                 - be trainable in bf16 autocast on cuda
  tok          must provide encode_prompt, encode_proof, decode, shift_abs, pad, eos, mode.
               Decoder rule: decode() is deterministic and single-pass; it may not call the
               verifier or try alternatives.
  stats        dict recorded in summary.json and copied into results.json. MUST contain `curve`:
               a list of {secs, step, train_loss, val_loss} taken every ~CURVE_EVERY seconds
               (see log_point below), so a run's trajectory survives in git. Add anything else.
               val_loss is on VAL_N records held out from train_recs by this file; keep that.

Anything imported from the fork (ND_FORK) is fair game to copy in here and modify.
"""

import math
import time

import torch
from model import GPT  # fork
from tokenizer import MAXN, Tokenizer  # fork
from train import loss_on  # fork

# ---- knobs
N_LAYER, D, N_HEAD = 4, 256, 8
BS, LR, MIN_LR_FRAC, WARMUP, WD = 256, 1e-3, 0.1, 200, 0.1
TOK_MODE = "abs"
VAL_N, CURVE_EVERY = 1024, 15  # records held out for val_loss; seconds between curve points


def val_loss(model, ids, mask, ref_max, lens, tok, gen, autocast):
    """Mean loss over the first VAL_N records (never trained on), no shift augmentation."""
    model.eval()
    tot = 0.0
    with torch.no_grad(), autocast:
        for i in range(0, VAL_N, BS):
            idxs = torch.arange(i, min(i + BS, VAL_N), device=ids.device)
            x, m = ids[idxs], mask[idxs]
            T = int(lens[idxs].max())
            tot += loss_on(model, x[:, :T], m[:, :T]).item() * len(idxs)
    model.train()
    return tot / VAL_N


def to_gpu(data, tok, device):
    """Pad all (prompt, proof) records once into (N, T) ids + proof mask on the device."""
    N, T = len(data), max(len(p) + len(q) for p, q in data)
    ids = torch.full((N, T), tok.pad, dtype=torch.long)
    mask = torch.zeros((N, T), dtype=torch.bool)
    ref_max = torch.zeros(
        N, dtype=torch.long
    )  # largest N<k> index used in the proof, for the shift augmentation
    for i, (p, q) in enumerate(data):
        ids[i, : len(p) + len(q)] = torch.tensor(p + q)
        mask[i, len(p) : len(p) + len(q)] = True
        ref_max[i] = max((x - tok.ref0 + 1 for x in q if x >= tok.ref0), default=0)
    return ids.to(device), mask.to(device), ref_max.to(device), (ids != tok.pad).sum(1).to(device)


def gpu_batch(ids, mask, ref_max, lens, idxs, tok, gen):
    """The fork's batch() + tok.shift_abs, vectorised: a random per-row offset is added to
    every ref token."""
    x, m = ids[idxs], mask[idxs]
    T = int(lens[idxs].max())
    x, m = x[:, :T].clone(), m[:, :T]
    if tok.mode == "abs":
        room = MAXN - ref_max[idxs]
        shift = (
            (torch.rand(len(idxs), device=x.device, generator=gen) * (room + 1))
            .long()
            .clamp(max=room)
        )
        is_ref = (x >= tok.ref0) & m
        x = torch.where(is_ref, x + shift[:, None], x)
    return x, m


def pretrain(train_recs, budget_secs, device, seed, log):
    t0 = time.time()
    tok = Tokenizer(TOK_MODE)
    data = [(tok.encode_prompt(r["prompt"]), tok.encode_proof(r["proof"])) for r in train_recs]
    ids, mask, ref_max, lens = to_gpu(data, tok, device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    model = GPT(tok.vocab_size, N_LAYER, D, N_HEAD).to(device)
    log(
        f"pretrain: {model.n_params():,} params, {len(data):,} records on {device} "
        f"after {time.time() - t0:.0f}s, budget {budget_secs}s"
    )
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.99))
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda"))
    model.train()
    perm, step, last_loss = torch.empty(0, dtype=torch.long, device=device), 0, float("nan")
    curve, recent, next_point = [], [], CURVE_EVERY
    n_train = len(data) - VAL_N  # records [0, VAL_N) are the val slice
    while True:
        frac = (
            time.time() - t0
        ) / budget_secs  # cosine over wall-clock, so the cap is the schedule
        if frac >= 1.0:
            break
        step += 1
        lr = (
            LR * step / WARMUP
            if step < WARMUP
            else LR * MIN_LR_FRAC + 0.5 * LR * (1 - MIN_LR_FRAC) * (1 + math.cos(math.pi * frac))
        )
        for g in opt.param_groups:
            g["lr"] = lr
        if len(perm) < BS:
            perm = torch.randperm(n_train, device=device, generator=gen) + VAL_N
        idxs, perm = perm[:BS], perm[BS:]
        x, m = gpu_batch(ids, mask, ref_max, lens, idxs, tok, gen)
        with autocast:
            loss = loss_on(model, x, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 20 == 0:
            last_loss = loss.item()
            recent.append(last_loss)
        if (
            time.time() - t0 >= next_point
        ):  # curve point: mean recent train loss + val loss on the held-out slice
            curve.append(
                {
                    "secs": round(time.time() - t0),
                    "step": step,
                    "train_loss": sum(recent) / len(recent) if recent else last_loss,
                    "val_loss": val_loss(model, ids, mask, ref_max, lens, tok, gen, autocast),
                }
            )
            log(
                f"pretrain step {step} train {curve[-1]['train_loss']:.4f} "
                f"val {curve[-1]['val_loss']:.4f} lr {lr:.2e} {curve[-1]['secs']}s"
            )
            recent, next_point = [], next_point + CURVE_EVERY
    model.eval()
    return (
        model,
        tok,
        {
            "steps": step,
            "records_seen": step * BS,
            "epochs": step * BS / n_train,
            "last_loss": last_loss,
            "curve": curve,
        },
    )
