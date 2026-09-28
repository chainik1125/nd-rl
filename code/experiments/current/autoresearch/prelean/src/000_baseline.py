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
  stats        dict of anything you want recorded in summary.json (steps, val_loss, ...)

Anything imported from the fork (ND_FORK) is fair game to copy in here and modify.
"""
import math, random, sys, time

import torch
from model import GPT                    # fork
from tokenizer import Tokenizer          # fork
from train import batch as mk_batch, loss_on   # fork

# ---- knobs
N_LAYER, D, N_HEAD = 4, 256, 8
BS, LR, MIN_LR_FRAC, WARMUP, WD = 256, 1e-3, 0.1, 200, 0.1
TOK_MODE = 'abs'


def pretrain(train_recs, budget_secs, device, seed, log):
    t0 = time.time()
    rng = random.Random(seed)
    tok = Tokenizer(TOK_MODE)
    data = [(tok.encode_prompt(r['prompt']), tok.encode_proof(r['proof'])) for r in train_recs]
    model = GPT(tok.vocab_size, N_LAYER, D, N_HEAD).to(device)
    log(f'pretrain: {model.n_params():,} params, {len(data):,} records, budget {budget_secs}s')
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.95))
    autocast = torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda'))
    model.train()
    perm, step, last_loss = [], 0, float('nan')
    while True:
        frac = (time.time() - t0) / budget_secs      # cosine over wall-clock, so the cap is the schedule
        if frac >= 1.0:
            break
        step += 1
        lr = LR * step / WARMUP if step < WARMUP else LR * MIN_LR_FRAC + 0.5 * LR * (1 - MIN_LR_FRAC) * (1 + math.cos(math.pi * frac))
        for g in opt.param_groups:
            g['lr'] = lr
        if len(perm) < BS:
            perm = list(range(len(data)))
            rng.shuffle(perm)
        idxs = [perm.pop() for _ in range(BS)]
        x, m = mk_batch(data, idxs, tok, rng, device)
        with autocast:
            loss = loss_on(model, x, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0:
            last_loss = loss.item()
            log(f'pretrain step {step} loss {last_loss:.4f} lr {lr:.2e} {time.time() - t0:.0f}s')
    model.eval()
    return model, tok, {'steps': step, 'records_seen': step * BS, 'epochs': step * BS / len(data), 'last_loss': last_loss}
