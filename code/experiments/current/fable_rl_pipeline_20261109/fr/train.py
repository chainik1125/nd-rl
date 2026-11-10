"""Loss, optimiser and the stage-1 (pretraining on x) entry point.

python -m fr.train --x /workspace/data/train_k12.jsonl.gz --out /workspace/runs/pre_s0 --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from .common import VOCAB
from .data import Mixer, load_x, make_batch, shift_names
from .model import GPT, save_ckpt


def losses(model, b, gen=None, shift=True):
    """-> (policy loss, verifier loss, value loss, counts). Each is a mean over its own targets
    (0 when a batch has none)."""
    x = shift_names(b["x"], gen) if shift else b["x"]
    ln, nst, lab = b["len"], b["nst"], b["lab"]
    logits, aux = model(x[:, :-1])
    T = x.shape[1] - 1
    t = torch.arange(T, device=x.device)[None]
    pm = (t + 1 >= nst[:, None]) & (t + 1 < ln[:, None]) & (lab[:, 0:1] == 1)
    ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1),
                         reduction="none").view_as(pm)  # fmt: skip
    npol = pm.sum()
    lpol = (ce * pm).sum() / npol.clamp(min=1)
    aux = aux.float()
    rows = torch.arange(x.shape[0], device=x.device)
    vlogit = aux[rows, (ln - 2).clamp(min=0), 0]
    vm = lab[:, 1] >= 0
    lver = (F.binary_cross_entropy_with_logits(vlogit, lab[:, 1].clamp(min=0).float(),
                                               reduction="none") * vm).sum() / vm.sum().clamp(min=1)  # fmt: skip
    qlogit = aux[rows, (nst - 1).clamp(min=0), 1]
    qm = lab[:, 2] >= 0
    lval = (F.binary_cross_entropy_with_logits(qlogit, lab[:, 2].clamp(min=0).float(),
                                               reduction="none") * qm).sum() / qm.sum().clamp(min=1)  # fmt: skip
    with torch.no_grad():
        vacc = (((vlogit > 0) == (lab[:, 1] == 1)) & vm).sum()
        vneg = (vm & (lab[:, 1] == 0)).sum()
        vneg_ok = ((vlogit <= 0) & vm & (lab[:, 1] == 0)).sum()
    stats = {"npol": int(npol), "nver": int(vm.sum()), "nval": int(qm.sum()), "vacc": int(vacc),
             "vneg": int(vneg), "vneg_ok": int(vneg_ok)}  # fmt: skip
    return lpol, lver, lval, stats


def make_opt(model, lr, wd=0.1):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd}, {"params": no, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95), fused=torch.cuda.is_available(),
    )  # fmt: skip


def fit(model, opt, mixer, steps, lr_fn, *, w_ver=0.5, w_val=0.25, log=print, log_every=100,
        gen=None, tag=""):  # fmt: skip
    """`steps` optimiser steps on batches from `mixer`; lr_fn(step index from 0) -> lr."""
    dev = next(model.parameters()).device
    cuda = dev.type == "cuda"
    model.train()
    t0 = time.time()
    acc = {"pol": 0.0, "ver": 0.0, "val": 0.0, "n": 0, "tok": 0, "vacc": 0, "nver": 0,
           "vneg": 0, "vneg_ok": 0}  # fmt: skip
    for s in range(steps):
        b = mixer.next(dev)
        for gp in opt.param_groups:
            gp["lr"] = lr_fn(s)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            lp, lv, lq, st = losses(model, b, gen)
        loss = lp + w_ver * lv + w_val * lq
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        acc["pol"] += float(lp)
        acc["ver"] += float(lv)
        acc["val"] += float(lq)
        acc["n"] += 1
        acc["tok"] += int(b["len"].sum())
        for k in ("vacc", "nver", "vneg", "vneg_ok"):
            acc[k] += st[k]
        if (s + 1) % log_every == 0 or s + 1 == steps:
            n = acc["n"]
            msg = (f"{tag}step {s + 1}/{steps} pol {acc['pol'] / n:.4f} ver {acc['ver'] / n:.4f} "
                   f"val {acc['val'] / n:.4f} lr {lr_fn(s):.2e} tok/s {acc['tok'] / (time.time() - t0):.0f}")  # fmt: skip
            if acc["nver"]:
                msg += (f" ver_acc {acc['vacc'] / acc['nver']:.4f} "
                        f"neg_recall {acc['vneg_ok'] / max(1, acc['vneg']):.4f}")  # fmt: skip
            log(msg)
            last = dict(acc)
            acc = dict.fromkeys(acc, 0)
            acc["pol"] = acc["ver"] = acc["val"] = 0.0
            t0 = time.time()
    model.eval()
    return last


@torch.no_grad()
def val_loss(model, store, dev, n=4000, bs=256):
    model.eval()
    cuda = dev.type == "cuda"
    tot, cnt = 0.0, 0
    import numpy as np

    idx = np.arange(min(n, len(store)))
    for i in range(0, len(idx), bs):
        b = make_batch([(store, idx[i : i + bs])], dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            lp, _, _, st = losses(model, b, shift=False)
        tot += float(lp) * st["npol"]
        cnt += st["npol"]
    return tot / max(1, cnt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--tok_budget", type=int, default=120_000)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--min_lr", type=float, default=8e-5)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--n_layer", type=int, default=8)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--val_every", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "train.log"), "a")

    def log(m):
        line = f"{time.strftime('%FT%TZ', time.gmtime())} {m}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = a.cache or os.path.join(os.path.dirname(a.x), "x_cache.pkl")
    t0 = time.time()
    tr, va, meta = load_x(a.x, cache)
    log(f"x: {json.dumps(meta)} load {time.time() - t0:.0f}s")
    model = GPT(VOCAB, a.n_layer, a.d, a.n_head).to(dev)
    log(f"params {model.n_params()} args {json.dumps(vars(a))}")
    opt = make_opt(model, a.lr)
    mixer = Mixer([(tr, 1.0)], tok_budget=a.tok_budget, seed=a.seed)
    gen = torch.Generator(device=dev)
    gen.manual_seed(a.seed)

    def lr_fn(s, base=0):
        s = s + base
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        p = (s - a.warmup) / max(1, a.steps - a.warmup)
        return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * p))

    t_start = time.time()
    done = 0
    while done < a.steps:
        n = min(a.val_every, a.steps - done)
        fit(model, opt, mixer, n, lambda s, base=done: lr_fn(s, base), log=log, gen=gen,
            tag=f"[{done}] ")  # fmt: skip
        done += n
        log(f"val_loss {val_loss(model, va, dev):.4f} at step {done} "
            f"({(time.time() - t_start) / 60:.1f} min)")  # fmt: skip
        save_ckpt(os.path.join(a.out, "pre.pt"), model,
                  {"stage": "pretrain", "step": done, "args": vars(a), "x_meta": meta,
                   "secs": time.time() - t_start})  # fmt: skip
    log("PRETRAIN_DONE")


if __name__ == "__main__":
    main()
