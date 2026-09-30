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
  tok          must provide encode_prompt, encode_proof, decode, shift_abs, pad, eos, mode, and
               mode must be "lean_seq" (harness TOK_MODE): Lean text in and out, Lean-gated samples.
               Decoder rule: decode() is deterministic and single-pass; it may not call the
               verifier or try alternatives.
  stats        dict recorded in summary.json and copied into results.json. MUST contain `curve`:
               a list of {secs, step, train_loss, val_loss} taken every ~CURVE_EVERY seconds,
               so a run's trajectory survives in git. Add anything else.
               val_loss is on VAL_N records held out from train_recs by this file; keep that.

Anything imported from the fork (ND_FORK) is fair game to copy in here and modify.
"""

import math
import os
import random
import time

import torch
from model import GPT, Block  # fork
from state_pairs import batch, pairs  # train.batch, plus state-mode pairs
from tokenizer import make_tokenizer  # fork
from train import loss_on  # fork

N_LAYER, D, N_HEAD = 6, 384, 8
D_FF = 1280
BS, LR, MIN_LR_FRAC, WD = 128, 1e-3, 0.3, 0.1
MUON_LR, MUON_MOMENTUM = 0.01, 0.95
MUON_WD = 10 * WD * LR / MUON_LR
# Polar Express (Amsel et al. 2025) per-iteration quintic coefficients, as in modded-nanogpt.
NS_COEFFS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)
TOK_MODE = os.environ.get("ND_TOK_MODE", "lean_seq")  # state arms: lean_staten / lean_stateh
# P34 x P32: ALiBi on half the heads (slopes 2^-2 .. 2^-8), no positional signal on the rest.
N_ALIBI = N_HEAD // 2
ALIBI_SLOPES = tuple(2.0 ** (-8.0 * (i + 1) / N_ALIBI) for i in range(N_ALIBI)) + (0.0,) * (
    N_HEAD - N_ALIBI
)
VAL_N, CURVE_EVERY = 1024, 15  # records held out for val_loss; seconds between curve points


# Muon update adapted from KellerJordan/Muon, commit
# f98f1cacc0263b04290753e32be8d498c1efc806 (muon.py).
# MIT License
# Copyright (c) 2024 Keller Jordan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
@torch.no_grad()
def muon_step(params, buffers, lr):
    """Nesterov momentum plus five bf16 Polar Express steps, on hidden matrices only."""
    for p, momentum in zip(params, buffers, strict=True):
        momentum.lerp_(p.grad, 1 - MUON_MOMENTUM)
        update = p.grad.lerp(momentum, MUON_MOMENTUM).bfloat16()
        # Only QKV has shape (3D, D) in this GPT; orthogonalize its roles separately.
        if update.size(0) == 3 * update.size(1):
            update = update.reshape(3, update.size(1), update.size(1))
        transpose = update.size(-2) > update.size(-1)
        if transpose:
            update = update.mT
        update = update / (update.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)
        for a, b, c in NS_COEFFS:
            gram = update @ update.mT
            update = a * update + (b * gram + c * gram @ gram) @ update
        if transpose:
            update = update.mT
        p.mul_(1 - lr * MUON_WD)
        p.add_(update.reshape_as(p).to(p.dtype), alpha=-lr)


class PeriLNBlock(Block):
    """P22 Peri-LN (Kim et al. 2025): the fork's pre-LN block plus a LayerNorm on each branch
    output, so every residual write has unit scale times a learned gain. Attention has no RoPE:
    `mask` is the additive ALiBi bias (B or 1, H, T, S) that ALiBiGPT builds once per forward
    and cos/sin are unused."""

    def __init__(self, d, h):
        super().__init__(d, h)
        self.fc1, self.fc2 = torch.nn.Linear(d, D_FF), torch.nn.Linear(D_FF, d)
        self.ln_attn_out, self.ln_mlp_out = torch.nn.LayerNorm(d), torch.nn.LayerNorm(d)

    def attn(self, x, cos, sin, mask, cache):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if cache is not None:  # the fork's preallocated KV cache, unchanged
            if "k" not in cache:
                cache["k"] = k.new_empty(B, self.h, cache["max"], D // self.h)
                cache["v"] = v.new_empty(B, self.h, cache["max"], D // self.h)
                cache["n"] = 0
            n0, n1 = cache["n"], cache["n"] + T
            cache["k"][:, :, n0:n1] = k
            cache["v"][:, :, n0:n1] = v
            cache["n"] = n1
            k, v = cache["k"][:, :, :n1], cache["v"][:, :, :n1]
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask.to(q.dtype))
        return self.proj(y.transpose(1, 2).reshape(B, T, D))

    def forward(self, x, cos, sin, mask=None, cache=None):
        x = x + self.ln_attn_out(self.attn(self.ln1(x), cos, sin, mask, cache))
        return x + self.ln_mlp_out(self.fc2(torch.nn.functional.gelu(self.fc1(self.ln2(x)))))


class ALiBiGPT(GPT):
    """P34: the fork's GPT with ALiBi (Press et al. 2021) instead of RoPE. Head h's attention
    logits get -slope_h * (query position - key position), slopes 2^-1 .. 2^-8 for 8 heads, so
    no rotation of q/k anywhere. Positions come from `pos` (sample.generate's left-padded
    batches) or arange; key positions are kept in caches[0] so cached decoding sees the same
    bias as the full forward. Causality/padding come from `mask` when given, else from rel < 0."""

    def __init__(self, vocab, n_layer, d, n_head, max_len=1024):
        super().__init__(vocab, n_layer, d, n_head, max_len)
        slopes = torch.tensor(ALIBI_SLOPES).view(1, n_head, 1, 1)
        self.register_buffer("slopes", slopes, persistent=False)

    def forward(self, idx, pos=None, mask=None, caches=None):
        return self.head(self.ln_f(self.features(idx, pos, mask, caches)[0]))

    def features(self, idx, pos=None, mask=None, caches=None):
        """The trunk: (final hidden state before ln_f, the ALiBi bias used)."""
        B, T = idx.shape
        if pos is None:  # training/val: one (1, H, T, T) bias broadcast over the batch
            pos = torch.arange(T, device=idx.device)[None]
        key_pos = pos
        if caches is not None:  # incremental decoding: keys are every cached token
            c = caches[0]
            if "pos" not in c:
                c["pos"], c["n_pos"] = pos.new_empty(pos.size(0), c["max"]), 0
            n0, n1 = c["n_pos"], c["n_pos"] + T
            c["pos"][:, n0:n1] = pos
            c["n_pos"] = n1
            key_pos = c["pos"][:, :n1]
        rel = pos[:, None, :, None] - key_pos[:, None, None, :]  # (B or 1, 1, T, S)
        bias = -self.slopes * rel.to(self.slopes.dtype)
        bias = bias.masked_fill(rel < 0 if mask is None else ~mask, float("-inf"))
        x = self.emb(idx)
        for i, b in enumerate(self.blocks):
            x = b(x, None, None, bias, None if caches is None else caches[i])
        return x, bias


@torch.no_grad()
def val_loss(model, val, tok, rng, device, autocast):
    """Mean loss over the VAL_N held-out records, no shift augmentation."""
    model.eval()
    tok.shift, total = False, 0.0
    for i in range(0, len(val), BS):
        idxs = range(i, min(i + BS, len(val)))
        x, m = batch(val, idxs, tok, rng, device)
        with autocast:
            total += loss_on(model, x, m).item() * len(idxs)
    tok.shift = True
    model.train()
    return total / len(val)


def pretrain(train_recs, budget_secs, device, seed, log):
    t0 = time.time()
    tok = make_tokenizer(TOK_MODE)
    data = (
        pairs(train_recs, tok)
        if TOK_MODE != "lean_seq"
        else [(tok.encode_prompt(r["prompt"]), tok.encode_proof(r["proof"])) for r in train_recs]
    )
    val, train = data[:VAL_N], data[VAL_N:]
    rng = random.Random(seed)
    model = ALiBiGPT(tok.vocab_size, N_LAYER, D, N_HEAD)
    model.blocks = torch.nn.ModuleList(PeriLNBlock(D, N_HEAD) for _ in range(N_LAYER))
    model.blocks.apply(model._init)  # fork init: zero biases
    model = model.to(device)
    log(f"pretrain: {model.n_params():,} params, {len(data):,} records on {device}")
    # Hidden matrices and the readout on Muon; embeddings, biases and norms on AdamW.
    matrices = [p for p in model.blocks.parameters() if p.ndim == 2] + [model.head.weight]
    in_muon = {id(p) for p in matrices}
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if id(p) not in in_muon],
        lr=LR,
        weight_decay=WD,
        betas=(0.9, 0.95),
    )
    buffers = [torch.zeros_like(p) for p in matrices]
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda")
    model.train()
    step, curve, recent, next_point = 0, [], [], CURVE_EVERY
    while (frac := (time.time() - t0) / budget_secs) < 1:  # cosine over wall-clock, no warmup
        step += 1
        lr = LR * (MIN_LR_FRAC + (1 - MIN_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = lr
        x, m = batch(train, rng.sample(range(len(train)), BS), tok, rng, device)
        with autocast:
            loss = loss_on(model, x, m)
        model.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        muon_step(matrices, buffers, MUON_LR * lr / LR)
        if step % 20 == 0:
            recent.append(loss.item())
        if time.time() - t0 >= next_point:
            vl = val_loss(model, val, tok, rng, device, autocast)
            curve.append(
                {
                    "secs": round(time.time() - t0),
                    "step": step,
                    "train_loss": sum(recent) / len(recent) if recent else None,
                    "val_loss": vl,
                }
            )
            log(f"pretrain step {step} val {vl:.4f} lr {lr:.2e} {round(time.time() - t0)}s")
            recent, next_point = [], next_point + CURVE_EVERY
    model.eval()
    return model, tok, {"steps": step, "curve": curve}
