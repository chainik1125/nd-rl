"""Robbie's "best" network and pretraining pieces, for the proof-state trainer (run `best-state`).

Copied from nd-rl `robbie-experiments`, `code/experiments/current/factorial_20260929/pretrain_best.py` (the
autoresearch incumbent 186-claude, sha f8451342; Robbie's factorial of 2026-09-29), which is built on this fork's
`GPT`.  What is his, unchanged in substance: the 6 x 384 Peri-LN network with hybrid ALiBi / NoPE heads and MLP width
1280, the Muon update (Polar Express coefficients, RMS-matched scaling, decoupled weight decay 10x the AdamW group's),
the orthogonal init, and the multi-token-prediction head at weight 0.3.  What changed for the state format:

  - `ALiBiGPT` builds its own blocks and records `arch='best'` in `cfg`, so `model.load_ckpt` can rebuild it and
    every loader (`state_sample`, `state_ladder_ei`, `lpool_reread`, `state_eval`) takes the checkpoint unchanged;
  - `mtp_loss` takes the loss mask of `state_train` (action tokens only), as his took the proof-token mask;
  - `muon_step` takes the model width instead of the module constant D.
The training loop that uses these is `state_train_best.py`.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GPT, Block

N_LAYER, D, N_HEAD, D_FF = 6, 384, 8, 1280
MUON_LR, MUON_MOMENTUM = 0.01, 0.95
MTP_WEIGHT = 0.3
# Polar Express (Amsel et al. 2025) per-iteration quintic coefficients, as in modded-nanogpt (num_iters=5, safety 2e-2).
NS_COEFFS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


def alibi_slopes(n_head):
    """Half the heads ALiBi (Press et al. 2021) with slopes 2^(-8 i / (H/2)), the other half slope 0 (NoPE)."""
    na = n_head // 2
    return tuple(2.0 ** (-8.0 * (i + 1) / na) for i in range(na)) + (0.0,) * (n_head - na)


# Muon update adapted from KellerJordan/Muon, commit f98f1cacc0263b04290753e32be8d498c1efc806 (muon.py).
# MIT License, Copyright (c) 2024 Keller Jordan.  Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
# following conditions: The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
@torch.no_grad()
def muon_step(params, buffers, lr, d, wd):
    """Nesterov momentum plus five bf16 Polar Express steps, on hidden matrices only."""
    for p, momentum in zip(params, buffers, strict=True):
        if p.grad is None:
            continue
        momentum.lerp_(p.grad, 1 - MUON_MOMENTUM)
        update = p.grad.lerp(momentum, MUON_MOMENTUM).bfloat16()
        if update.size(0) == 3 * update.size(1):    # only QKV has shape (3D, D): orthogonalise its roles separately
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
        update *= (max(update.size(-2), update.size(-1)) / d) ** 0.5    # RMS-matched scaling (Moonshot)
        p.mul_(1 - lr * wd)
        p.sub_(lr * update.reshape_as(p).to(p.dtype))


class PeriLNBlock(Block):
    """Peri-LN (Kim et al. 2025): the pre-LN block plus a LayerNorm on each branch output.  No RoPE: `mask` is the
    additive ALiBi bias built once per forward by ALiBiGPT; cos/sin are unused."""

    def __init__(self, d, h, d_ff=D_FF):
        super().__init__(d, h)
        self.fc1, self.fc2 = nn.Linear(d, d_ff), nn.Linear(d_ff, d)
        self.ln_attn_out, self.ln_mlp_out = nn.LayerNorm(d), nn.LayerNorm(d)

    def attn(self, x, cos, sin, mask, cache):
        B, T, Dm = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, Dm // self.h).permute(2, 0, 3, 1, 4)
        if cache is not None:  # the fork's preallocated KV cache, unchanged
            if 'k' not in cache:
                cache['k'] = k.new_empty(B, self.h, cache['max'], Dm // self.h)
                cache['v'] = v.new_empty(B, self.h, cache['max'], Dm // self.h)
                cache['n'] = 0
            n0, n1 = cache['n'], cache['n'] + T
            cache['k'][:, :, n0:n1] = k
            cache['v'][:, :, n0:n1] = v
            cache['n'] = n1
            k, v = cache['k'][:, :, :n1], cache['v'][:, :, :n1]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.to(q.dtype))
        return self.proj(y.transpose(1, 2).reshape(B, T, Dm))

    def forward(self, x, cos, sin, mask=None, cache=None):
        x = x + self.ln_attn_out(self.attn(self.ln1(x), cos, sin, mask, cache))
        return x + self.ln_mlp_out(self.fc2(F.gelu(self.fc1(self.ln2(x)))))


class ALiBiGPT(GPT):
    """The fork's GPT with Peri-LN blocks and ALiBi / NoPE heads instead of RoPE.  Positions come from `pos`
    (left-padded sampling batches) or arange; key positions are kept in caches[0]['pos'] so cached decoding sees the
    same bias as the full forward (sample.py's compaction moves that tensor with the K/V).  Causality / padding come
    from `mask` when given, else from rel < 0."""

    def __init__(self, vocab, n_layer=N_LAYER, d=D, n_head=N_HEAD, max_len=1024, d_ff=D_FF, arch='best'):
        super().__init__(vocab, n_layer, d, n_head, max_len)
        self.blocks = nn.ModuleList(PeriLNBlock(d, n_head, d_ff) for _ in range(n_layer))
        self.blocks.apply(self._init)
        self.cfg.update(d_ff=d_ff, arch='best')
        self.register_buffer('slopes', torch.tensor(alibi_slopes(n_head)).view(1, n_head, 1, 1), persistent=False)

    PREFILL_ELEMS = 2 ** 29    # rows per prefill chunk: keep one (rows, H, T, S) bias / score tensor near 1 GB in bf16

    def forward(self, idx, pos=None, mask=None, caches=None):
        B, T = idx.shape
        if caches is not None and T > 1 and B > 1 and B * self.cfg['n_head'] * T * T > self.PREFILL_ELEMS:
            return self._chunked_prefill(idx, pos, mask, caches)
        return self.head(self.ln_f(self.features(idx, pos, mask, caches)[0]))

    def _chunked_prefill(self, idx, pos, mask, caches):
        """The prompt forward of cached sampling, in row chunks.  An additive float mask sends SDPA to a kernel whose
        memory grows as B x H x T^2 (a batch-2,048 prefill of 500-token states OOMs a 48 GB card).  Rows are independent,
        so each chunk writes its K/V (and key positions) into row slices of the full preallocated caches."""
        B, T = idx.shape
        hd = self.cfg['d'] // self.cfg['n_head']
        rows = max(1, self.PREFILL_ELEMS // (self.cfg['n_head'] * T * T))
        if pos is None:
            pos = torch.arange(T, device=idx.device)[None].expand(B, T)
        dt = self.emb.weight.dtype
        if idx.is_cuda and torch.is_autocast_enabled('cuda'):
            dt = torch.get_autocast_dtype('cuda')
        for c in caches:
            assert 'k' not in c, 'chunked prefill expects fresh caches'
            c['k'] = torch.empty(B, self.cfg['n_head'], c['max'], hd, device=idx.device, dtype=dt)
            c['v'] = torch.empty_like(c['k'])
        caches[0]['pos'] = pos.new_empty(B, caches[0]['max'])
        out = []
        for r0 in range(0, B, rows):
            r1 = min(B, r0 + rows)
            sub = [{'max': c['max'], 'k': c['k'][r0:r1], 'v': c['v'][r0:r1], 'n': 0} for c in caches]
            sub[0].update(pos=caches[0]['pos'][r0:r1], n_pos=0)
            out.append(self.head(self.ln_f(self.features(idx[r0:r1], pos[r0:r1], None if mask is None else mask[r0:r1], sub)[0])))
        for c in caches:
            c['n'] = T
        caches[0]['n_pos'] = T
        return torch.cat(out)

    def features(self, idx, pos=None, mask=None, caches=None):
        """The trunk: (final hidden state before ln_f, the ALiBi bias used)."""
        B, T = idx.shape
        if pos is None:
            pos = torch.arange(T, device=idx.device)[None]
            if caches is not None:    # the cache keeps one row of key positions per batch row (compaction indexes it)
                pos = pos.expand(B, T)
        key_pos = pos
        if caches is not None:
            c = caches[0]
            if 'pos' not in c:
                c['pos'], c['n_pos'] = pos.new_empty(pos.size(0), c['max']), 0
            n0, n1 = c['n_pos'], c['n_pos'] + T
            c['pos'][:, n0:n1] = pos
            c['n_pos'] = n1
            key_pos = c['pos'][:, :n1]
        # Built directly in the attention dtype (bf16 under autocast) with in-place ops: a (B, H, T, S) prefill bias at
        # batch 2,048 x 400^2 is 5 GB this way against ~29 GB for fp32 + int64 + copies.  The values are identical to
        # computing in fp32 and casting (Robbie's `mask.to(q.dtype)`): the slopes are powers of two, so slope x bf16(rel)
        # is the bf16 rounding of the exact product.
        dt = torch.get_autocast_dtype('cuda') if idx.is_cuda and torch.is_autocast_enabled('cuda') else self.slopes.dtype
        rel = (pos[:, None, :, None].int() - key_pos[:, None, None, :].int())
        neg = rel < 0 if mask is None else ~mask
        bias = rel.to(dt)
        del rel
        bias = bias * (-self.slopes.to(dt))
        bias.masked_fill_(neg, float('-inf'))
        x = self.emb(idx)
        for i, b in enumerate(self.blocks):
            x = b(x, None, None, bias, None if caches is None else caches[i])
        return x, bias


class MTPHead(nn.Module):
    """The MTP module: never registered on the model, so it is not in the checkpoint or the parameter count."""

    def __init__(self, d, h, d_ff=D_FF):
        super().__init__()
        self.ln_h, self.ln_e = nn.LayerNorm(d), nn.LayerNorm(d)
        self.proj = nn.Linear(2 * d, d, bias=False)
        self.block = PeriLNBlock(d, h, d_ff)


def orthogonal_init(params):
    """Robbie's init of the block / MTP matrices: orthogonal with element second moment 0.02^2."""
    for p in params:
        if p.ndim == 2:
            torch.nn.init.orthogonal_(p, gain=0.02 * math.sqrt(max(p.shape)))


def mtp_loss(model, mtp, x, m, weight=MTP_WEIGHT):
    """Next-token CE on the masked (action) tokens plus `weight` x the MTP head's token-(t+2) CE at the same kind of
    positions.  Returns (total, main, aux)."""
    h, bias = model.features(x[:, :-1])
    logits = model.head(model.ln_f(h))
    V = logits.size(-1)
    ce = F.cross_entropy(logits.reshape(-1, V).float(), x[:, 1:].reshape(-1), reduction='none')
    main = (ce * m[:, 1:].reshape(-1)).sum() / m[:, 1:].sum()
    if not weight:
        return main, main, main.detach() * 0
    z = mtp.proj(torch.cat((mtp.ln_h(h[:, :-1]), mtp.ln_e(model.emb(x[:, 1:-1]))), -1))
    z = mtp.block(z, None, None, bias[..., :-1, :-1], None)
    ce2 = F.cross_entropy(model.head(model.ln_f(z)).reshape(-1, V).float(), x[:, 2:].reshape(-1), reduction='none')
    aux = (ce2 * m[:, 2:].reshape(-1)).sum() / m[:, 2:].sum().clamp(min=1)
    return main + weight * aux, main, aux
