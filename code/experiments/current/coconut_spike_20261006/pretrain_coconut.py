# imports the fork (model, tokenizer, train, ...), which is not resolvable here
# pyright: reportMissingImports=false
"""Coconut-style spike: autoresearch run 198's pretrain.py (sha 6bdbe0047ac4) plus continuous
thoughts between the theorem statement and the proof. See ThoughtGPT and arm_loss; the optimizer,
data pipeline and speed code are 198's. ND_ARM picks one of six training recipes.

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
               a list of {secs, step, train_loss, val_loss} taken every ~CURVE_EVERY seconds
               (see log_point below), so a run's trajectory survives in git. Add anything else.
               val_loss is on VAL_N records held out from train_recs by this file; keep that.

Anything imported from the fork (ND_FORK) is fair game to copy in here and modify.
"""

import math
import os
import random
import time

import torch
import torch._inductor.config  # noqa: F401  (explicit, so the attribute resolves for pyright)
from lean_tok import MAXN  # fork
from model import GPT, Block  # fork
from tokenizer import make_tokenizer  # fork
from train import loss_on  # fork

# ---- knobs
N_LAYER, D, N_HEAD = 6, 384, 8
# MLP width held at 151's 1280: the fork's Block hardcodes 4 * D (1536 at D 384 would pass the
# 10M cap), so PeriLNBlock swaps fc1/fc2. 1280 != 3 * D keeps muon_step's QKV-by-shape test valid.
D_FF = 1280
BS, LR, MIN_LR_FRAC, WD = 128, 1e-3, 0.3, 0.1
MUON_LR, MUON_MOMENTUM = 0.01, 0.95
# Polar Express (Amsel et al. 2025) per-iteration quintic coefficients, as in modded-nanogpt
# (num_iters=5, safety factor 2e-2): singular values land in ~[0.86, 1.14] instead of the
# fixed quintic's ~[0.68, 1.13], at the same five iterations.
NS_COEFFS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)
# Ten times the AdamW group's per-step multiplicative decay (lr * wd) on the Muon matrices.
MUON_WD = 10 * WD * LR / MUON_LR
TOK_MODE = "lean_seq"  # required by the harness (Lean protocol)
# Hybrid heads: half the heads get ALiBi (Press et al. 2021) with the standard slope set for
# H/2 heads, 2^(-8 i / (H/2)) = 1/4 .. 1/256 with 8 heads; the other half get slope 0, i.e. no
# positional signal at all (NoPE), so they can look citations up by content at any distance.
N_ALIBI = N_HEAD // 2
ALIBI_SLOPES = tuple(2.0 ** (-8.0 * (i + 1) / N_ALIBI) for i in range(N_ALIBI)) + (0.0,) * (
    N_HEAD - N_ALIBI
)
VAL_N, CURVE_EVERY = 1024, 15  # records held out for val_loss; seconds between curve points
# Token-budget batches (fairseq max-tokens): each step takes a run of length-sorted records with
# count * padded_length <= TOK_BUDGET, so a batch of short proofs has many records and a batch of
# long ones few, and the padded token count per step is ~constant. 128 x 144 matches the real
# (unpadded) tokens of the incumbent's 128-record batches (mean record ~135 tokens), whose
# padding to the batch max (~400) made every step ~3x larger than its content.
TOK_BUDGET = 128 * 144
# Multi-token prediction (DeepSeek-V3 2024, Gloeckle et al. 2024), pretraining only: one extra
# Peri-LN block reads [trunk state at t; embedding of token t+1] and predicts token t+2 through
# the model's own ln_f/head. Its loss is added at this weight (DeepSeek-V3's early-training value).
MTP_WEIGHT = 0.3
# ---- continuous thoughts (Coconut, Hao et al. 2024)
# After the prompt's last token the model takes T extra steps whose input is its own previous
# output, then writes the proof. Training draws T from 0..T_MAX per batch (pretraining and the
# harness's RL fine-tune, which runs in train mode); eval and sampling use model.n_thoughts.
#   none        thoughts get gradient from the proof loss only
#   pause       control: every thought's input is one learned vector, no feedback
#               (Goyal et al. 2023)
#   probe       a pretraining-only head must read the proof's t-th inference rule off thought t
#   curriculum  Coconut's schedule: the plan (the proof's rule sequence) starts as explicit steps
#               whose inputs are rule embeddings, and the first s of them turn latent as s grows
#               over pretraining; the last fifth is fully latent
#   codi        self-distillation (Shen et al. 2025): an explicit-plan pass teaches the latent pass
#               to reach the same per-layer state on the step that starts the proof
#   future      thought t must predict the hidden state the model reaches at the proof's t-th `;`
# The handed-off model is always fully latent; the plan vocabulary never enters the tokenizer.
ARM = os.environ.get("ND_ARM", "none")
assert ARM in ("none", "pause", "probe", "curriculum", "codi", "future"), ARM
T_MAX = 8
EVAL_THOUGHTS = 8
VAL_THOUGHTS = (0, 1, 2, 4, 8, 16, 32)  # val loss at each count, once, when pretraining ends
RULES = ("AS", "IMPI", "IMPE", "ANDI", "ANDE1", "ANDE2", "ORI1", "ORI2", "ORE", "NEGI", "NEGE")
RULES += ("BOTE", "DN", "R")
PLAN_END = len(RULES)  # the plan class after a proof's last rule
AUX_WEIGHT = 0.5  # probe and future losses


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
        if p.grad is None:
            continue
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
            polynomial = b * gram + c * gram @ gram
            update = a * update + polynomial @ update
        if transpose:
            update = update.mT
        p.mul_(1 - lr * MUON_WD)
        # lr may be a 0-d tensor (CUDA-graphed step); scale in p's dtype like add_(alpha=-lr)
        p.sub_(lr * update.reshape_as(p).to(p.dtype))


class PeriLNBlock(Block):
    """P22 Peri-LN (Kim et al. 2025): the fork's pre-LN block plus a LayerNorm on each branch
    output, so every residual write has unit scale times a learned gain. Attention has no RoPE:
    `mask` is the additive ALiBi bias (B or 1, H, T, S) that ThoughtGPT builds and cos/sin are
    unused. `cache` is the fork's preallocated dict when sampling, or a list when training: the
    keys and values of the segments already run (prompt, thoughts), which this call extends.
    The dict's in-place writes cannot carry gradients; the list is concatenated instead."""

    def __init__(self, d, h):
        super().__init__(d, h)
        self.fc1, self.fc2 = torch.nn.Linear(d, D_FF), torch.nn.Linear(D_FF, d)
        self.ln_attn_out, self.ln_mlp_out = torch.nn.LayerNorm(d), torch.nn.LayerNorm(d)

    def attn(self, x, cos, sin, mask, cache):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if isinstance(cache, list):
            if cache:
                k, v = torch.cat((cache[0], k), 2), torch.cat((cache[1], v), 2)
            cache[:] = [k, v]
        elif cache is not None:  # the fork's preallocated KV cache, unchanged
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


class ThoughtGPT(GPT):
    """Run 198's model (the fork's GPT with half-ALiBi, half-NoPE heads) plus continuous thoughts.

    A sequence is [prompt][T thoughts][proof]. Thought 1's input is made from the output at the
    prompt's last token, thought t+1's from thought t's output, and the last thought's output
    predicts the first proof token. Every thought sits at the prompt's last position on the ALiBi
    axis, so the distance from statement to proof does not depend on T. With T = 0 this is
    exactly run 198's network.

    Sampling (`caches` given) runs the thoughts inside the prefill call, so the fork's sampler
    needs no change. Training and loss evaluation (no caches) go through prompt_pass + finish."""

    def __init__(self, vocab, n_layer, d, n_head, prompt_end, seed, max_len=1024):
        super().__init__(vocab, n_layer, d, n_head, max_len)
        slopes = torch.tensor(ALIBI_SLOPES).view(1, n_head, 1, 1)
        self.register_buffer("slopes", slopes, persistent=False)
        self.ln_thought = torch.nn.LayerNorm(d)
        self.thought_emb = torch.nn.Parameter(torch.randn(d) * 0.02)
        self.feedback = ARM != "pause"
        self.prompt_end = prompt_end  # the token every prompt ends with, and contains once
        self.n_thoughts, self.t_rng = EVAL_THOUGHTS, random.Random(seed)

    def slot_input(self, h):
        """A latent thought's input, from the previous step's output h (B, 1, D)."""
        x = self.thought_emb.expand(h.size(0), 1, -1)
        return x + self.ln_thought(h) if self.feedback else x

    def forward(self, idx, pos=None, mask=None, caches=None):
        if caches is not None:
            return self.head(self.ln_f(self.cached(idx, pos, mask, caches)))
        T = self.t_rng.randint(0, T_MAX) if self.training else self.n_thoughts
        return self.head(self.ln_f(self.finish(self.prompt_pass(idx), T)["h"]))

    # ---- sampling: the fork's incremental decoding, with the thoughts appended to the prefill

    def cached(self, idx, pos, mask, caches):
        c = caches[0]
        if "n_pos" not in c:  # prefill: the prompt, then the thoughts
            T = c["T"] = self.n_thoughts
            c["L"] = idx.size(1)
            for cache in caches:
                cache["max"] += T  # room for the thoughts' keys and values
            c["pos"], c["n_pos"] = pos.new_empty(pos.size(0), c["max"]), 0
            h = self.cached_step(self.emb(idx), pos, mask, caches)
            keys = mask[:, :, -1:]  # (B, 1, 1, L): which prompt columns are real tokens
            for _ in range(T):
                keys = torch.cat((keys, keys.new_ones(keys.size(0), 1, 1, 1)), -1)
                h = self.cached_step(self.slot_input(h[:, -1:]), pos[:, -1:], keys, caches)
            return h
        # later tokens: the sampler's key mask has no columns for the thoughts
        T, L = c["T"], c["L"]
        mask = torch.cat((mask[..., :L], mask.new_ones(*mask.shape[:-1], T), mask[..., L:]), -1)
        return self.cached_step(self.emb(idx), pos, mask, caches)

    def cached_step(self, x, pos, mask, caches):
        c, n = caches[0], x.size(1)
        c["pos"][:, c["n_pos"] : c["n_pos"] + n] = pos
        c["n_pos"] += n
        rel = pos[:, None, :, None] - c["pos"][:, None, None, : c["n_pos"]]  # (B, 1, T, S)
        bias = (-self.slopes * rel.to(self.slopes.dtype)).masked_fill(~mask, float("-inf"))
        for b, cache in zip(self.blocks, caches, strict=True):
            x = b(x, None, None, bias, cache)
        return x

    # ---- training: one pass over the prompt, T single-position steps, one pass over the proof

    def causal_bias(self, n, device):
        """Run 198's bias for an n-token sequence: (1, H, n, n), causal."""
        rel = torch.arange(n, device=device)[:, None] - torch.arange(n, device=device)[None]
        return (-self.slopes * rel).masked_fill(rel < 0, float("-inf"))

    def prompt_pass(self, idx):
        """Run the prompt of each right-padded [prompt, proof] row. Rows are re-laid-out so every
        prompt ends in column P - 1 (left-padded) and every proof starts in column P; relative
        ALiBi positions are then the same in every row, and only the left padding is per-row."""
        B, S = idx.shape
        p = (idx == self.prompt_end).int().argmax(1)  # the prompt's last token, per row
        P, L = int(p.max()) + 1, S - 1 - int(p.min())
        shift = P - 1 - p
        cols = torch.arange(P + L, device=idx.device)[None] - shift[:, None]  # source index
        aligned, valid = idx.gather(1, cols.clamp(0, S - 1)), cols[:, :P] >= 0
        dtype = torch.bfloat16 if torch.is_autocast_enabled() else torch.float32
        full = self.causal_bias(P + L, idx.device)  # layout positions: column index
        eye = torch.eye(P, dtype=torch.bool, device=idx.device)  # pad queries attend to themselves
        blocked = ~(valid[:, None, None, :] | eye)
        st = {
            "S": S, "P": P, "shift": shift, "aligned": aligned, "dtype": dtype,
            # proof queries x prompt keys, proof queries x proof keys
            "bias_proof": (
                full[..., P:, :P].masked_fill(~valid[:, None, None, :], float("-inf")),
                full[..., P:, P:],
            ),
            # a thought at position P - 1 x prompt keys
            "bias_slot": full[..., P - 1 : P, :P].masked_fill(
                ~valid[:, None, None, :], float("-inf")
            ),
            "pasts": [[] for _ in self.blocks],
        }  # fmt: skip
        x = self.emb(aligned[:, :P])
        bias = full[..., :P, :P].masked_fill(blocked, float("-inf")).to(dtype)
        for b, past in zip(self.blocks, st["pasts"], strict=True):
            x = b(x, None, None, bias, past)
        st["prompt_out"] = x
        return st

    def finish(self, st, T, plan_emb=None, plan=None, n_latent=0):
        """The thoughts and the proof, on top of prompt_pass's state (left intact, so two passes
        can share one prompt). Step t + 1's input is latent (slot_input) when t < n_latent or no
        plan is given, and otherwise the embedding of the plan's t-th rule: an explicit step.

        Returns h (B, S, D), the final hidden states in idx's own layout with the last thought's
        output at the prompt's last position (it predicts the first proof token); slots
        (B, T + 1, D), the output at the prompt's end followed by each step's; and layers, each
        block's output on the last step."""
        P, dtype = st["P"], st["dtype"]
        B, H = st["aligned"].size(0), self.slopes.size(1)
        pasts = [list(past) for past in st["pasts"]]
        h = st["prompt_out"][:, -1:]
        slots, layers = [h], []
        for t in range(T):
            explicit = plan is not None and t >= n_latent
            x = plan_emb(plan[:, t])[:, None] if explicit else self.slot_input(h)
            # earlier thoughts and this one are at the same position: zero bias
            zeros = st["bias_slot"].new_zeros(B, H, 1, t + 1)
            bias = torch.cat((st["bias_slot"].expand(B, H, 1, P), zeros), -1).to(dtype)
            layers = []
            for b, past in zip(self.blocks, pasts, strict=True):
                x = b(x, None, None, bias, past)
                layers.append(x)
            h = x
            slots.append(h)
        on_prompt, on_proof = st["bias_proof"]
        n = on_proof.size(-1)
        # every thought is one position before the proof's first token, j + 1 before its j-th
        on_slots = on_proof[..., :1].expand(1, H, n, T) - self.slopes
        x = self.emb(st["aligned"][:, P:])
        bias = torch.cat(
            (
                on_prompt.expand(B, H, n, P),
                on_slots.expand(B, H, n, T),
                on_proof.expand(B, H, n, n),
            ),
            -1,
        ).to(dtype)
        for b, past in zip(self.blocks, pasts, strict=True):
            x = b(x, None, None, bias, past)
        out = torch.cat((st["prompt_out"][:, :-1], h, x), 1)  # aligned layout, thoughts folded
        back = torch.arange(st["S"], device=x.device)[None] + st["shift"][:, None]
        h = out.gather(1, back[..., None].expand(-1, -1, out.size(-1)))
        return {"h": h, "slots": torch.cat(slots, 1), "layers": layers}


class MTPHead(torch.nn.Module):
    """The MTP module. A separate nn.Module, never registered on the model, so the harness's
    parameter count, RL AdamW and sampler never see it; it is dropped when pretrain() returns."""

    def __init__(self, d, h):
        super().__init__()
        self.ln_h, self.ln_e = torch.nn.LayerNorm(d), torch.nn.LayerNorm(d)
        self.proj = torch.nn.Linear(2 * d, d, bias=False)
        self.block = PeriLNBlock(d, h)


class Aux(torch.nn.Module):
    """Everything that exists only during pretraining: the MTP module and the arms' plan and
    future-state heads. Never registered on the model, so it is dropped when pretrain() returns."""

    def __init__(self, d, h):
        super().__init__()
        self.mtp = MTPHead(d, h)
        self.plan_emb = torch.nn.Embedding(PLAN_END + 1, d)  # explicit plan steps' inputs
        self.ln_plan, self.plan_head = torch.nn.LayerNorm(d), torch.nn.Linear(d, PLAN_END + 1)
        self.ln_future, self.future_head = torch.nn.LayerNorm(d), torch.nn.Linear(d, d)

    def plan_loss(self, slots, plan):
        """Cross-entropy of reading plan[:, t] off slots[:, t]."""
        logits = self.plan_head(self.ln_plan(slots)).float()
        return torch.nn.functional.cross_entropy(logits.flatten(0, 1), plan.flatten())


def lm_losses(model, mtp, h, x, m):
    """The fork's loss_on (next-token CE on proof tokens) plus MTP_WEIGHT times the CE of the
    MTP module's token-(t+2) prediction at the same proof positions, from the trunk states h of
    x[:, :-1]. Returns (total, main, mtp)."""
    logits = model.head(model.ln_f(h))
    V = logits.size(-1)
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, V).float(), x[:, 1:].reshape(-1), reduction="none"
    )
    main = (ce * m[:, 1:].reshape(-1)).sum() / m[:, 1:].sum()
    # Position t (0..T-3) gets h_t and emb(x_{t+1}); causal attention over these keeps every
    # prediction of x_{t+2} a function of x_{<=t+1}.
    z = mtp.proj(torch.cat((mtp.ln_h(h[:, :-1]), mtp.ln_e(model.emb(x[:, 1:-1]))), -1))
    z = mtp.block(z, None, None, model.causal_bias(z.size(1), z.device), None)
    ce2 = torch.nn.functional.cross_entropy(
        model.head(model.ln_f(z)).reshape(-1, V).float(), x[:, 2:].reshape(-1), reduction="none"
    )
    aux = (ce2 * m[:, 2:].reshape(-1)).sum() / m[:, 2:].sum().clamp(min=1)
    return main + MTP_WEIGHT * aux, main, aux


def arm_loss(model, aux, x, m, plan, ends, frac):
    """One training step's loss for ARM, with a thought count drawn for the batch. plan (B, T_MAX)
    is each proof's rule sequence, ends (B, T_MAX) the index in x of its t-th `;`, frac the
    fraction of the pretraining budget used. Returns (total, main, mtp, arm): main is the latent
    pass's next-token CE, arm the arm's own extra loss."""
    T = model.t_rng.randint(0, T_MAX)
    st = model.prompt_pass(x[:, :-1])
    # curriculum: stage s of 0..T_MAX over the first 80% of the budget; steps before s are latent
    stage = int(frac * (T_MAX + 1) / 0.8) if ARM == "curriculum" else T_MAX
    out = model.finish(st, T, aux.plan_emb, plan if stage < T else None, stage)
    total, main, mtp = lm_losses(model, aux.mtp, out["h"], x, m)
    arm = main.new_zeros(())
    if T == 0:
        return total, main, mtp, arm
    if ARM == "probe":
        arm = AUX_WEIGHT * aux.plan_loss(out["slots"][:, :T], plan[:, :T])
    elif ARM == "curriculum" and stage < T:  # the explicit steps are supervised, as in Coconut
        arm = aux.plan_loss(out["slots"][:, stage:T], plan[:, stage:T])
    elif ARM == "future":
        idx = ends[:, :T, None].expand(-1, -1, out["h"].size(-1))
        target = out["h"].detach().gather(1, idx).float()
        pred = aux.future_head(aux.ln_future(out["slots"][:, 1:])).float()
        arm = AUX_WEIGHT * (1 - torch.nn.functional.cosine_similarity(pred, target, -1).mean())
    elif ARM == "codi":
        # teacher: the same prompt states, every step explicit; it learns the plan and the proof
        teacher = model.finish(st, T, aux.plan_emb, plan, 0)
        _, t_main, _ = lm_losses(model, aux.mtp, teacher["h"], x, m)
        gap = sum(
            torch.nn.functional.smooth_l1_loss(a.float(), b.detach().float())
            / b.detach().float().std()
            for a, b in zip(out["layers"], teacher["layers"], strict=True)
        ) / len(out["layers"])
        arm = t_main + aux.plan_loss(teacher["slots"][:, :T], plan[:, :T]) + gap
    return total + arm, main, mtp, arm


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


def plans(recs, data, tok, device):
    """Per record: its proof's inference rules in line order (premise lines skipped), padded with
    PLAN_END to T_MAX; and the index in the record's token row of the proof's t-th `;`, padded
    with the index of the last token before <eos>."""
    semi = tok.stoi[";"]
    plan = torch.full((len(recs), T_MAX), PLAN_END, dtype=torch.long)
    ends = torch.zeros((len(recs), T_MAX), dtype=torch.long)
    for i, (r, (p, q)) in enumerate(zip(recs, data, strict=True)):
        rules = [
            line.split(" : ")[1].split()[0] for line in r["proof"].split(" ; ") if " : " in line
        ]
        rules = [RULES.index(rule) for rule in rules if rule != "PR"][:T_MAX]
        plan[i, : len(rules)] = torch.tensor(rules)
        at = [len(p) + j for j, t in enumerate(q) if t == semi][:T_MAX]
        ends[i] = len(p) + len(q) - 2
        ends[i, : len(at)] = torch.tensor(at, dtype=torch.long)
    return plan.to(device), ends.to(device)


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


def token_batches(lens, n_train, gen, rng):
    """One epoch of token-budget batches: a random permutation of the training records, stably
    sorted by length (equal lengths stay shuffled), cut greedily into runs of consecutive records
    with count * max_len <= TOK_BUDGET, and the runs' order shuffled. Returns (order, cuts)."""
    perm = torch.randperm(n_train, device=lens.device, generator=gen) + VAL_N
    order = perm[torch.argsort(lens[perm], stable=True)]
    sorted_lens = lens[order].cpu()
    cuts, s = [], 0
    while s < len(sorted_lens):
        window = sorted_lens[s : s + TOK_BUDGET // int(sorted_lens[s])]
        # count * max_len is non-decreasing along the window, so the fits form a prefix.
        e = s + int((torch.arange(1, len(window) + 1) * window <= TOK_BUDGET).sum())
        cuts.append((s, e, int(sorted_lens[e - 1])))  # sorted: last is the batch's max length
        s = e
    rng.shuffle(cuts)
    return order, cuts


def gpu_batch(ids, mask, ref_max, lens, idxs, tok, gen, max_len=None):
    """The fork's batch() + tok.shift_abs, vectorised: a random per-row offset is added to
    every ref token."""
    x, m = ids[idxs], mask[idxs]
    # Known on the host for training batches: passing it avoids a GPU sync every step.
    T = int(lens[idxs].max()) if max_len is None else max_len
    x, m = x[:, :T].clone(), m[:, :T]
    if tok.mode in ("abs", "lean_seq"):  # lean_seq: names n1.. get the same start offset
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
    tok = make_tokenizer(TOK_MODE)
    data = [(tok.encode_prompt(r["prompt"]), tok.encode_proof(r["proof"])) for r in train_recs]
    ids, mask, ref_max, lens = to_gpu(data, tok, device)
    plan, ends = plans(train_recs, data, tok, device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    prompt_end = data[0][0][-1]  # `by`: every prompt ends with it and has it nowhere else
    assert all(p[-1] == prompt_end and prompt_end not in p[:-1] for p, _ in data[:2000])
    model = ThoughtGPT(tok.vocab_size, N_LAYER, D, N_HEAD, prompt_end, seed)
    model.blocks = torch.nn.ModuleList(PeriLNBlock(D, N_HEAD) for _ in range(N_LAYER))
    model.blocks.apply(model._init)  # fork init: zero biases
    model = model.to(device)
    aux = Aux(D, N_HEAD)
    aux.apply(model._init)
    aux = aux.to(device)
    mtp = aux.mtp
    # Compile the repeated blocks once each (torch >= 2.9; dynamic shapes, ~14 s): ~14% faster
    # steps. The coalescing tiling analysis crashes on dynamic shapes, so it is off. Undone at
    # handoff so the harness's RL fine-tune and KV-cache sampling run eager.
    torch._inductor.config.triton.coalesce_tiling_analysis = False
    muon_lr, muon_graph = torch.zeros((), device=device), None
    compiled = list(model.blocks) + [mtp.block]
    # ND_COMPILE=0: eager blocks, if compiling the list-cache path misbehaves
    if os.environ.get("ND_COMPILE", "1") == "0":
        compiled = []
    for b in compiled if device == "cuda" else []:  # CPU smoke runs stay eager
        b.compile(dynamic=True)
    for p in list(model.blocks.parameters()) + list(mtp.parameters()):
        if p.ndim == 2:
            # Orthogonal matrices have element second moment gain^2 / max(rows, cols).
            torch.nn.init.orthogonal_(p, gain=0.02 * math.sqrt(max(p.shape)))
    log(
        f"pretrain: {model.n_params():,} params, {len(data):,} records on {device} "
        f"after {time.time() - t0:.0f}s, budget {budget_secs}s"
    )
    # The output readout joins the hidden matrices under Muon; embeddings stay on AdamW.
    trainable = list(model.parameters()) + list(aux.parameters())
    matrices = [p for p in model.blocks.parameters() if p.ndim == 2] + [model.head.weight]
    matrices += [p for p in mtp.parameters() if p.ndim == 2]
    matrix_ids = {id(p) for p in matrices}
    auxiliary = [p for p in trainable if id(p) not in matrix_ids]
    momentum_buffers = [torch.zeros_like(p) for p in matrices]
    opt = torch.optim.AdamW(
        auxiliary, lr=LR, weight_decay=WD, betas=(0.9, 0.95), fused=device == "cuda"
    )
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda"))
    model.train()
    step, last_loss, records_seen = 0, float("nan"), 0
    curve, recent, recent_aux, recent_arm, next_point = [], [], [], [], CURVE_EVERY
    n_train = len(data) - VAL_N  # records [0, VAL_N) are the val slice
    rng, cuts, batches_per_epoch = random.Random(seed), [], None
    order = torch.empty(0, dtype=torch.long, device=device)
    while True:
        frac = (
            time.time() - t0
        ) / budget_secs  # cosine over wall-clock, so the cap is the schedule
        if frac >= 1.0:
            break
        step += 1
        # No warmup (as in modded-nanogpt): Muon's step is normalized and Adam's is ~lr per
        # element from step 1, and Peri-LN normalizes every branch write.
        lr = LR * MIN_LR_FRAC + 0.5 * LR * (1 - MIN_LR_FRAC) * (1 + math.cos(math.pi * frac))
        for g in opt.param_groups:
            g["lr"] = lr
        if not cuts:  # new epoch
            order, cuts = token_batches(lens, n_train, gen, rng)
            batches_per_epoch = len(cuts)
        s, e, T = cuts.pop()
        idxs = order[s:e]
        records_seen += e - s
        x, m = gpu_batch(ids, mask, ref_max, lens, idxs, tok, gen, T)
        with autocast:
            loss, main, mtp_ce, arm = arm_loss(model, aux, x, m, plan[idxs], ends[idxs], frac)
        model.zero_grad(set_to_none=False)  # grads keep fixed addresses for the Muon CUDA graph
        aux.zero_grad(set_to_none=False)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        muon_lr.fill_(MUON_LR * lr / LR)
        # Capture once; later steps replay one graph instead of ~400 kernel launches. CPU smoke
        # runs call the step directly.
        if device != "cuda":
            muon_step(matrices, momentum_buffers, muon_lr)
        elif muon_graph is None:
            state = matrices + momentum_buffers
            saved = [t.clone() for t in state]
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                muon_step(matrices, momentum_buffers, muon_lr)  # warm-up (lazy cuBLAS init)
            torch.cuda.current_stream().wait_stream(side)
            muon_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(muon_graph):
                muon_step(matrices, momentum_buffers, muon_lr)
            with torch.no_grad():
                for t, v in zip(state, saved, strict=True):
                    t.copy_(v)  # undo the warm-up and capture updates
            muon_graph.replay()
        else:
            muon_graph.replay()
        if step % 20 == 0:
            last_loss = main.item()  # the next-token CE, comparable with earlier runs
            recent.append(last_loss)
            recent_aux.append(mtp_ce.item())
            recent_arm.append(arm.item())
        if (
            time.time() - t0 >= next_point
        ):  # curve point: mean recent train loss + val loss on the held-out slice
            curve.append(
                {
                    "secs": round(time.time() - t0),
                    "step": step,
                    "train_loss": sum(recent) / len(recent) if recent else last_loss,
                    "val_loss": val_loss(model, ids, mask, ref_max, lens, tok, gen, autocast),
                    "mtp_loss": sum(recent_aux) / len(recent_aux) if recent_aux else None,
                    "arm_loss": sum(recent_arm) / len(recent_arm) if recent_arm else None,
                }
            )
            log(
                f"pretrain step {step} train {curve[-1]['train_loss']:.4f} "
                f"val {curve[-1]['val_loss']:.4f} arm {curve[-1]['arm_loss'] or 0:.4f} "
                f"lr {lr:.2e} {curve[-1]['secs']}s"
            )
            recent, recent_aux, recent_arm, next_point = [], [], [], next_point + CURVE_EVERY
    val_by_thoughts = {}
    for n in VAL_THOUGHTS if step else ():  # no steps: a checkpoint rebuild, not a training run
        model.n_thoughts = n
        val_by_thoughts[n] = val_loss(model, ids, mask, ref_max, lens, tok, gen, autocast)
    model.n_thoughts = EVAL_THOUGHTS
    log(f"pretrain val loss by thoughts: {val_by_thoughts}")
    for b in compiled:
        b._compiled_call_impl = None
    model.eval()
    return (
        model,
        tok,
        {
            "steps": step,
            "records_seen": records_seen,
            "epochs": records_seen / n_train,
            "batches_per_epoch": batches_per_epoch,
            "tok_budget": TOK_BUDGET,
            "last_loss": last_loss,
            "arm": ARM,
            "t_max": T_MAX,
            "eval_thoughts": EVAL_THOUGHTS,
            "val_by_thoughts": val_by_thoughts,
            "curve": curve,
        },
    )
