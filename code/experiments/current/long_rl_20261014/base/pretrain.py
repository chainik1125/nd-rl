# imports the fork (model, tokenizer, train, ...), which is not resolvable here
# pyright: reportMissingImports=false
"""Retry experiment: run 590's file (autoresearch_loops, sha b27dd2cb03ba: run 198's untied model
with the loops harness's RL recipe) plus two retry arms. ND_ARM picks:

  iid    590 unchanged: every sample is an independent first attempt
  retry  RL samples episodes (episodes.py): after a failed attempt the model sees an error code
         and the failing line's name, then writes a whole new proof; the fine-tune adds records
         whose context is the failed attempts and whose target is an accepted proof
  blank  as retry, with every error replaced by one <fail> token
  resume iid's RL unchanged; only the eval differs: a new attempt keeps the failed one's error-free
         prefix and samples the rest (no history, no error tokens, nothing learned)
  resume_rl  resume, and RL's 24 extra samples for a target whose first 8 all failed are 3
         resumed episodes of up to 8 attempts
  repair as retry, trained to REPAIR: the synthetic records put a known proof with one token
         made wrong, and its real error, in front of that same proof, in pretraining too

Pretraining is the same in all three; the vocabulary carries the error tokens in all three.
The original docstring follows.

AGENT-EDITABLE: pretraining and RL. This is the only file the autoresearch agent changes.

Contract with harness.py (frozen):

  pretrain(train_recs, budget_secs, device, seed, log) -> (model, tok, stats)

  train_recs   list of dicts {prompt, proof, n_lines<=6, key, ...}; the frozen cap-6 Stage-1 data
  budget_secs  wall-clock you are charged for, from call to return, INCLUDING model construction
               and torch.compile. Overrun by more than the harness grace -> the run is a timeout.
  model        nn.Module; <= PARAM_CAP unique params (checked again after rl). The harness samples
               from it with the fork's sample.generate, so it must provide:
                 - forward(idx) -> logits (B,T,V)  and  forward(idx, pos=, mask=, caches=) with a
                   per-block KV cache dict, exactly like the fork's GPT (model.blocks must exist)
                 - an integer `n_loops` (the harness's KNOB): the loop count applied in eval
                   mode. The harness sets it to EVAL_LEVEL for the scored eval and to DIAG_LEVEL
                   for the diagnostic one.
  tok          must provide encode_prompt, encode_proof, decode, shift_abs, pad, eos, mode, and
               mode must be "lean_seq" (harness TOK_MODE): Lean text in and out, Lean-gated samples.
               Decoder rule: decode() is deterministic and single-pass; it may not call the
               verifier or try alternatives.
  stats        dict recorded in summary.json and copied into results.json. MUST contain `curve`:
               a list of {secs, step, train_loss, val_loss} taken every ~CURVE_EVERY seconds
               (see log_point below), so a run's trajectory survives in git. Add anything else.
               val_loss is on VAL_N records held out from train_recs by this file; keep that.

  rl(model, tok, targets, train_recs, oracle, budget_secs, device, seed, log) -> stats | None

  targets      list of dicts {name, prompt, n_lines}: the 1,500 RL target statements. No proofs.
  oracle       oracle.sample(model, targets_subset, k, temperature=0.8, batch=1024) samples k
               attempts per target from the model in eval mode at its current n_loops, and returns
               per target {name, prompt, n_lines, proofs, n_ok, n_tried}; `proofs` are the distinct
               attempts Lean AND nd_verify accepted. This is the only source of new training data.
               oracle.used() / oracle.left(): seconds of the RL budget charged / remaining.
  budget_secs  RL_SECS: wall-clock inside rl() minus the oracle's Lean-gate time. Returning more
               than the grace late, or calling the oracle with nothing left, is a timeout.
  The model is trained in place. Everything else about RL is yours: rounds, samples per target,
  loop counts when sampling, which proofs to keep, weighting, replay, optimizer, loss.

Anything imported from the fork (ND_FORK) is fair game to copy in here and modify.
"""

import math
import os
import random
import time

import torch
import torch._inductor.config  # noqa: F401  (explicit, so the attribute resolves for pyright)
from episodes import MAX_CTX, context_ids
from errors import OTHER, classify, error_ids, extend
from lean_tok import MAXN  # fork
from model import GPT, Block  # fork
from normalize import norm  # fork
from tokenizer import make_tokenizer  # fork
from train import loss_on  # fork

# ---- knobs
# Unique blocks: one untied entry block, N_CORE shared core blocks, one untied exit block.
N_CORE = 4
# Control: run 198's trunk. The six blocks are independent and applied once, with no re-injection;
# n_loops is accepted and ignored, so the harness's two evals and rl()'s retry depths all sample
# the same model.
UNTIED = True
N_LAYER, D, N_HEAD = 2 + N_CORE, 384, 8
# Core applications (loops). Training draws one count per batch from TRAIN_LOOPS, in pretraining
# and in rl()'s fine-tune; eval and sampling apply model.n_loops, which the harness sets for its
# evals and rl() sets before each oracle call. DEFAULT_LOOPS is used for the validation curve and
# the harness's post-pretraining greedy check.
TRAIN_LOOPS = (1, 2, 3, 4, 5, 6, 7, 8)
DEFAULT_LOOPS = 2
VAL_LOOPS = (1, 2, 4, 8)  # val loss at each count, once, when pretraining ends
# ---- RL knobs: the old harness's expert iteration (600 AdamW steps per round on up to 4 proofs
# per solved target, weighted x4, mixed with 20,000 replayed pretraining records), except for
# where the samples go. Each round every target gets RL_K attempts at RL_FIRST_LOOPS; a target
# all of them fail on gets RL_RETRY_K more (32 in all, as before) at ONE deeper loop count from
# RL_RETRY_LOOPS, assigned at random, so proofs that only deeper sampling finds enter the
# training set and no deep sample is spent on a target one loop already solves. The random
# assignment makes the retry calls a controlled comparison of loop counts on resistant targets.
# Rounds repeat until another one would not fit in the budget.
RL_FIRST_LOOPS, RL_K = 1, 8
RL_RETRY_LOOPS, RL_RETRY_K = (2, 4, 8), 24
# The fine-tune sees FT_RECORDS records per round (the old 600 steps x 128 records) in the same
# token-budget batches as pretraining (TOK_BUDGET below) instead of 128-record batches padded to
# their longest proof, which with 7-14 line RL proofs in nearly every batch was ~3x the content.
FT_RECORDS, FT_LR, FT_WARMUP = 600 * 128, 3e-4, 50
RETAIN, MAX_PER_THM, RL_WEIGHT = 20000, 4, 4
# ---- retry arms. Each round every target gets RETRY_EP episodes of up to RETRY_ATTEMPTS
# attempts: 32 proofs at most, the iid arm's 8 + 24. A solved target contributes up to EP_PER_THM
# episode records: first the round's real retries (an episode accepted on attempt 2+), then
# synthetic ones that put 1..RETRY_ATTEMPTS-1 of the round's failed attempts in front of a known
# proof. The synthetic records are what teach the format before any real retry has worked.
# Episode records are long, so the retry fine-tune is cut by padded tokens instead of records:
# FT_TOKENS is the iid arm's 600 steps of TOK_BUDGET.
ARM = os.environ.get("ND_ARM", "iid")
RETRY_EP, EP_PER_THM = 4, 4
# ---- repair arm (follow-up 1). In retry seed 0 the attempt after a failure succeeded exactly as
# often as an independent sample: the synthetic records pair failures with unrelated proofs, so
# ignoring the context is the best the model can learn from them. Here the synthetic record is a
# known proof with one token made wrong, its real error, and the same proof as the target, names
# unshifted, so the context says what to keep and what to change. REPAIR_N of them join
# pretraining (built inside the pretraining budget) and CORRUPT_PER_THM per solved target join each
# RL round, in place of the unrelated pairings. RL episodes stop at 3 attempts: attempts 4-8 were
# accepted ~1% of the time and cost half the sampling.
RETRY_ATTEMPTS = 3 if ARM == "repair" else 8
REPAIR_N, CORRUPT_PER_THM = 40000, 2
FT_TOKENS = 600 * 128 * 144
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
    `mask` is the additive ALiBi bias (B or 1, H, T, S) that LoopedGPT builds once per forward
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


class LoopedGPT(GPT):
    """Looped trunk (Universal Transformer, Dehghani et al. 2018; sandwich layout and input
    injection as in Geiping et al. 2025): blocks[0] is the entry block, blocks[1:-1] the shared
    core applied n times, blocks[-1] the exit block. Every core pass starts from
    ln_loop(state) + e, with e the entry block's output, so each pass sees an input of the same
    scale and the prompt is re-injected instead of being buried under accumulated writes.

    P34: the fork's GPT with ALiBi (Press et al. 2021) instead of RoPE. Head h's attention
    logits get -slope_h * (query position - key position), slopes 2^-1 .. 2^-8 for 8 heads, so
    no rotation of q/k anywhere. Positions come from `pos` (sample.generate's left-padded
    batches) or arange; key positions are kept in caches[0] so cached decoding sees the same
    bias as the full forward. Causality/padding come from `mask` when given, else from rel < 0."""

    def __init__(self, vocab, n_layer, d, n_head, seed, max_len=1024):
        super().__init__(vocab, n_layer, d, n_head, max_len)
        slopes = torch.tensor(ALIBI_SLOPES).view(1, n_head, 1, 1)
        self.register_buffer("slopes", slopes, persistent=False)
        self.ln_loop = torch.nn.LayerNorm(d)
        self.n_loops, self.loop_rng = DEFAULT_LOOPS, random.Random(seed)

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
        if UNTIED:
            x = self.emb(idx)
            for j, b in enumerate(self.blocks):
                x = b(x, None, None, bias, None if caches is None else caches[j])
            return x, bias
        n = self.loop_rng.choice(TRAIN_LOOPS) if self.training else self.n_loops
        entry, *core, exit_block = self.blocks
        x = e = entry(self.emb(idx), None, None, bias, None if caches is None else caches[0])
        for loop in range(n):
            x = self.ln_loop(x) + e
            for j, b in enumerate(core, 1):
                # sample.generate makes one cache dict per unique block; a shared block's keys
                # and values differ per pass, so each pass gets its own sub-cache inside it.
                c = caches and caches[j].setdefault(loop, {"max": caches[j]["max"]})
                x = b(x, None, None, bias, c)
        x = exit_block(x, None, None, bias, None if caches is None else caches[-1])
        return x, bias


class MTPHead(torch.nn.Module):
    """The MTP module. A separate nn.Module, never registered on the model, so the harness's
    parameter count, RL AdamW and sampler never see it; it is dropped when pretrain() returns."""

    def __init__(self, d, h):
        super().__init__()
        self.ln_h, self.ln_e = torch.nn.LayerNorm(d), torch.nn.LayerNorm(d)
        self.proj = torch.nn.Linear(2 * d, d, bias=False)
        self.block = PeriLNBlock(d, h)


def mtp_loss(model, mtp, x, m):
    """The fork's loss_on (next-token CE on proof tokens) plus MTP_WEIGHT times the CE of the
    MTP module's token-(t+2) prediction at the same proof positions. Returns (total, main)."""
    h, bias = model.features(x[:, :-1])
    logits = model.head(model.ln_f(h))
    V = logits.size(-1)
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, V).float(), x[:, 1:].reshape(-1), reduction="none"
    )
    main = (ce * m[:, 1:].reshape(-1)).sum() / m[:, 1:].sum()
    # Position t (0..T-3) gets h_t and emb(x_{t+1}); causal attention over these keeps every
    # prediction of x_{t+2} a function of x_{<=t+1}.
    z = mtp.proj(torch.cat((mtp.ln_h(h[:, :-1]), mtp.ln_e(model.emb(x[:, 1:-1]))), -1))
    z = mtp.block(z, None, None, bias[..., :-1, :-1], None)
    ce2 = torch.nn.functional.cross_entropy(
        model.head(model.ln_f(z)).reshape(-1, V).float(), x[:, 2:].reshape(-1), reduction="none"
    )
    aux = (ce2 * m[:, 2:].reshape(-1)).sum() / m[:, 2:].sum().clamp(min=1)
    return main + MTP_WEIGHT * aux, main, aux


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
    N, T = len(data), max(len(d[0]) + len(d[1]) for d in data)
    ids = torch.full((N, T), tok.pad, dtype=torch.long)
    mask = torch.zeros((N, T), dtype=torch.bool)
    ref_max = torch.zeros(
        N, dtype=torch.long
    )  # largest N<k> index used in the proof, for the shift augmentation
    for i, (p, q, *fixed) in enumerate(data):
        ids[i, : len(p) + len(q)] = torch.tensor(p + q)
        mask[i, len(p) : len(p) + len(q)] = True
        # a third element marks a record whose target names must stay as written (repair arm):
        # MAXN leaves gpu_batch no room to shift them
        ref_max[i] = (
            MAXN if fixed else max((x - tok.ref0 + 1 for x in q if x >= tok.ref0), default=0)
        )
    return ids.to(device), mask.to(device), ref_max.to(device), (ids != tok.pad).sum(1).to(device)


def token_batches(lens, n_train, gen, rng, first=VAL_N):
    """One epoch of token-budget batches: a random permutation of the n_train records from
    `first` on, stably sorted by length (equal lengths stay shuffled), cut greedily into runs of
    consecutive records with count * max_len <= TOK_BUDGET, and the runs' order shuffled.
    Returns (order, cuts)."""
    perm = torch.randperm(n_train, device=lens.device, generator=gen) + first
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


def is_name(tok, x):
    return tok.ref0 <= x < tok.ref0 + MAXN


def repair_record(tok, prompt, proof, rng):
    """(context, proof ids, True): the proof with one token made wrong and its error in front of
    the proof itself, both at one random name offset. None if no single swap breaks the proof."""
    ids = tok.encode_proof(proof)
    top = max((x - tok.ref0 + 1 for x in ids if is_name(tok, x)), default=0)
    shift = rng.randint(0, MAXN - top)
    ids = [x + shift if is_name(tok, x) else x for x in ids]
    names = sorted({x for x in ids if is_name(tok, x)})
    kinds = [names] + [
        [tok.stoi[t] for t in group]
        for group in (
            ("P", "Q", "R", "S", "False"),
            ("∧", "∨", "→"),
            (".1", ".2"),
            ("Or.inl", "Or.inr"),
        )
    ]
    spots = [(i, k) for i, x in enumerate(ids) for k in kinds if x in k and len(k) > 1]
    # mostly wrong citations: 'unbound' and the citation type errors were ~70% of retry seed 0's
    # failures, and a wrong formula symbol mostly gives 'premise-type', which the model rarely makes
    cited = [(i, k) for i, k in spots if k is names]
    if cited and rng.random() < 0.8:
        spots = cited
    for i, kind in rng.sample(spots, min(4, len(spots))):
        bad = list(ids)
        bad[i] = rng.choice([x for x in kind if x != ids[i]])
        if classify(tok, prompt, bad)[0] != OTHER:  # OTHER: it parses and type-checks
            prompt_ids = tok.encode_prompt(prompt)
            return context_ids(prompt_ids, [(bad, error_ids(tok, prompt, bad, "error"))]), ids, True
    return None


def align(tok, proof_ids, attempt):
    """proof ids (names from n1) moved to the name offset the failed attempt used, if it fits."""
    first = next((x for x in attempt if is_name(tok, x)), tok.ref0)
    top = max((x for x in proof_ids if is_name(tok, x)), default=tok.ref0)
    shift = first - tok.ref0
    if top + shift >= tok.ref0 + MAXN:
        return proof_ids
    return [x + shift if is_name(tok, x) else x for x in proof_ids]


def pretrain(train_recs, budget_secs, device, seed, log):
    t0 = time.time()
    tok = extend(make_tokenizer(TOK_MODE))
    data = [(tok.encode_prompt(r["prompt"]), tok.encode_proof(r["proof"])) for r in train_recs]
    if ARM == "repair":  # appended, so the validation slice (the first VAL_N records) is unchanged
        rng = random.Random(seed)
        picks = rng.sample(train_recs, REPAIR_N)
        repairs = [repair_record(tok, r["prompt"], r["proof"], rng) for r in picks]
        data += [x for x in repairs if x]
        log(f"pretrain: {len(data) - len(train_recs):,} repair records, {time.time() - t0:.0f}s")
    ids, mask, ref_max, lens = to_gpu(data, tok, device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    model = LoopedGPT(tok.vocab_size, N_LAYER, D, N_HEAD, seed)
    model.blocks = torch.nn.ModuleList(PeriLNBlock(D, N_HEAD) for _ in range(N_LAYER))
    model.blocks.apply(model._init)  # fork init: zero biases
    model = model.to(device)
    mtp = MTPHead(D, N_HEAD)
    mtp.apply(model._init)
    mtp = mtp.to(device)
    # Compile the repeated blocks once each (torch >= 2.9; dynamic shapes, ~14 s): ~14% faster
    # steps. The coalescing tiling analysis crashes on dynamic shapes, so it is off. Undone at
    # handoff so the harness's RL fine-tune and KV-cache sampling run eager.
    torch._inductor.config.triton.coalesce_tiling_analysis = False
    muon_lr, muon_graph = torch.zeros((), device=device), None
    compiled = list(model.blocks) + [mtp.block]
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
    trainable = list(model.parameters()) + list(mtp.parameters())
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
    curve, recent, recent_aux, next_point = [], [], [], CURVE_EVERY
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
            loss, main, aux = mtp_loss(model, mtp, x, m)
        model.zero_grad(set_to_none=False)  # grads keep fixed addresses for the Muon CUDA graph
        mtp.zero_grad(set_to_none=False)
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
            recent_aux.append(aux.item())
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
                }
            )
            log(
                f"pretrain step {step} train {curve[-1]['train_loss']:.4f} "
                f"val {curve[-1]['val_loss']:.4f} lr {lr:.2e} {curve[-1]['secs']}s"
            )
            recent, recent_aux, next_point = [], [], next_point + CURVE_EVERY
    val_by_loops = {}
    for n in VAL_LOOPS if step else ():  # no steps: a checkpoint rebuild, not a training run
        model.n_loops = n
        val_by_loops[n] = val_loss(model, ids, mask, ref_max, lens, tok, gen, autocast)
    model.n_loops = DEFAULT_LOOPS
    log(f"pretrain val loss by loops: {val_by_loops}")
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
            "train_loops": TRAIN_LOOPS,
            "default_loops": DEFAULT_LOOPS,
            "val_by_loops": val_by_loops,
            "curve": curve,
        },
    )


def finetune(model, tok, data, rng, device, by_tokens=False):
    """The old harness's RL fine-tune: AdamW, linear warmup then cosine to lr/10, over FT_RECORDS
    records in token-budget batches (to_gpu / token_batches / gpu_batch, as pretraining). The
    model is in train mode, so every batch draws its own loop count from TRAIN_LOOPS. Returns
    (last loss, steps)."""
    ids, mask, ref_max, lens = to_gpu(data, tok, device)
    gen = torch.Generator(device=device)
    gen.manual_seed(rng.getrandbits(31))
    plan, seen = [], 0  # (order, start, end, padded length) per step, epoch after epoch
    limit = FT_TOKENS if by_tokens else FT_RECORDS
    while seen < limit:
        order, cuts = token_batches(lens, len(data), gen, rng, first=0)
        for s, e, T in cuts:
            if seen >= limit:
                break
            plan.append((order, s, e, T))
            seen += (e - s) * T if by_tokens else e - s
    opt = torch.optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=0.1, betas=(0.9, 0.95))
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda"))
    model.train()
    for step, (order, s, e, T) in enumerate(plan, 1):
        lr = (
            FT_LR * step / FT_WARMUP
            if step < FT_WARMUP
            else FT_LR / 10 + 0.45 * FT_LR * (1 + math.cos(math.pi * step / len(plan)))
        )
        for g in opt.param_groups:
            g["lr"] = lr
        x, m = gpu_batch(ids, mask, ref_max, lens, order[s:e], tok, gen, T)
        with autocast:
            loss = loss_on(model, x, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    return loss.item(), len(plan)


def episode_records(tok, targets, eps, found, rng):
    """(context ids, proof ids) fine-tune records from one round's episodes; see the retry knobs."""
    data, n_real = [], 0
    for t, row in zip(targets, eps, strict=True):
        proofs = list(found[t["name"]].values())
        fails = [f for ep in row for f in ep["fails"]]
        if not proofs or (not fails and ARM != "repair"):
            continue
        recs = [(ep["fails"], ep["proof"]) for ep in row if ep["proof"] and ep["fails"]]
        rng.shuffle(recs)
        recs = recs[:EP_PER_THM]
        n_real += len(recs)
        if ARM == "repair":
            prompt_ids = tok.encode_prompt(t["prompt"])
            for fl, proof in recs:
                q = align(tok, tok.encode_proof(proof), fl[-1][0])
                if len(context_ids(prompt_ids, fl)) + len(q) <= MAX_CTX:
                    data.append((context_ids(prompt_ids, fl), q, True))
            made = [
                repair_record(tok, t["prompt"], rng.choice(proofs), rng)
                for _ in range(CORRUPT_PER_THM)
            ]
            data += [x for x in made if x]
            continue
        while len(recs) < EP_PER_THM:
            k = rng.randint(1, min(RETRY_ATTEMPTS - 1, len(fails)))
            recs.append((rng.sample(fails, k), rng.choice(proofs)))
        prompt_ids = tok.encode_prompt(t["prompt"])
        for fl, proof in recs:
            q = tok.encode_proof(proof)
            while fl and len(context_ids(prompt_ids, fl)) + len(q) > MAX_CTX:
                fl = fl[:-1]
            if fl:
                data.append((context_ids(prompt_ids, fl), q))
    return data, n_real


def rl_retry(model, tok, targets, train_recs, oracle, device, log, rng):
    """The iid arm's expert iteration with episodes for samples and episode records in the mix."""
    found = {t["name"]: {} for t in targets}
    rounds, round_secs, per_round = 0, 0.0, []
    while oracle.left() > 1.15 * round_secs:
        t0 = oracle.used()
        eps = oracle.sample_episodes(
            model, targets, RETRY_EP, RETRY_ATTEMPTS, "blank" if ARM == "blank" else "error"
        )
        for t, row in zip(targets, eps, strict=True):
            for ep in row:
                if ep["proof"]:
                    found[t["name"]].setdefault(norm(ep["proof"]), ep["proof"])
        mix = []
        for t in targets:
            proofs = list(found[t["name"]].values())
            rng.shuffle(proofs)
            mix += [(t["prompt"], proof) for proof in proofs[:MAX_PER_THM]] * RL_WEIGHT
        n_rl = len(mix)
        if not n_rl:
            break
        mix += [(r["prompt"], r["proof"]) for r in rng.sample(train_recs, RETAIN)]
        data = [(tok.encode_prompt(p), tok.encode_proof(q)) for p, q in mix]
        ep_data, n_real = episode_records(tok, targets, eps, found, rng)
        loss, steps = finetune(model, tok, data + ep_data, rng, device, by_tokens=True)
        rounds += 1
        round_secs = oracle.used() - t0
        per_round.append(
            {
                "rl_records": n_rl,
                "episode_records": len(ep_data),
                "real_retries": n_real,
                "ft_steps": steps,
            }
        )
        log(
            f"rl round {rounds}: {n_rl} RL records, {len(ep_data)} episode records "
            f"({n_real} real retries), fine-tune loss {loss:.4f} in {steps} steps, "
            f"round {round_secs:.0f}s, {oracle.left():.0f}s left"
        )
    return {
        "rounds": rounds,
        "arm": ARM,
        "episodes": RETRY_EP,
        "max_attempts": RETRY_ATTEMPTS,
        "per_round": per_round,
    }


def rl(model, tok, targets, train_recs, oracle, budget_secs, device, seed, log):
    rng = random.Random(seed)
    if ARM in ("retry", "blank", "repair"):
        return rl_retry(model, tok, targets, train_recs, oracle, device, log, rng)
    found = {t["name"]: {} for t in targets}  # name -> {normalized proof: proof text}
    rounds, round_secs, retries, ft_steps = 0, 0.0, [], []
    # Start a round only if one as long as the last (plus 15%) still fits in the budget.
    while oracle.left() > 1.15 * round_secs:
        t0 = oracle.used()
        model.n_loops = RL_FIRST_LOOPS
        rows = oracle.sample(model, targets, RL_K)
        failed = [t for t, row in zip(targets, rows, strict=True) if not row["proofs"]]
        rng.shuffle(failed)
        if ARM == "resume_rl" and failed:
            eps = oracle.sample_episodes(model, failed, 3, 8, "resume")
            part_rows = [
                {"name": t["name"], "proofs": [ep["proof"] for ep in row if ep["proof"]]}
                for t, row in zip(failed, eps, strict=True)
            ]
            retries.append(
                {
                    "round": rounds + 1,
                    "targets": len(failed),
                    "cracked": sum(bool(r["proofs"]) for r in part_rows),
                }
            )
            rows += part_rows
            failed = []
        for i, n in enumerate(RL_RETRY_LOOPS):
            part = failed[i :: len(RL_RETRY_LOOPS)]
            if not part:
                continue
            model.n_loops = n
            part_rows = oracle.sample(model, part, RL_RETRY_K)
            cracked = sum(bool(row["proofs"]) for row in part_rows)
            retries.append(
                {"round": rounds + 1, "loops": n, "targets": len(part), "cracked": cracked}
            )
            rows += part_rows
        for row in rows:
            for proof in row["proofs"]:
                found[row["name"]].setdefault(norm(proof), proof)
        mix = []
        for t in targets:
            proofs = list(found[t["name"]].values())
            rng.shuffle(proofs)
            mix += [(t["prompt"], proof) for proof in proofs[:MAX_PER_THM]] * RL_WEIGHT
        n_rl = len(mix)
        if not n_rl:
            break
        mix += [(r["prompt"], r["proof"]) for r in rng.sample(train_recs, RETAIN)]
        data = [(tok.encode_prompt(p), tok.encode_proof(q)) for p, q in mix]
        loss, steps = finetune(model, tok, data, rng, device)
        ft_steps.append(steps)
        rounds += 1
        round_secs = oracle.used() - t0
        log(
            f"rl round {rounds}: {n_rl} RL records, fine-tune loss {loss:.4f} in {steps} steps, "
            f"round {round_secs:.0f}s, {oracle.left():.0f}s left"
        )
    model.n_loops = DEFAULT_LOOPS
    return {
        "rounds": rounds,
        "first_loops": RL_FIRST_LOOPS,
        "k_first": RL_K,
        "retry_loops": RL_RETRY_LOOPS,
        "k_retry": RL_RETRY_K,
        "retries": retries,
        "ft_records": FT_RECORDS,
        "ft_steps": ft_steps,
    }
