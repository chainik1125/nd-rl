# imports the fork (model, tokenizer, train, ...), which is not resolvable here
# pyright: reportMissingImports=false
"""AGENT-EDITABLE: pretraining and RL. This is the only file the autoresearch agent changes.

Contract with harness.py (frozen):

  pretrain(train_recs, budget_secs, device, seed, log) -> (model, tok, stats)

  train_recs   list of dicts {prompt, proof, n_lines<=6, key, ...}; the frozen cap-6 Stage-1 data
  budget_secs  wall-clock you are charged for, from call to return, INCLUDING model construction
               and torch.compile. Overrun by more than the harness grace -> the run is a timeout.
  model        nn.Module; <= PARAM_CAP unique params (checked again after rl). The harness samples
               from it with the fork's sample.generate, so it must provide:
                 - forward(idx) -> logits (B,T,V)  and  forward(idx, pos=, mask=, caches=) with a
                   per-block KV cache dict, exactly like the fork's GPT (model.blocks must exist)
                 - an integer `n_thoughts`: continuous thoughts per proof line in eval mode
                   (the harness's KNOB). The harness sets it to EVAL_LEVEL for the scored eval
                   and to DIAG_LEVEL for the diagnostic one.
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
               attempts per target from the model in eval mode at its current n_thoughts, and
               returns per target {name, prompt, n_lines, proofs, n_ok, n_tried}; `proofs` are the
               distinct attempts Lean AND nd_verify accepted. This is the only source of new
               training data.
               oracle.used() / oracle.left(): seconds of the RL budget charged / remaining.
  budget_secs  RL_SECS: wall-clock inside rl() minus the oracle's Lean-gate time. Returning more
               than the grace late, or calling the oracle with nothing left, is a timeout.
  The model is trained in place. Everything else about RL is yours: rounds, samples per target,
  thought counts when sampling, which proofs to keep, weighting, replay, optimizer, loss.

Anything imported from the fork (ND_FORK) is fair game to copy in here and modify.
"""

import math
import random
import time

import torch
import torch._inductor.config  # noqa: F401  (explicit, so the attribute resolves for pyright)
from lean_tok import MAXN  # fork
from model import GPT, Block  # fork
from normalize import norm  # fork
from tokenizer import make_tokenizer  # fork
from train import loss_on  # fork

# ---- knobs
N_LAYER, D, N_HEAD = 6, 384, 8  # run 198's untied blocks
# One thinking round before each proof line. A line starts after a trigger token: `;` between
# Lean tactics, and `by`, which ends the theorem statement and opens every nested block. After a
# trigger the model takes n thought positions whose inputs are learned slot vectors (pause
# tokens, Goyal et al. 2023: not a token and not the model's own output), and the last thought's
# output predicts the line's first token. Because a slot's input does not depend on the model's
# state, a training step is ONE causal pass over the expanded sequence, like run 198's, where
# experiments 701-703 fed the trigger's final state into the thoughts and needed one pass per
# line. Pretraining draws n per batch from TRAIN_THOUGHTS; rl()'s fine-tune uses FT_THOUGHTS
# (below); eval and sampling apply model.n_thoughts, which the harness sets for its evals and
# rl() sets before each oracle call. DEFAULT_THOUGHTS is used for the validation curve and the
# harness's post-pretraining greedy check.
TRAIN_THOUGHTS = (0, 1, 2, 3, 4, 5, 6, 7, 8)
DEFAULT_THOUGHTS = 8
MAX_THOUGHTS = 16  # rows of the thought-slot embedding; n_thoughts above this is an error
VAL_THOUGHTS = (0, 2, 8)  # val loss at each count, once, when pretraining ends
# Sampling keeps one KV cache row per sample with its own write pointer. Each cache is sized for
# the sampler's tokens plus n_thoughts * MAX_THINKS thought positions; a sample that has used its
# share (a degenerate one emitting `;` forever) goes on without thinking.
MAX_THINKS = 48
# The plan probe: at every thinking round, a head that exists only during training must read the
# line's "move" off the last thought: the token that follows the line's `:=` or `exact` (the name
# applied, `⟨`, `Or.inl`, the `(` of a `fun`, ...). It is computed from the tokens alone, so it
# stays on during RL fine-tuning, on proofs the model found.
PROBE_WEIGHT = 0.3
# ---- RL knobs: 704's expert iteration (every target sampled at RL_THOUGHTS = 8 thoughts, up to
# MAX_PER_THM proofs per solved target weighted RL_WEIGHT times in the fine-tune mix with RETAIN
# replayed pretraining records) with the round's sample budget, RL_K x targets, reallocated as
# targets saturate. A target with MAX_PER_THM distinct accepted proofs is saturated: the mix takes
# at most that many, so more samples of it can only swap proofs. Saturated targets get RL_K_SAT
# samples; the rest share what is left of the budget, RL_K_MAX each at most. With nothing solved
# every target gets RL_K = 32, 704's round exactly; once most targets are saturated the unsolved
# ones get 64 and the round costs about half of 704's. The sibling loop's 8 + 24 retries (706)
# focused on the failures from round 1 and stalled on a weak seed; this keeps 704's flat sampling
# until there is something to focus on. The fine-tune runs over FT_RECORDS records in
# pretraining's token-budget batches (706's port of the sibling's finetune: the same record
# count as 704's 600 steps of 128, about twice as fast), AdamW with linear warmup then cosine to
# lr/10. Rounds repeat until another would not fit the budget. RL runs entirely at the eval
# level: every sample is drawn at 8 thoughts per line and the fine-tune trains at 8 only
# (FT_THOUGHTS), never at 0. In 701 the fine-tune drew n from 0..8 and the no-thought path caught
# up within one round (sample accuracy at 0 thoughts went from 3-6% to 28-33%, against 39% at 8):
# the proofs found with thoughts were being distilled into the path that does not think. Here
# they are only ever learned with thoughts.
RL_THOUGHTS = 8
RL_K, RL_K_SAT, RL_K_MAX = 32, 8, 64
FT_THOUGHTS = (8,)
FT_RECORDS, FT_LR, FT_WARMUP = 600 * 128, 3e-4, 50
RETAIN, MAX_PER_THM, RL_WEIGHT = 20000, 4, 4
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
    `mask` is the additive ALiBi bias (B or 1, H, T, S) that LineThoughtGPT builds and cos/sin
    are unused. Three ways to supply earlier keys and values: none (`cache` None, one full
    pass); a list, in training, holding the keys and values of the rounds already run, which this
    call extends by concatenation so gradients flow; and sample_step, in sampling, which writes
    into preallocated buffers at a position of each row's own."""

    def __init__(self, d, h):
        super().__init__(d, h)
        self.fc1, self.fc2 = torch.nn.Linear(d, D_FF), torch.nn.Linear(D_FF, d)
        self.ln_attn_out, self.ln_mlp_out = torch.nn.LayerNorm(d), torch.nn.LayerNorm(d)

    def attn(self, x, cos, sin, mask, cache):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if cache is not None:
            if cache:
                k, v = torch.cat((cache[0], k), 2), torch.cat((cache[1], v), 2)
            cache[:] = [k, v]
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask.to(q.dtype))
        return self.proj(y.transpose(1, 2).reshape(B, T, D))

    def forward(self, x, cos, sin, mask=None, cache=None):
        x = x + self.ln_attn_out(self.attn(self.ln1(x), cos, sin, mask, cache))
        return x + self.ln_mlp_out(self.fc2(torch.nn.functional.gelu(self.fc1(self.ln2(x)))))

    def sample_step(self, x, bias, cache, rows, cols):
        """forward() for sampling. x (b, T, D) are new positions of the samples `rows` (b, 1),
        all of them if b is the batch; their keys and values go to columns `cols` (b, T) of the
        buffers, and they attend to those samples' first bias.size(-1) columns."""
        b, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if "k" not in cache:
            # zeros, not empty: a row reads columns past its own pointer (masked), and a masked
            # key that happens to hold inf or nan still poisons the attention output
            cache["k"] = k.new_zeros(cache["B"], self.h, cache["max"], D // self.h)
            cache["v"] = v.new_zeros(cache["B"], self.h, cache["max"], D // self.h)
        cache["k"][rows, :, cols] = k.transpose(1, 2)  # indexed dims first: (b, T, H, hd)
        cache["v"][rows, :, cols] = v.transpose(1, 2)
        S, every = bias.size(-1), b == cache["B"]
        k = cache["k"][:, :, :S] if every else cache["k"][rows[:, 0], :, :S]
        v = cache["v"][:, :, :S] if every else cache["v"][rows[:, 0], :, :S]
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        x = x + self.ln_attn_out(self.proj(y.transpose(1, 2).reshape(b, T, D)))
        return x + self.ln_mlp_out(self.fc2(torch.nn.functional.gelu(self.fc1(self.ln2(x)))))


class LineThoughtGPT(GPT):
    """Run 198's model (the fork's GPT with half-ALiBi, half-NoPE heads) with one thinking round
    before each proof line: n pause positions (Goyal et al. 2023) after every trigger.

    After a trigger token the sequence gets n thought positions whose inputs are slot_emb[j], a
    learned vector per slot, the same at every line. They attend to everything before them and
    to each other causally, sit at the trigger's position on the ALiBi axis (so distances
    between tokens do not depend on n), and the last one's output replaces the trigger's as the
    predictor of the next token. With n = 0 this is exactly run 198's network.

    Training (no caches) goes through run(), one causal pass over the expanded sequence;
    sampling (`caches` given) through cached(), which does the thinking inside the forward call,
    so the fork's sampler needs no change."""

    def __init__(self, vocab, n_layer, d, n_head, triggers, seed, max_len=1024):
        super().__init__(vocab, n_layer, d, n_head, max_len)
        slopes = torch.tensor(ALIBI_SLOPES).view(1, n_head, 1, 1)
        self.register_buffer("slopes", slopes, persistent=False)
        self.register_buffer("triggers", torch.tensor(triggers), persistent=False)
        self.slot_emb = torch.nn.Parameter(torch.randn(MAX_THOUGHTS, d) * 0.02)
        self.n_thoughts, self.t_rng = DEFAULT_THOUGHTS, random.Random(seed)

    def slots(self, b, n):
        """The n thought inputs (b, n, D) of one round, for b rows: the slot vectors."""
        return self.slot_emb[:n].expand(b, n, -1)

    def bias(self, qpos, kpos, allowed):
        """ALiBi bias (B, H, Q, S) for queries at positions qpos (B, Q) over keys at kpos (B, S),
        with -inf wherever `allowed` (B, Q, S) is False."""
        rel = qpos[:, None, :, None] - kpos[:, None, None, :]
        return (-self.slopes * rel).masked_fill(~allowed[:, None], float("-inf"))

    def forward(self, idx, pos=None, mask=None, caches=None):
        if caches is not None:
            return self.head(self.ln_f(self.cached(idx, pos, mask, caches)))
        n = self.t_rng.choice(TRAIN_THOUGHTS) if self.training else self.n_thoughts
        return self.head(self.ln_f(self.run(idx, n)))

    # ---- training: one causal pass over the expanded sequence, thoughts inserted after triggers

    def run(self, idx, n):
        """Final hidden states (B, S, D) of right-padded rows idx with n thoughts per line, in
        idx's own layout; at a trigger it is the last thought's output, which is what predicts
        the next token. The expanded layout puts token t of a row at column
        t + n * (triggers before t) and a trigger's n thoughts at the columns after it; rows
        with fewer triggers than the longest leave empty columns at the end, masked as keys."""
        B, S = idx.shape
        dev, D = idx.device, self.emb.embedding_dim
        dtype = torch.bfloat16 if torch.is_autocast_enabled() else torch.float32
        ar = torch.arange(S, device=dev)[None].expand(B, S)
        rows = torch.arange(B, device=dev)[:, None].expand(B, S)
        trig = torch.isin(idx, self.triggers)
        col = ar + n * (trig.cumsum(1) - trig.long())  # each token's column
        E = S + n * int(trig.sum(1).max())
        e = self.emb(idx)
        x, pos, valid = e.new_zeros(B, E, D), idx.new_zeros(B, E), trig.new_zeros(B, E)
        x[rows, col], pos[rows, col], valid[rows, col] = e, ar, True
        if n:
            tr, tc, tp = rows[trig][:, None], col[trig][:, None], ar[trig][:, None]
            tc = tc + 1 + torch.arange(n, device=dev)[None]  # (triggers, n)
            x[tr, tc], pos[tr, tc], valid[tr, tc] = self.slot_emb[:n], tp, True
        ec = torch.arange(E, device=dev)
        # column 0 is a token in every row, so no query is left without a key
        allowed = valid[:, None, :] & (ec[None, :] <= ec[:, None])[None]
        bias = self.bias(pos, pos, allowed).to(dtype)
        for b in self.blocks:
            x = b(x, None, None, bias, None)
        out = col + n * trig.long()  # a trigger reads out at its last thought
        return x.gather(1, out[..., None].expand(B, S, D))

    # ---- sampling: the fork's incremental decoding, thinking after every trigger

    def cached(self, idx, pos, mask, caches):
        """One sampler call: the left-padded prompts first, then one new token per sample. After
        the call's last token, the samples for which it is a trigger think, and their returned
        state is the last thought's. Samples differ in how many thoughts they have taken, so
        each has its own write pointer into the buffers and its own key positions and validity,
        kept in caches[0]; the sampler's key mask is only read for the prompt's padding."""
        c, (B, T), n = caches[0], idx.shape, self.n_thoughts
        assert n <= MAX_THOUGHTS, n
        if "ptr" not in c:
            c["tokens_left"] = c["max"] - T
            for cache in caches:
                cache["max"] += n * MAX_THINKS
                cache["B"] = B
            c["ptr"] = pos.new_zeros(B)
            c["kpos"] = pos.new_zeros(B, c["max"])
            c["kvalid"] = mask.new_zeros(B, c["max"])
            valid = mask[:, 0, -1]  # (B, T): which prompt columns are real tokens
        else:
            c["tokens_left"] -= T
            valid = mask.new_ones(B, T)
        every = torch.arange(B, device=idx.device)[:, None]
        h = self.cached_pass(self.emb(idx), pos, valid, caches, every)
        think = torch.isin(idx[:, -1], self.triggers)
        think &= c["ptr"] + n + c["tokens_left"] <= c["max"]
        if n and bool(think.any()):
            rows = every[think]
            x = self.slots(rows.size(0), n)
            real = valid.new_ones(x.shape[:2])
            out = self.cached_pass(x, pos[think, -1:].expand(-1, n), real, caches, rows)
            h[think, -1] = out[:, -1]
        return h

    def cached_pass(self, x, qpos, valid, caches, rows):
        c, T = caches[0], x.size(1)
        at = c["ptr"][rows[:, 0]]
        cols = at[:, None] + torch.arange(T, device=x.device)[None]
        c["kpos"][rows, cols], c["kvalid"][rows, cols] = qpos, valid
        c["ptr"][rows[:, 0]] = at + T
        S = int(at.max()) + T
        col = torch.arange(S, device=x.device)[None, None, :]
        kvalid = c["kvalid"][rows[:, 0], :S]
        allowed = (kvalid[:, None, :] & (col <= cols[:, :, None])) | (col == cols[:, :, None])
        bias = self.bias(qpos, c["kpos"][rows[:, 0], :S], allowed)
        for b_, cache in zip(self.blocks, caches, strict=True):
            x = b_.sample_step(x, bias, cache, rows, cols)
        return x


class MTPHead(torch.nn.Module):
    """The MTP module. A separate nn.Module, never registered on the model, so the harness's
    parameter count, RL AdamW and sampler never see it; it is dropped when pretrain() returns."""

    def __init__(self, d, h):
        super().__init__()
        self.ln_h, self.ln_e = torch.nn.LayerNorm(d), torch.nn.LayerNorm(d)
        self.proj = torch.nn.Linear(2 * d, d, bias=False)
        self.block = PeriLNBlock(d, h)


class Aux(torch.nn.Module):
    """What exists only during training: the MTP module (pretraining) and the plan probe
    (pretraining and RL). Never registered on the model, so the harness's parameter count and
    sampler never see it. pretrain() leaves it in AUX for rl()."""

    def __init__(self, d, h, vocab):
        super().__init__()
        self.mtp = MTPHead(d, h)
        self.ln_probe, self.probe = torch.nn.LayerNorm(d), torch.nn.Linear(d, vocab)


AUX = None


def next_move(x, moves):
    """For each position of the token rows x (B, S): the token after the next `:=` or `exact`
    strictly to its right (-100 if there is none). At a trigger this is the coming line's move."""
    B, S = x.shape
    at = torch.where(torch.isin(x, moves), torch.arange(S, device=x.device)[None], S)
    at = at.flip(1).cummin(1).values.flip(1)  # the nearest one at or after each position
    label_at = torch.cat((at[:, 1:], at.new_full((B, 1), S)), 1) + 1
    label = x.gather(1, label_at.clamp(max=S - 1))
    return label.masked_fill(label_at >= S, -100)


def train_loss(model, aux, x, m, tok, mtp=True, thoughts=TRAIN_THOUGHTS):
    """Loss of one training batch at a thought count drawn for it from `thoughts`: the fork's
    loss_on (next-token CE on proof tokens); plus, in pretraining, MTP_WEIGHT times the CE of the
    MTP module's token-(t+2) prediction at the same positions; plus PROBE_WEIGHT times the plan
    probe's CE at the triggers when the batch has thoughts. Returns (total, main, mtp, probe)."""
    n = model.t_rng.choice(thoughts)
    h = model.run(x[:, :-1], n)
    logits = model.head(model.ln_f(h))
    V = logits.size(-1)
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, V).float(), x[:, 1:].reshape(-1), reduction="none"
    )
    main = (ce * m[:, 1:].reshape(-1)).sum() / m[:, 1:].sum()
    total, aux_ce, probe = main, main.new_zeros(()), main.new_zeros(())
    if mtp:
        # Position t (0..T-3) gets h_t and emb(x_{t+1}); causal attention over these keeps every
        # prediction of x_{t+2} a function of x_{<=t+1}.
        head = aux.mtp
        z = head.proj(torch.cat((head.ln_h(h[:, :-1]), head.ln_e(model.emb(x[:, 1:-1]))), -1))
        ar = torch.arange(z.size(1), device=z.device)[None]
        z = head.block(z, None, None, model.bias(ar, ar, ar[:, None, :] <= ar[:, :, None]), None)
        ce2 = torch.nn.functional.cross_entropy(
            model.head(model.ln_f(z)).reshape(-1, V).float(), x[:, 2:].reshape(-1), reduction="none"
        )
        aux_ce = (ce2 * m[:, 2:].reshape(-1)).sum() / m[:, 2:].sum().clamp(min=1)
        total = total + MTP_WEIGHT * aux_ce
    if n:
        moves = torch.tensor([tok.stoi[":="], tok.stoi["exact"]], device=x.device)
        label = next_move(x, moves)[:, :-1]
        at = torch.isin(x[:, :-1], model.triggers) & (label >= 0)
        if bool(at.any()):
            probe = torch.nn.functional.cross_entropy(
                aux.probe(aux.ln_probe(h[at])).float(), label[at]
            )
            total = total + PROBE_WEIGHT * probe
    return total, main, aux_ce, probe


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


def pretrain(train_recs, budget_secs, device, seed, log):
    global AUX
    t0 = time.time()
    tok = make_tokenizer(TOK_MODE)
    data = [(tok.encode_prompt(r["prompt"]), tok.encode_proof(r["proof"])) for r in train_recs]
    ids, mask, ref_max, lens = to_gpu(data, tok, device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    triggers = (tok.stoi[";"], tok.stoi["by"])
    model = LineThoughtGPT(tok.vocab_size, N_LAYER, D, N_HEAD, triggers, seed)
    model.blocks = torch.nn.ModuleList(PeriLNBlock(D, N_HEAD) for _ in range(N_LAYER))
    model.blocks.apply(model._init)  # fork init: zero biases
    model = model.to(device)
    AUX = aux = Aux(D, N_HEAD, tok.vocab_size)
    aux.apply(model._init)
    aux = aux.to(device)
    mtp = aux.mtp
    # No torch.compile: the expanded length changes every batch and the blocks hit the recompile
    # limit (measured on the 2026-10-06 Coconut spike: 10.9 steps/s compiled, 10.1 eager).
    muon_lr, muon_graph = torch.zeros((), device=device), None
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
    curve, recent, recent_aux, recent_probe, next_point = [], [], [], [], CURVE_EVERY
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
            loss, main, mtp_ce, probe = train_loss(model, aux, x, m, tok)
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
            recent_probe.append(probe.item())
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
                    # batches drawn with zero thoughts count as 0 here
                    "probe_loss": sum(recent_probe) / len(recent_probe) if recent_probe else None,
                }
            )
            log(
                f"pretrain step {step} train {curve[-1]['train_loss']:.4f} "
                f"val {curve[-1]['val_loss']:.4f} lr {lr:.2e} {curve[-1]['secs']}s"
            )
            recent, recent_aux, recent_probe = [], [], []
            next_point += CURVE_EVERY
    val_by_thoughts = {}
    for n in VAL_THOUGHTS if step else ():  # no steps: a checkpoint rebuild, not a training run
        model.n_thoughts = n
        val_by_thoughts[n] = val_loss(model, ids, mask, ref_max, lens, tok, gen, autocast)
    model.n_thoughts = DEFAULT_THOUGHTS
    log(f"pretrain val loss by thoughts: {val_by_thoughts}")
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
            "train_thoughts": TRAIN_THOUGHTS,
            "default_thoughts": DEFAULT_THOUGHTS,
            "val_by_thoughts": val_by_thoughts,
            "curve": curve,
        },
    )


def finetune(model, tok, data, rng, device):
    """The RL fine-tune: AdamW (linear warmup then cosine to lr/10) over FT_RECORDS records in
    token-budget batches (to_gpu / token_batches / gpu_batch, as pretraining), with the plan
    probe's loss kept on. Every batch runs at a thought count drawn from FT_THOUGHTS: the eval
    level only. The probe trains along; the MTP module is not used. Returns (last loss, steps)."""
    ids, mask, ref_max, lens = to_gpu(data, tok, device)
    gen = torch.Generator(device=device)
    gen.manual_seed(rng.getrandbits(31))
    plan, seen = [], 0  # (order, start, end, padded length) per step, epoch after epoch
    while seen < FT_RECORDS:
        order, cuts = token_batches(lens, len(data), gen, rng, first=0)
        for s, e, T in cuts:
            if seen >= FT_RECORDS:
                break
            plan.append((order, s, e, T))
            seen += e - s
    params = (
        list(model.parameters()) + list(AUX.ln_probe.parameters()) + list(AUX.probe.parameters())
    )
    opt = torch.optim.AdamW(params, lr=FT_LR, weight_decay=0.1, betas=(0.9, 0.95))
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
            loss, main, _, _ = train_loss(model, AUX, x, m, tok, mtp=False, thoughts=FT_THOUGHTS)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    model.eval()
    return main.item(), len(plan)


def rl(model, tok, targets, train_recs, oracle, budget_secs, device, seed, log):
    rng = random.Random(seed)
    found = {t["name"]: {} for t in targets}  # name -> {normalized proof: proof text}
    rounds, round_secs, allocation, ft_steps = 0, 0.0, [], []
    model.n_thoughts = RL_THOUGHTS
    # Start a round only if one as long as the last (plus 15%) still fits in the budget.
    while oracle.left() > 1.15 * round_secs:
        t0 = oracle.used()
        # The round's RL_K x targets samples: RL_K_SAT each for the saturated targets (MAX_PER_THM
        # distinct proofs found), the rest shared evenly by the others, RL_K_MAX each at most.
        sat = [t for t in targets if len(found[t["name"]]) >= MAX_PER_THM]
        rest = [t for t in targets if len(found[t["name"]]) < MAX_PER_THM]
        k_rest = min(RL_K_MAX, (RL_K * len(targets) - RL_K_SAT * len(sat)) // max(len(rest), 1))
        rows = oracle.sample(model, sat, RL_K_SAT) if sat else []
        rows += oracle.sample(model, rest, k_rest) if rest else []
        allocation.append({"round": rounds + 1, "saturated": len(sat), "k_rest": k_rest})
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
            f"rl round {rounds}: {len(sat)} saturated targets x {RL_K_SAT}, {len(rest)} x "
            f"{k_rest}, {n_rl} RL records, fine-tune loss {loss:.4f} in {steps} steps, "
            f"round {round_secs:.0f}s, {oracle.left():.0f}s left"
        )
    model.n_thoughts = DEFAULT_THOUGHTS
    return {
        "rounds": rounds,
        "thoughts": RL_THOUGHTS,
        "k": RL_K,
        "k_saturated": RL_K_SAT,
        "k_max": RL_K_MAX,
        "allocation": allocation,
        "ft_records": FT_RECORDS,
        "ft_steps": ft_steps,
        "ft_thoughts": FT_THOUGHTS,
        "ft_lr": FT_LR,
    }
