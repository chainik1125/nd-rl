"""Autoresearch run 198's pretraining recipe (Robbie's workstream), for the ``lean_seq`` pipeline.

Run 198 (``198-claude_lean-muon-no-rms-retest``, branch ``autoresearch-pretrain``, where
``code/experiments/current/autoresearch/RESULTS.md`` has every run) is the best pretraining stage
the autoresearch loop found: 681 of 1,108 dev theorems of true length >= 7 solved after a fixed
expert-iteration stage, 3-seed mean, at <= 10M parameters and 300 s of pretraining on one A40.
This module ports that stage so it runs with ``LeanSeqTokenizer`` and the pipeline's
``generate_bodies`` / ``save_checkpoint``:

- ``RecipeModel``: 6 x 384 transformer, 8 heads, MLP 1280. Half the heads get an ALiBi distance
  penalty (Press et al. 2021), the other half no positional signal (NoPE). Peri-LN blocks
  (Kim et al. 2025): pre-LN plus a LayerNorm on each branch output. Same call signature as
  ``nd_model.ProofModel`` (``forward(tokens, positions, attention_mask, cache, cache_start)``,
  ``allocate_cache``), so the pipeline's sampler takes it unchanged.
- ``pretrain``: Muon with Polar Express coefficients (Amsel et al. 2025) on the hidden matrices
  and the readout, AdamW on the rest; a multi-token-prediction head (DeepSeek-V3 2024) used only
  during pretraining; orthogonal init; token-budget batches (length-sorted runs of
  ``count * max_len <= TOK_BUDGET`` tokens); a random name shift per proof; cosine learning rate
  over wall-clock with no warmup.
- ``load_checkpoint`` for checkpoints saved with ``nd_model.save_checkpoint``, and
  ``load_autoresearch_checkpoint`` for the loop's own ``final.pt`` files on Hugging Face
  (``robbiethompson2018/nd-rl-checkpoints``, ``runs/198-claude_lean-muon-no-rms-retest_s{0,1,2}``).

Not ported: 198's per-block ``torch.compile`` and CUDA-graphed Muon step (+23 % steps on torch
2.9.1; the loop measured no score change from them, 680 vs 684). The hyperparameters were tuned
for a 10M model, 300 s and cap-6 data; they are untested at other sizes and data.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from code.shared_code.lean_seq import MAX_NAMES, LeanSeqTokenizer

# ---- 198's knobs
LR, MIN_LR_FRAC, WD = 1e-3, 0.3, 0.1
MUON_LR, MUON_MOMENTUM = 0.01, 0.95
# Ten times the AdamW group's per-step multiplicative decay (lr * wd) on the Muon matrices.
MUON_WD = 10 * WD * LR / MUON_LR
# Polar Express per-iteration quintic coefficients, as in modded-nanogpt (5 iterations, safety
# factor 2e-2): singular values land in ~[0.86, 1.14] instead of the fixed quintic's ~[0.68, 1.13].
NS_COEFFS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)
# Token-budget batches (fairseq max-tokens): 128 x 144 matches the real tokens of 128 random
# records (mean ~135), whose padding to the batch maximum (~400) made every step ~3x its content.
TOK_BUDGET = 128 * 144
MTP_WEIGHT = 0.3  # DeepSeek-V3's early-training value
VAL_BS, CURVE_EVERY = 128, 15  # val batch size; seconds between curve points


@dataclass
class RecipeConfig:
    vocab_size: int = 107
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1280  # != 3 * d_model, which muon_step's QKV-by-shape test relies on
    max_seq_len: int = 1024


def fork_init(module: nn.Module) -> None:
    """The fork GPT's init: N(0, 0.02) Linear/Embedding weights, zero biases."""
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            bias = getattr(m, "bias", None)
            if isinstance(bias, torch.Tensor):
                nn.init.zeros_(bias)


def alibi_slopes(n_heads: int) -> list[float]:
    """Heads 0..H/2-1: the standard ALiBi slopes for H/2 heads, 2^(-8 i / (H/2)) (1/4 .. 1/256
    at 8 heads). The other half: slope 0, no positional signal, so they can look citations up by
    content at any distance."""
    n = n_heads // 2
    return [2.0 ** (-8.0 * (i + 1) / n) for i in range(n)] + [0.0] * (n_heads - n)


class PeriLNBlock(nn.Module):
    """Pre-LN block plus a LayerNorm on each branch output. Attention takes the additive ALiBi
    bias (B or 1, H, T, S) that ``RecipeModel`` builds once per forward. Parameter names match
    autoresearch 198's, so its checkpoints load."""

    def __init__(self, c: RecipeConfig) -> None:
        super().__init__()
        self.h = c.n_heads
        self.ln1, self.ln2 = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        self.qkv, self.proj = nn.Linear(c.d_model, 3 * c.d_model), nn.Linear(c.d_model, c.d_model)
        self.fc1, self.fc2 = nn.Linear(c.d_model, c.d_ff), nn.Linear(c.d_ff, c.d_model)
        self.ln_attn_out, self.ln_mlp_out = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)

    def attn(
        self, x: torch.Tensor, bias: torch.Tensor, cache: Any, cache_start: int
    ) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if cache is not None:
            end = cache_start + T
            cache[0][:, :, cache_start:end] = k
            cache[1][:, :, cache_start:end] = v
            k, v = cache[0][:, :, :end], cache[1][:, :, :end]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        return self.proj(y.transpose(1, 2).reshape(B, T, D))

    def forward(
        self, x: torch.Tensor, bias: torch.Tensor, cache: Any = None, cache_start: int = 0
    ) -> torch.Tensor:
        x = x + self.ln_attn_out(self.attn(self.ln1(x), bias, cache, cache_start))
        return x + self.ln_mlp_out(self.fc2(F.gelu(self.fc1(self.ln2(x)))))


class RecipeModel(nn.Module):
    """198's network, with ``nd_model.ProofModel``'s call signature. ``attention_mask`` is a bool
    (B, 1, T, S) mask, True where a query may attend; without it attention is causal."""

    def __init__(self, config: RecipeConfig | None = None) -> None:
        super().__init__()
        self.config = c = config or RecipeConfig()
        self.emb = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(PeriLNBlock(c) for _ in range(c.n_layers))
        self.ln_f = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        slopes = torch.tensor(alibi_slopes(c.n_heads)).view(1, c.n_heads, 1, 1)
        self.register_buffer("slopes", slopes, persistent=False)
        fork_init(self)

    def features(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cache: Any = None,
        cache_start: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The trunk: (final hidden state before ln_f, the ALiBi bias it used)."""
        T = tokens.shape[1]
        if positions is None:
            positions = torch.arange(cache_start, cache_start + T, device=tokens.device)
        if positions.ndim == 1:
            positions = positions[None]
        key_pos = positions
        if cache is not None:  # every layer's tuple shares one key-position buffer
            key_pos_buffer = cache[0][2]
            key_pos_buffer[:, cache_start : cache_start + T] = positions
            key_pos = key_pos_buffer[:, : cache_start + T]
        rel = positions[:, None, :, None] - key_pos[:, None, None, :]  # (B or 1, 1, T, S)
        slopes: torch.Tensor = self.slopes  # type: ignore[assignment]
        bias = -slopes * rel.to(slopes.dtype)
        bias = bias.masked_fill(rel < 0 if attention_mask is None else ~attention_mask, -math.inf)
        # A query with no visible key (a left-pad row in a sampled batch) would make NaNs that leak
        # through the padded keys' values; let it attend everywhere instead.
        bias = bias.masked_fill(torch.isinf(bias).all(-1, keepdim=True), 0.0)
        x = self.emb(tokens)
        for i, block in enumerate(self.blocks):
            x = block(x, bias, None if cache is None else cache[i], cache_start)
        return x, bias

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cache: Any = None,
        cache_start: int = 0,
    ) -> torch.Tensor:
        h, _ = self.features(tokens, positions, attention_mask, cache, cache_start)
        return self.head(self.ln_f(h))

    def allocate_cache(
        self, batch_size: int, max_length: int, dtype: torch.dtype | None = None
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        p = next(self.parameters())
        c = self.config
        shape = (batch_size, c.n_heads, max_length, c.d_model // c.n_heads)
        key_pos = torch.zeros((batch_size, max_length), device=p.device, dtype=torch.long)
        return [
            (
                torch.empty(shape, device=p.device, dtype=dtype or p.dtype),
                torch.empty(shape, device=p.device, dtype=dtype or p.dtype),
                key_pos,
            )
            for _ in self.blocks
        ]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MTPHead(nn.Module):
    """Pretraining-only multi-token prediction: one Peri-LN block reads [trunk state at t;
    embedding of token t+1] and predicts token t+2 through the model's own ln_f/head. It is not
    part of the returned model."""

    def __init__(self, c: RecipeConfig) -> None:
        super().__init__()
        self.ln_h, self.ln_e = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        self.proj = nn.Linear(2 * c.d_model, c.d_model, bias=False)
        self.block = PeriLNBlock(c)


def masked_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), reduction="none"
    )
    return (ce * mask.reshape(-1)).sum() / mask.sum().clamp(min=1)


def mtp_loss(
    model: RecipeModel, mtp: MTPHead, x: torch.Tensor, m: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """(next-token CE + MTP_WEIGHT * token-(t+2) CE, next-token CE), both on proof tokens only."""
    h, bias = model.features(x[:, :-1])
    main = masked_ce(model.head(model.ln_f(h)), x[:, 1:], m[:, 1:])
    # Position t gets h_t and emb(x_{t+1}); causal attention keeps x_{t+2}'s prediction a
    # function of x_{<=t+1}.
    z = mtp.proj(torch.cat((mtp.ln_h(h[:, :-1]), mtp.ln_e(model.emb(x[:, 1:-1]))), -1))
    z = mtp.block(z, bias[..., :-1, :-1])
    aux = masked_ce(model.head(model.ln_f(z)), x[:, 2:], m[:, 2:])
    return main + MTP_WEIGHT * aux, main


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
def polar_express(update: torch.Tensor) -> torch.Tensor:
    """Approximately U V^T for update = U S V^T (every singular value pushed to ~1), batched over
    leading dims. Each iteration maps X -> a X + b (X X^T) X + c (X X^T)^2 X, which applies the
    odd quintic s -> a s + b s^3 + c s^5 to every singular value s. After scaling by the Frobenius
    norm all s lie in (0, 1]; the five (a, b, c) are chosen so the composition sends that interval
    near 1."""
    transpose = update.size(-2) > update.size(-1)
    if transpose:  # X X^T is then the smaller Gram matrix
        update = update.mT
    update = update / (update.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)
    for a, b, c in NS_COEFFS:
        gram = update @ update.mT
        update = a * update + (b * gram + c * gram @ gram) @ update
    return update.mT if transpose else update


@torch.no_grad()
def muon_step(params: Sequence[torch.Tensor], buffers: Sequence[torch.Tensor], lr: float) -> None:
    """Nesterov momentum, then a bf16 ``polar_express`` of the update, on hidden matrices only."""
    for p, momentum in zip(params, buffers, strict=True):
        if p.grad is None:
            continue
        momentum.lerp_(p.grad, 1 - MUON_MOMENTUM)
        update = p.grad.lerp(momentum, MUON_MOMENTUM).bfloat16()
        if update.size(0) == 3 * update.size(1):  # QKV: orthogonalize q, k, v separately
            update = update.reshape(3, update.size(1), update.size(1))
        update = polar_express(update)
        p.mul_(1 - lr * MUON_WD)
        p.sub_(lr * update.reshape_as(p).to(p.dtype))


def to_device(
    tok: LeanSeqTokenizer, records: Sequence[dict[str, str] | str], device: torch.device
) -> list[torch.Tensor]:
    """All records padded once: (N, T) ids, proof mask, largest name index used, length."""
    seqs = [tok.encode_record(r) for r in records]
    ids = torch.full((len(seqs), max(map(len, seqs))), tok.pad_id)
    mask = torch.zeros(ids.shape, dtype=torch.bool)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, s.index(tok.prf_id) + 1 : len(s)] = True
    names = (ids >= tok.first_name) & mask
    top_name = torch.where(names, ids - tok.first_name + 1, 0).amax(1)
    return [t.to(device) for t in (ids, mask, top_name, (ids != tok.pad_id).sum(1))]


def token_batches(
    lens: torch.Tensor, gen: torch.Generator, rng: random.Random
) -> tuple[torch.Tensor, list[tuple[int, int, int]]]:
    """One epoch: a random permutation stably sorted by length (equal lengths stay shuffled), cut
    greedily into runs with count * max_len <= TOK_BUDGET, run order shuffled. Returns (order,
    [(start, end, max_len)])."""
    perm = torch.randperm(len(lens), device=lens.device, generator=gen)
    order = perm[torch.argsort(lens[perm], stable=True)]
    sorted_lens = lens[order].cpu()
    cuts, s = [], 0
    while s < len(sorted_lens):
        window = sorted_lens[s : s + TOK_BUDGET // int(sorted_lens[s])]
        # count * max_len is non-decreasing along the window, so the fits form a prefix.
        e = s + int((torch.arange(1, len(window) + 1) * window <= TOK_BUDGET).sum())
        cuts.append((s, e, int(sorted_lens[e - 1])))
        s = e
    rng.shuffle(cuts)
    return order, cuts


def get_batch(
    tok: LeanSeqTokenizer,
    data: list[torch.Tensor],
    idxs: torch.Tensor,
    T: int,
    gen: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows idxs cut to length T. With gen, ``tok.shift_names`` vectorised: one random offset
    per row added to every name in the proof."""
    ids, mask, top_name, _ = data
    x, m = ids[idxs, :T], mask[idxs, :T]
    if gen is not None:
        room = MAX_NAMES - top_name[idxs]
        shift = (torch.rand(len(idxs), device=x.device, generator=gen) * (room + 1)).long()
        x = torch.where((x >= tok.first_name) & m, x + shift.clamp(max=room)[:, None], x)
    return x, m


@torch.no_grad()
def val_loss(
    model: RecipeModel, tok: LeanSeqTokenizer, data: list[torch.Tensor], autocast: Any
) -> float:
    """Mean next-token loss on proof tokens, no name shift."""
    model.eval()
    total, lens = 0.0, data[3]
    for i in range(0, len(lens), VAL_BS):
        idxs = torch.arange(i, min(i + VAL_BS, len(lens)), device=lens.device)
        x, m = get_batch(tok, data, idxs, int(lens[idxs].max()))
        with autocast:
            total += masked_ce(model(x[:, :-1]), x[:, 1:], m[:, 1:]).item() * len(idxs)
    model.train()
    return total / len(lens)


def pretrain(
    tok: LeanSeqTokenizer,
    records: Sequence[dict[str, str] | str],
    budget_secs: float,
    *,
    config: RecipeConfig | None = None,
    val_records: Sequence[dict[str, str] | str] = (),
    seed: int = 0,
    device: str | torch.device | None = None,
    log: Callable[[str], None] = print,
    curve_every: float = CURVE_EVERY,
) -> tuple[RecipeModel, dict[str, Any]]:
    """Train a ``RecipeModel`` from scratch for ``budget_secs`` of wall-clock (setup included; the
    cosine schedule runs over wall-clock, so the budget is the schedule). ``records`` are the
    pipeline's formats (``{"prompt", "proof"}`` or ``"THM ... PRF <proof>"``). Returns the model
    in eval mode and stats with a ``curve`` of {secs, step, train_loss, val_loss} every
    ``curve_every`` seconds (val_loss only when ``val_records`` is given)."""
    t0 = time.time()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    config = config or RecipeConfig(vocab_size=len(tok))
    data = to_device(tok, records, device)
    val = to_device(tok, val_records, device) if val_records else None
    model = RecipeModel(config).to(device)
    mtp = MTPHead(config).to(device)
    fork_init(mtp)
    for p in [*model.blocks.parameters(), *mtp.parameters()]:
        if p.ndim == 2:  # orthogonal, with the N(0, 0.02) element second moment
            nn.init.orthogonal_(p, gain=0.02 * math.sqrt(max(p.shape)))
    log(f"pretrain: {model.n_params():,} params, {len(records):,} records on {device}")
    trainable = [*model.parameters(), *mtp.parameters()]
    matrices: list[torch.Tensor] = [
        p for p in [*model.blocks.parameters(), *mtp.parameters()] if p.ndim == 2
    ]
    matrices.append(model.head.weight)
    in_muon = {id(p) for p in matrices}
    opt = torch.optim.AdamW(
        [p for p in trainable if id(p) not in in_muon],
        lr=LR,
        weight_decay=WD,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )
    buffers = [torch.zeros_like(p) for p in matrices]
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")
    gen = torch.Generator(device=device).manual_seed(seed)
    rng, cuts = random.Random(seed), []
    order = torch.empty(0, dtype=torch.long, device=device)
    model.train()
    step, records_seen, curve, recent, next_point = 0, 0, [], [], curve_every
    while (frac := (time.time() - t0) / budget_secs) < 1:
        step += 1
        lr = LR * (MIN_LR_FRAC + (1 - MIN_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = lr
        if not cuts:  # new epoch
            order, cuts = token_batches(data[3], gen, rng)
        s, e, T = cuts.pop()
        records_seen += e - s
        x, m = get_batch(tok, data, order[s:e], T, gen)
        with autocast:
            loss, main = mtp_loss(model, mtp, x, m)
        model.zero_grad(set_to_none=True)
        mtp.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        muon_step(matrices, buffers, MUON_LR * lr / LR)
        if step % 20 == 0:
            recent.append(main.item())
        if time.time() - t0 >= next_point:
            point = {
                "secs": round(time.time() - t0),
                "step": step,
                "train_loss": sum(recent) / len(recent) if recent else None,
                "val_loss": val_loss(model, tok, val, autocast) if val is not None else None,
            }
            curve.append(point)
            log(f"pretrain step {step} train {point['train_loss']} val {point['val_loss']}")
            recent, next_point = [], next_point + curve_every
    model.eval()
    return model, {"steps": step, "records_seen": records_seen, "curve": curve}


def load_checkpoint(
    path: str | Path, device: str = "cpu"
) -> tuple[RecipeModel, LeanSeqTokenizer, dict[str, Any]]:
    """A ``RecipeModel`` saved with ``nd_model.save_checkpoint(path, model, tok)``."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    tok = LeanSeqTokenizer.from_state_dict(ckpt["tokenizer"])
    model = RecipeModel(RecipeConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"], strict=True)
    return model.to(device).eval(), tok, ckpt.get("metadata", {})


def load_autoresearch_checkpoint(path: str | Path, device: str = "cpu") -> RecipeModel:
    """An autoresearch-loop ``final.pt`` / ``pretrain.pt`` from a run of this architecture at the
    default size (e.g. 198 on Hugging Face). The loop's token ids are ``LeanSeqTokenizer``'s."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("tok_mode") != "lean_seq":
        raise ValueError(f"not a lean_seq checkpoint: tok_mode {ckpt.get('tok_mode')!r}")
    model = RecipeModel()
    model.load_state_dict(ckpt["state"], strict=True)
    return model.to(device).eval()
