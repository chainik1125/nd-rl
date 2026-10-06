"""Stage-1 training of `lean_thinkn` models with Dan's best recipe (fork `state_train.py --recipe
best`).

    python -m code.thinking.train --data pairs.pkl --heldout heldout.jsonl --out DIR --seed 0 \
        --best_steps 24000
    python -m code.thinking.train ... --init stage1_best12_s0_b1200.pt --lr_scale 0.3   # cold start

From scratch with the default loss (`--think_weight 1`) this calls the fork's
`state_train_best.train` unchanged: the `(state, think + action + <eos>)` pairs are exactly its
input format and its name shift reads `tok.ref0` (50). `train_best` below is a copy of that function
for the two things it cannot do -- start from a given model (cold start: `ckpt.cold_start` expands a
`lean_staten` checkpoint to 114 tokens; the MTP head starts fresh as in Stage 1) and weight think
tokens (`--think_weight w`: every token of the think block, tags included, weighs w in the loss,
action + <eos> weigh 1; 0 trains the action only).  With neither, the copy reproduces the fork's run
bit for bit (test_train_copy_matches_fork).

Outputs in DIR: model.pt (fork checkpoint format, mode lean_thinkn), train.json (curve, losses),
args.json, registry/. ND_OFFLINE=1 is forced (`ckpt.run_env`): nothing is uploaded.  Schedule:
`--best_steps N` (step cosine; ~24,000 on an RTX 3090 for the 1,200 A40-s run), else `--budget_secs`
(wall clock).

**Formats R and M** (`rootplan`; the pickle's `think_format`, recorded in the checkpoint's extra so
sampling and eval dispatch on it).  M (`thinking-pairs/1`, "memo") is a pair layout: the prompt
`state <think> MEMO_prev </think> <think>` is the fork's state, the target `MEMO </think> action
<eos>` its action, so it trains through the same code as the per-step format.  R
(`thinking-seq/1`, "root") is one sequence per proof: `SeqData` keeps the flattened ids and a label
per token (0 state, 1 action + <eos>, 2 the think block with its tags; `rootplan.flatten` /
`think_mask`), the loss mask is labels > 0 (think tokens weighted `--think_weight`), one name
offset per sequence, and the recipe's token-budget batches cut by sequence length.  Steps,
`--tok_budget` and the optimiser are the recipe's.  The model's `max_len` (`--max_len`, 0: the
longest train / held-out example rounded up to a multiple of 512, at least 1,024; R: 4,608) is
checked against every example (an example over it is an error, never truncated).  ALiBiGPT has no
position table (ALiBi heads read `pos_q - pos_k`, NoPE heads nothing), so `max_len` is the context
the model was trained to, which the samplers enforce.  Held-out validation uses the format's own
empty layout (`rootplan.encode(plain_record(...))`), not `heldout_records`' per-step one.

**State format** (`stateview`: `flat`, or `goalstack` with every enclosing frame's goal shown):
the pickle's `state_format` (absent: flat) becomes the process's before the held-out set is
rendered, so validation states match the training pairs, and is recorded in the checkpoint's extra
(`ckpt.load_ckpt` sets it back for sampling).
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from . import stateview
from .ckpt import cold_start, run_env, save_ckpt
from .dan import bare, load
from .data import FORMAT as PAIR_FORMAT
from .data import Pair, heldout_records, read_jsonl, to_fork
from .tok import ThinkTokenizer

D = load()
STB: Any = D.state_train_best
BM: Any = D.best_model
SEQ_FORMAT = "thinking-seq/1"  # rootplan.SEQ_FORMAT
RM_FORMATS = ("root", "memo")
Seg = tuple[bytes, bool]
LAB_STATE, LAB_ACTION, LAB_THINK = 0, 1, 2


def think_format(data: dict[str, Any]) -> str:
    """'root' / 'memo' (rootplan's whole-proof formats), else 'think' if the model learns to write
    `<think> ... </think>` per step (traces or empty blocks), else 'none' (T0)."""
    if data.get("think_format") in RM_FORMATS:
        return str(data["think_format"])
    return "think" if data["has_traces"] or data["no_trace_mode"] == "empty" else "none"


def load_data(path: str | Path, tok: Any | None = None) -> dict[str, Any]:
    """`data.load_pairs`, plus the R pickle (`thinking-seq/1`: per proof a list of `(ids bytes,
    is_loss)` segments)."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    assert data["format"] in (PAIR_FORMAT, SEQ_FORMAT), data.get("format")
    assert (data["format"] == SEQ_FORMAT) == (data.get("think_format") == "root"), (
        data["format"],
        data.get("think_format"),
    )
    if tok is not None:
        assert list(tok.itos) == data["itos"], "data were built for another vocabulary"
    return data


def heldout_rm(path: str | Path, tok: Any, fmt: str, limit: int = 2000) -> list[Any]:
    """Validation examples of format R / M in the format's empty layout (first `limit` records
    that decompose): R `state_0 <think> </think> action_0 <eos> state_1 ...`, M pairs with every
    memo empty."""
    from . import rootplan

    out: list[Any] = []
    for i, rec in enumerate(read_jsonl(path)):
        try:
            r = rootplan.plain_record(rec["prompt"], rec["proof"], fmt, f"heldout:{i}")
            out.append(rootplan.encode(r, tok, fmt))
        except (D.lean_tok.ParseFail, ValueError, AssertionError, IndexError, KeyError):
            continue
        if len(out) >= limit:
            break
    return out


def example_lens(records: Sequence[Any], fmt: str) -> list[int]:
    """Tokens per training example: per R sequence, else per pair."""
    if fmt == "root":
        return [sum(len(b) for b, _ in segs) for segs in records]
    return [len(p) + len(q) for rec in records for p, q, _ in rec]


def auto_max_len(n: int) -> int:
    """The longest example rounded up to a multiple of 512, at least the fork's 1,024."""
    return max(1024, -(-n // 512) * 512)


def think_lens_tensor(records: list[list[Pair]]) -> torch.Tensor:
    return torch.tensor([t for rec in records for _, _, t in rec], dtype=torch.long)


def make_batch_w(
    ids: torch.Tensor,
    plen: torch.Tensor,
    tlen: torch.Tensor,
    lens: torch.Tensor,
    idxs: torch.Tensor,
    T: int,
    tok: Any,
    gen: Any,
    w: float,
    shift: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The fork's make_batch, then a float loss mask: think block w, action + <eos> 1, state 0."""
    x, m = STB.make_batch(ids, plen, lens, idxs, T, tok, gen, shift=shift)
    ar = torch.arange(T, device=x.device)[None]
    think = (ar >= plen[idxs, None]) & (ar < (plen + tlen)[idxs, None])
    return x, torch.where(think, torch.full_like(m, w, dtype=torch.float32), m.float())


def val_loss_w(model: Any, v: Any, tlen: torch.Tensor, tok: Any, w: float, bs: int = 256) -> float:
    """Held-out CE with think tokens weighted w (w = 0: action tokens only, comparable across
    think formats)."""
    ids, plen, lens_cpu, lens = v
    tlen = tlen.to(ids.device)
    model.eval()
    tot = n = 0.0
    cuda = ids.is_cuda
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
        for i in range(0, len(lens_cpu), bs):
            idxs = torch.arange(i, min(i + bs, len(lens_cpu)), device=ids.device)
            T = int(lens_cpu[i : i + bs].max())
            x, m = make_batch_w(ids, plen, tlen, lens, idxs, T, tok, None, w, shift=False)
            k = float(m[:, 1:].sum())
            main = BM.mtp_loss(model, None, x, m, weight=0)[1]
            tot += float(main) * k
            n += k
    model.train()
    return tot / n


class PairData:
    """The fork's padded pair tensors (`state_train_best.to_device`) and the think-block lengths."""

    kind = "pairs"

    def __init__(self, records: list[list[Pair]], dev: str) -> None:
        self.v = STB.to_device(to_fork(records), dev)
        self.ids, self.plen, self.lens_cpu, self.lens = self.v
        self.tlen = think_lens_tensor(records).to(dev)  # in to_device's pair order
        self.T = int(self.ids.shape[1])

    def batch(
        self, idxs: torch.Tensor, T: int, tok: Any, gen: Any, w: float, shift: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return make_batch_w(self.ids, self.plen, self.tlen, self.lens, idxs, T, tok, gen, w, shift)

    def val_loss(self, model: Any, tok: Any, autocast: Any, w: float = 1.0) -> float:
        if w == 1.0:
            return float(STB.val_loss(model, self.v, tok, autocast))  # the fork's, unchanged
        return val_loss_w(model, self.v, self.tlen, tok, w)


class SeqData:
    """Format R: whole-proof sequences, flattened on the device (ids, a label per token: 0 state, 1
    action + <eos>, 2 the think block with its tags; offsets, lengths)."""

    kind = "sequences"

    def __init__(self, seqs: Sequence[Sequence[Seg]], tok: Any, dev: str) -> None:
        ids, lab = bytearray(), bytearray()
        lens = []
        think0 = bytes([tok.think])
        for segs in seqs:
            n = 0
            for b, loss in segs:
                ids += b
                k = LAB_STATE if not loss else LAB_THINK if b[:1] == think0 else LAB_ACTION
                lab += bytes([k]) * len(b)
                n += len(b)
            lens.append(n)
        assert lens and min(lens) > 0, "empty sequence"
        self.ids = torch.frombuffer(ids, dtype=torch.uint8).to(dev)
        self.lab = torch.frombuffer(lab, dtype=torch.uint8).to(dev)
        self.lens_cpu = torch.tensor(lens, dtype=torch.long)
        self.lens = self.lens_cpu.to(dev)
        self.offs = (self.lens_cpu.cumsum(0) - self.lens_cpu).to(dev)
        self.T = int(self.lens_cpu.max())

    def gather(self, idxs: torch.Tensor, T: int) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (ids (B, T) right-padded with <pad> = 0, labels (B, T), 0 on padding)."""
        ar = torch.arange(T, device=self.ids.device)[None]
        valid = ar < self.lens[idxs, None]
        at = torch.where(valid, self.offs[idxs, None] + ar, torch.zeros_like(ar))
        x = torch.where(valid, self.ids[at].long(), torch.zeros_like(at))
        lab = torch.where(valid, self.lab[at].long(), torch.zeros_like(at))
        return x, lab

    def batch(
        self, idxs: torch.Tensor, T: int, tok: Any, gen: Any, w: float, shift: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(x, float loss mask): the think block weighs w, actions + <eos> 1, states 0.  With
        `shift`, one random name offset per sequence (the fork's `make_batch`, run on the
        gathered rows)."""
        x, lab = self.gather(idxs, T)
        if shift:
            rows = torch.arange(len(idxs), device=x.device)
            x, _ = STB.make_batch(x, torch.zeros_like(rows), self.lens[idxs], rows, T, tok, gen)
        m = torch.where(
            lab == LAB_THINK,
            torch.full(lab.shape, w, dtype=torch.float32, device=x.device),
            (lab == LAB_ACTION).float(),
        )
        return x, m

    def val_loss(
        self, model: Any, tok: Any, autocast: Any, w: float = 1.0, max_tokens: int = 2**17
    ) -> float:
        """Held-out CE over the loss tokens (think weighted w), length-sorted batches of at most
        max_tokens padded tokens, no name offset."""
        order = torch.argsort(self.lens_cpu, stable=True)
        model.eval()
        tot = n = 0.0
        s = 0
        with torch.no_grad(), autocast:
            while s < len(order):
                e = s + 1
                while e < len(order) and (e + 1 - s) * int(self.lens_cpu[order[e]]) <= max_tokens:
                    e += 1
                idxs = order[s:e].to(self.ids.device)
                x, m = self.batch(idxs, int(self.lens_cpu[order[e - 1]]), tok, None, w, False)
                k = float(m[:, 1:].sum())
                if k > 0:
                    tot += float(BM.mtp_loss(model, None, x, m, weight=0)[1]) * k
                    n += k
                s = e
        model.train()
        return tot / max(n, 1e-9)


def make_data(records: list[Any], fmt: str, tok: Any, dev: str) -> PairData | SeqData:
    return SeqData(records, tok, dev) if fmt == "root" else PairData(records, dev)


def train_best(
    a: Any,
    tok: Any,
    records: list[Any],
    held: list[Any] | None,
    dev: str,
    record: Any,
    init_model: Any = None,
    force_copy: bool = False,
    fmt: str = "think",
) -> tuple[Any, dict[str, Any]]:
    """state_train_best.train, or its copy (state_train_best.py:90-199@4dcd0e4f) with the changes
    marked `# THINK`.  `fmt` 'root': `records` are R sequences (always the copy)."""
    max_len = int(getattr(a, "max_len", 0) or 1024)
    if (
        fmt != "root"
        and init_model is None
        and a.think_weight == 1.0
        and a.lr_scale == 1.0
        and max_len == 1024
        and not force_copy
    ):
        return STB.train(a, tok, to_fork(records), to_fork(held) if held else None, dev, record)
    cuda = dev == "cuda"
    t_data = time.time()
    tr = make_data(records, fmt, tok, dev)  # THINK: pairs or R sequences
    lens_cpu = tr.lens_cpu
    vd = make_data(held, fmt, tok, dev) if held else None
    print(
        f"best: {len(lens_cpu):,} train {tr.kind} (T {tr.T}), "
        f"{0 if not vd else len(vd.lens_cpu):,} val {tr.kind}, max_len {max_len}, "
        f"on {dev} in {time.time() - t_data:.0f}s",
        flush=True,
    )
    longest = max(tr.T, vd.T if vd else 0)  # THINK: an example over max_len is never truncated
    assert longest <= max_len, f"an example of {longest} tokens exceeds max_len {max_len}"
    t0 = time.time()
    torch.manual_seed(a.seed)
    gen = torch.Generator(device=dev)
    gen.manual_seed(a.seed)
    gen_cpu = torch.Generator()
    gen_cpu.manual_seed(a.seed)
    rng = random.Random(a.seed)
    if init_model is None:
        model = BM.ALiBiGPT(
            tok.vocab_size, a.n_layer, a.d, a.n_head, max_len=max_len, d_ff=a.d_ff
        ).to(dev)  # THINK: max_len (a cfg entry: no parameter depends on it)
    else:  # THINK: cold start -- the given (expanded) model; only the MTP head is new
        model = init_model.to(dev)
        model.train()
        model.cfg["max_len"] = max_len
    mtp = BM.MTPHead(a.d, a.n_head, a.d_ff)
    mtp.apply(model._init)
    mtp = mtp.to(dev)
    compiled = []
    if cuda and not a.no_compile:
        torch._inductor.config.triton.coalesce_tiling_analysis = False  # type: ignore[attr-defined]
        compiled = list(model.blocks) + [mtp.block]
        for b in compiled:
            b.compile(dynamic=True)
    if init_model is None:
        BM.orthogonal_init(list(model.blocks.parameters()) + list(mtp.parameters()))
    else:  # THINK: keep the trunk
        BM.orthogonal_init(list(mtp.parameters()))
    print(
        f"best: {model.n_params():,} params "
        f"(+{sum(p.numel() for p in mtp.parameters()):,} MTP, not saved)",
        flush=True,
    )
    trainable = list(model.parameters()) + list(mtp.parameters())
    matrices = [p for p in model.blocks.parameters() if p.ndim == 2] + [model.head.weight]
    matrices += [p for p in mtp.parameters() if p.ndim == 2]
    mids = {id(p) for p in matrices}
    aux_params = [p for p in trainable if id(p) not in mids]
    bufs = [torch.zeros_like(p) for p in matrices]
    muon_wd = 10 * a.wd * a.lr / BM.MUON_LR
    peak = (
        a.lr * a.lr_scale
    )  # THINK: --lr_scale scales the AdamW and Muon peaks together (1: the recipe)
    opt = torch.optim.AdamW(aux_params, lr=peak, weight_decay=a.wd, betas=(0.9, 0.95), fused=cuda)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda)
    muon_lr = torch.zeros((), device=dev)
    muon_graph = None
    model.train()
    step, seen, curve, recent, recent_aux = 0, 0, [], [], []
    tokens_seen = 0  # THINK: real (unpadded) tokens, for throughput
    next_point = a.curve_every
    cuts, order, epochs_cuts = [], None, None
    main = aux = torch.zeros(())
    lr = peak
    while True:
        el = time.time() - t0
        frac = step / a.best_steps if a.best_steps else el / a.budget_secs
        if frac >= 1.0:
            break
        step += 1
        lr = (
            peak * step / STB.WARMUP
            if step < STB.WARMUP
            else peak * STB.MIN_LR_FRAC
            + 0.5 * peak * (1 - STB.MIN_LR_FRAC) * (1 + math.cos(math.pi * frac))
        )
        for g in opt.param_groups:
            g["lr"] = lr
        if not cuts:
            order, cuts = STB.token_batches(lens_cpu, gen_cpu, rng, a.tok_budget)
            epochs_cuts = len(cuts)
        s, e, T, ntok = cuts.pop()
        assert order is not None
        idxs = order[s:e].to(dev, non_blocking=True)
        seen += e - s
        record.count(train_steps=1, train_tokens=ntok)
        tokens_seen += ntok
        x, m = tr.batch(idxs, T, tok, gen, a.think_weight)  # THINK: weighted mask, pairs or R
        with autocast:
            loss, main, aux = BM.mtp_loss(model, mtp, x, m, weight=a.mtp)
        model.zero_grad(set_to_none=False)
        mtp.zero_grad(set_to_none=False)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        muon_lr.fill_(
            BM.MUON_LR * a.lr_scale * lr / peak
        )  # THINK: lr_scale (1: BM.MUON_LR * lr / a.lr)
        if not cuda:
            BM.muon_step(matrices, bufs, muon_lr, a.d, muon_wd)
        else:
            if (
                muon_graph is None
            ):  # capture once: warm-up, capture, then undo both updates (the fork's)
                state = matrices + bufs
                saved = [t.clone() for t in state]
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    BM.muon_step(matrices, bufs, muon_lr, a.d, muon_wd)
                torch.cuda.current_stream().wait_stream(side)
                muon_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(muon_graph):
                    BM.muon_step(matrices, bufs, muon_lr, a.d, muon_wd)
                with torch.no_grad():
                    for t, v in zip(state, saved, strict=True):
                        t.copy_(v)
            muon_graph.replay()
        if step % 20 == 0 or not cuda:
            recent.append(float(main))
            recent_aux.append(float(aux))
        el = time.time() - t0
        if el >= next_point or (a.best_steps and step == a.best_steps):
            if not recent:
                recent.append(float(main))
                recent_aux.append(float(aux))
            pt = {
                "secs": round(el, 1),
                "step": step,
                "train_loss": sum(recent) / max(1, len(recent)),
                "mtp_loss": sum(recent_aux) / max(1, len(recent_aux)),
                "epochs": seen / len(lens_cpu),
            }
            if vd:
                pt["val_loss"] = vd.val_loss(model, tok, autocast)
            curve.append(pt)
            print(
                f"step {step} loss {pt['train_loss']:.4f} mtp {pt['mtp_loss']:.4f} lr {lr:.2e} "
                f"epochs {pt['epochs']:.2f} {el:.0f}s"
                + (f" val {pt['val_loss']:.4f}" if vd else ""),
                flush=True,
            )
            recent, recent_aux = [], []
            next_point += a.curve_every
    secs = time.time() - t0
    for b in compiled:
        b._compiled_call_impl = None
    final_val = vd.val_loss(model, tok, autocast) if vd else None
    print(
        f"best: done {step} steps, {seen / len(lens_cpu):.2f} epochs of {tr.kind}, {secs:.0f}s "
        f"({step / max(secs, 1e-9):.2f} steps/s), final val {final_val}",
        flush=True,
    )
    model.eval()
    extra = {
        "recipe": "best",
        "steps": step,
        "secs": secs,
        "pairs_seen": seen,
        "epochs": seen / len(lens_cpu),
        "batches_per_epoch": epochs_cuts,
        "tok_budget": a.tok_budget,
        "curve": curve,
        "final_val": final_val,
        "last_train_loss": curve[-1]["train_loss"] if curve else float(main),
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if cuda else None,
        "max_len": max_len,
        "examples": tr.kind,
        "tokens_seen": tokens_seen,
        "steps_per_s": step / max(secs, 1e-9),
    }
    return model, extra


def batch_w(
    records: list[list[Pair]], idxs: list[int], tok: Any, rng: random.Random, dev: str, w: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """The fork's state_train.batch (all pairs of the drawn records, one name offset per pair),
    then a float mask with think tokens weighted w."""
    x, m = D.state_train.batch(to_fork(records), idxs, tok, rng, dev)
    if w == 1.0:
        return x, m
    rows = [(len(p), t) for i in idxs for p, _, t in records[i]]
    plen = torch.tensor([r[0] for r in rows], device=x.device)[:, None]
    tl = torch.tensor([r[1] for r in rows], device=x.device)[:, None]
    ar = torch.arange(x.shape[1], device=x.device)[None]
    think = (ar >= plen) & (ar < plen + tl)
    return x, torch.where(think, torch.full_like(m, w, dtype=torch.float32), m.float())


def finetune(
    model: Any,
    tok: Any,
    records: list[list[Pair]],
    steps: int = 600,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    warmup: int = 50,
    wd: float = 0.1,
    recs: int = 128,
    seed: int = 0,
    dev: str = "cpu",
    think_weight: float = 1.0,
    max_tokens: int = 200000,
    log_every: int = 200,
) -> tuple[Any, float]:
    """The fork's plain AdamW loop (state_train.py:107-108,138-185@4dcd0e4f, the expert-iteration
    fine-tune: the ladder calls it with --steps 600 --lr 3e-4 --min_lr 3e-5 --warmup 50 --recs 128)
    on in-memory pairs, with the think weight.  -> (model, last loss)."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95))

    def sched(s: int) -> float:
        if s < warmup:
            return lr * s / warmup
        return min_lr + 0.5 * (lr - min_lr) * (
            1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup))
        )

    model.train()
    t0 = time.time()
    perm: list[int] = []
    loss_sum = 0.0
    for step in range(1, steps + 1):
        if len(perm) < recs:
            perm = list(range(len(records)))
            rng.shuffle(perm)
        idxs = [perm.pop() for _ in range(min(recs, len(perm)))]
        x, m = batch_w(records, idxs, tok, rng, dev, think_weight)
        D.record.count(
            train_steps=1, train_tokens=sum(len(p) + len(q) for i in idxs for p, q, _ in records[i])
        )
        for g in opt.param_groups:
            g["lr"] = sched(step)
        opt.zero_grad(set_to_none=True)
        per = max(1, max_tokens // x.shape[1])
        ntok = m[:, 1:].sum()
        loss_sum = 0.0
        for s0 in range(0, x.shape[0], per):
            xb, mb = x[s0 : s0 + per], m[s0 : s0 + per]
            kb = mb[:, 1:].sum()
            if float(kb) == 0:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                lb = D.state_train.loss_on(model, xb, mb) * (kb / ntok)
            lb.backward()
            loss_sum += float(lb)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == steps:
            print(
                f"ft step {step} loss {loss_sum:.4f} lr {sched(step):.2e} {time.time() - t0:.0f}s",
                flush=True,
            )
    model.eval()
    return model, loss_sum


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--data", required=True, help="pairs (code.thinking.data) or an R / M pickle (rootplan)"
    )
    ap.add_argument(
        "--heldout", default=None, help="K12-style jsonl; first 2,000 records, no trace"
    )
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--best_steps", type=int, default=0, help="step-based cosine schedule (0: --budget_secs)"
    )
    ap.add_argument("--budget_secs", type=float, default=1200)
    ap.add_argument("--init", default=None, help="cold start from this lean_staten checkpoint")
    ap.add_argument("--init_seed", type=int, default=0, help="seed of the new embedding rows")
    ap.add_argument(
        "--think_weight", type=float, default=1.0, help="loss weight of think-block tokens"
    )
    ap.add_argument("--lr_scale", type=float, default=1.0, help="scale both peak learning rates")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--tok_budget", type=int, default=128 * 144)
    ap.add_argument("--mtp", type=float, default=0.3)
    ap.add_argument("--curve_every", type=float, default=30)
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument(
        "--best_dims",
        default=None,
        help='"layers,d,heads,d_ff" (tests; the recipe is 6,384,8,1280)',
    )
    ap.add_argument(
        "--max_len",
        type=int,
        default=0,
        help="the model's context (0: longest example rounded up to 512, at least 1,024)",
    )
    ap.add_argument("--device", default=None)
    return ap


def run(a: argparse.Namespace) -> dict[str, Any]:
    out = Path(a.out)
    run_env(out)
    record = D.record
    init_model, init_info = None, {}
    if a.init:
        init_model, _, init_info = cold_start(a.init, "cpu", seed=a.init_seed)
        cfg = init_model.cfg
        a.n_layer, a.d, a.n_head, a.d_ff = cfg["n_layer"], cfg["d"], cfg["n_head"], cfg["d_ff"]
    elif a.best_dims:
        a.n_layer, a.d, a.n_head, a.d_ff = (int(x) for x in a.best_dims.split(","))
    else:
        a.n_layer, a.d, a.n_head, a.d_ff = BM.N_LAYER, BM.D, BM.N_HEAD, BM.D_FF
    tok = ThinkTokenizer()
    data = load_data(a.data, tok)
    fmt = think_format(data)
    # the pickle's state format (flat / goalstack) for the held-out states and the checkpoint
    sfmt = stateview.set_state_format(stateview.data_format(data))
    records = data["records"]
    held = None
    if a.heldout:  # the format's own layout (R / M: every think empty), in the data's state format
        held = (
            heldout_rm(a.heldout, tok, fmt)
            if fmt in RM_FORMATS
            else heldout_records(a.heldout, tok, data["no_trace_mode"])
        )
    longest = max(example_lens(records, fmt) + (example_lens(held, fmt) if held else []))
    if not a.max_len:
        a.max_len = auto_max_len(longest)
    elif longest > a.max_len:
        raise SystemExit(f"error: an example of {longest} tokens exceeds --max_len {a.max_len}")
    with bare():
        record.save_config(vars(a), str(out) + "/", role="coldstart" if a.init else "stage1")
    torch.manual_seed(a.seed)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    nex = len(records) if fmt == "root" else sum(len(r) for r in records)
    unit = "sequences" if fmt == "root" else "pairs"
    print(
        f"mode {tok.mode}, {len(records)} records, {nex} {unit} (longest {longest} tokens, "
        f"max_len {a.max_len}), format {fmt}, state format {sfmt}",
        flush=True,
    )
    model, extra = train_best(a, tok, records, held, dev, record, init_model=init_model, fmt=fmt)
    if held:
        autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda")
        extra["final_val_action"] = make_data(held, fmt, tok, dev).val_loss(
            model, tok, autocast, w=0.0
        )
    secs = extra["secs"]
    extra.setdefault("max_len", model.cfg["max_len"])
    extra.setdefault("examples", unit)
    extra.setdefault("steps_per_s", extra["steps"] / max(secs, 1e-9))
    extra.update(
        args=vars(a),
        n_params=model.n_params(),
        pairs=nex,
        records=len(records),
        data=str(a.data),
        think_format=fmt,
        state_format=sfmt,
        no_trace_mode=data["no_trace_mode"],
        has_traces=data["has_traces"],
        longest_example=longest,
        **init_info,
    )
    assert model.cfg["max_len"] == a.max_len, (model.cfg["max_len"], a.max_len)
    ck = out / "model.pt"
    save_ckpt(ck, model, tok.mode, extra=extra)
    lab = dict(ckpt=str(ck), data=str(a.data), source=str(ck), n_params=model.n_params(), pairs=nex)
    with bare():
        record.record("train_loss", extra["last_train_loss"], recipe="best", **lab)
        if held:
            record.record("val_loss", extra["final_val"], n=len(held), recipe="best", **lab)
    (out / "train.json").write_text(
        json.dumps({k: v for k, v in extra.items()}, indent=1, default=str)
    )
    print("saved", ck, flush=True)
    return extra


def main(argv: list[str] | None = None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
