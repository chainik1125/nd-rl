"""The verifier: small networks, separate from the policy, that judge one proof step.

Their input is the "compact" step: the hypothesis lines the step cites, copied from the proof
state, then the step itself:

    <st> n3 : ( P → Q ) <nl> n5 : P <nl> <act> have n9 : Q := n3 n5 ;

Which lines a step cites is read off the step's text (the names in its term; `h<k>` for a premise
line). Whether the step is well typed is the networks' judgement. The training data are steps
the policy sampled during RL on y, saved each round by `fr/rl.py` as `ver_r<round>.pkl` (refused
by the per-step type check, or accepted), and the steps of x proofs. At test time nothing but
these networks screens a step (no Lean, no type check).

Three kinds (`--kind`), and what the final solver uses (PREREG amendment 5):

- `disc`: one logit at the last token, trained on refused (0) and accepted (1) steps. Catches
  gross mismatches, lets near misses through. Used, threshold 0.05.
- `mslot`: the masked slot model, trained on accepted steps only; see the comment above
  `mslot_parts`. Catches near misses, can explain away a gross mismatch. Used, threshold 0.001.
- `slot`: the unmasked slot model; see the comment above `slots`. Precise but refuses valid
  steps on unseen theorems. Not used.

`Combo` is "every part must pass": the final verifier is `mslot` and `disc` together. Also in
this file and not used in the final recipe: several name views per step (`VIEWS`), mining of the
network's own mistakes, and one-symbol edits of accepted steps labelled by the training-time
checker (`near_misses`).

They are separate networks because training this judgement on the policy's own trunk damaged the
policy (PREREG amendment 2).

    python -m fr.verifier --kind disc  --run RUN --out RUN/verifier_disc.pt  --x X --steps 6000 --max_per_round 150000 --x_pos 400000
    python -m fr.verifier --kind mslot --run RUN --out RUN/verifier_mslot.pt --x X --steps 8000 --n_layer 6 --d 384 --x_pos 1200000
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .common import EOS, PAD, TOK, VOCAB, enc, is_name, step_kind
from .data import shift_names
from .model import GPT

ITOS = TOK.itos
VER_CFG = dict(n_layer=4, d=256, n_head=8)


def compact(state, act):
    """state tokens, (effective) action tokens of an atomic or premise `have` -> verifier input
    tokens, or None if the step cites a name the state does not show."""
    if step_kind(act) not in ("atom", "pr"):
        return None
    lines = {}
    cur = []
    for t in state:
        if t == "<st>":
            continue
        if t == "⊢":
            break
        if t == "<nl>":
            if cur:
                lines[cur[0]] = cur
            cur = []
        else:
            cur.append(t)
    j = act.index(":=")
    cited = []
    for t in act[j + 1 : -1]:
        if (is_name(t) or (t[0] == "h" and t[1:].isdigit())) and t not in cited:
            if t not in lines:
                return None
            cited.append(t)
    out = ["<st>"]
    for t in cited:
        out += lines[t] + ["<nl>"]
    return out + ["<act>"] + list(act)


def compact_bytes(state_b, act_b):
    c = compact([ITOS[i] for i in state_b], [ITOS[i] for i in act_b])
    return None if c is None else enc(c)


VIEWS = 1  # verifier passes per step at test time; views beyond the first shift the names


@torch.no_grad()
def ver_probs(model, seqs, tok_budget=400_000, views=None):
    """seqs: list of id sequences (bytes) -> probability that each step is well typed. With
    `views` > 1 the step is judged again under random name offsets (the same statement to the
    network, different name tokens) and the lowest probability is returned."""
    views = VIEWS if views is None else views
    if not seqs:
        return []
    kind = getattr(model, "kind", "disc")
    if kind == "combo":  # every part must pass its own threshold: score >= 0.5 iff all do
        out = [1.0] * len(seqs)
        for m, thr in model.parts:
            for i, p in enumerate(ver_probs(m, seqs, tok_budget, views)):
                out[i] = min(out[i], p / (2.0 * thr))
        return out
    if kind == "slot":
        return slot_probs(model, seqs, tok_budget)
    if kind == "mslot":
        return mslot_probs(model, seqs, tok_budget)
    dev = next(model.parameters()).device
    cuda = dev.type == "cuda"
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out = [0.0] * len(seqs)
    i0 = 0
    while i0 < len(order):
        T = len(seqs[order[min(len(order) - 1, i0 + 8191)]])
        n = max(1, min(8192, tok_budget // max(1, T)))
        chunk = order[i0 : i0 + n]
        i0 += n
        T = max(len(seqs[i]) for i in chunk)
        arr = np.full((len(chunk), T), PAD, np.uint8)
        for r, i in enumerate(chunk):
            arr[r, : len(seqs[i])] = np.frombuffer(bytes(seqs[i]), np.uint8)
        x = torch.from_numpy(arr).to(dev).long()
        last = torch.tensor([len(seqs[i]) - 1 for i in chunk], device=dev)
        rows = torch.arange(len(chunk), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            logit = model.aux(model.trunk(x)[rows, last])[:, 0].float()
            for _ in range(views - 1):
                lg = model.aux(model.trunk(shift_names(x, low_bias=0.0))[rows, last])[:, 0].float()
                logit = torch.minimum(logit, lg)
        for i, p in zip(chunk, torch.sigmoid(logit).tolist()):
            out[i] = p
    return out


class Combo:
    """Several verifier networks used together: a step passes only if every part gives it at
    least that part's threshold. `ver_probs` returns min over parts of p / (2 * threshold), so
    the solver's usual cut at 0.5 means "all parts pass" (and `--tau` scales all thresholds)."""

    kind = "combo"

    def __init__(self, parts):
        self.parts = parts  # [(network, threshold)]

    def parameters(self):
        return self.parts[0][0].parameters()


def _build_one(v, device):
    m = GPT(VOCAB, **v["cfg"]).to(device)
    m.load_state_dict(v["state"])
    m.kind = v.get("kind", "disc")
    m.eval()
    return m


def build_verifier(extra, device):
    """The verifier stored in a checkpoint's `extra['verifier']`, or None."""
    v = (extra or {}).get("verifier")
    if not v:
        return None
    if v.get("kind") == "combo":
        return Combo([(_build_one(p, device), float(p["thr"])) for p in v["parts"]])
    return _build_one(v, device)


def load_verifier_spec(spec):
    """`file.pt` or `a.pt:0.003,b.pt:0.2` (file:threshold parts of a combo) -> the dict stored
    in a checkpoint's `extra['verifier']`."""
    def one(fn):
        v = torch.load(os.path.abspath(fn), map_location="cpu", weights_only=False)
        return {"cfg": v["cfg"], "state": v["state"], "kind": v.get("kind", "disc"), "metrics": v.get("metrics"),
                "rounds": v.get("rounds"), "file": os.path.abspath(fn)}  # fmt: skip

    if "," not in spec and ":" not in spec:
        return one(spec)
    parts = []
    for item in spec.split(","):
        fn, thr = item.rsplit(":", 1)
        parts.append({**one(fn), "thr": float(thr)})
    return {"kind": "combo", "parts": parts}


# ---- the slot verifier: a generative consistency check ----
#
# A step `have n9 : F := term ;` relates a few formulas: the stated F and the formula of each line
# the term cites. The slot model is a small language model trained, on accepted steps only, to
# write any one of these formulas given the rest:
#
#     <st> n5 : P <nl> <act> have n9 : Q := n3 n5 ; <nl> n3 :   ->   ( P → Q ) <eos>
#     <st> n3 : ( P → Q ) <nl> n5 : P <nl> <act> have n9 : := n3 n5 ; <nl> n9 :   ->   Q <eos>
#
# To judge a step, every slot is scored with the step's own formula as the continuation; the
# step's score is the lowest probability the model gives to any token of any slot. Where a rule
# fixes a formula (most do) a wrong symbol gets a probability near zero; where a rule leaves part
# of it free (the other side of an `Or.inl`, the result of `False.elim`) the model spreads its
# probability and every choice stays above a small threshold. No refused steps are needed to train
# it and nothing but this network runs at test time.

_ST, _NL, _ACT = TOK.stoi["<st>"], TOK.stoi["<nl>"], TOK.stoi["<act>"]
_ASSIGN, _COLON = TOK.stoi[":="], TOK.stoi[":"]


def slots(seq):
    """compact step (ids) -> [(prefix ids, target ids)]: the stated formula, then each cited line."""
    seq = list(seq)
    ia = seq.index(_ACT)
    lines, cur = [], []
    for t in seq[1:ia]:
        if t == _NL:
            lines.append(cur)
            cur = []
        else:
            cur.append(t)
    act = seq[ia + 1 :]
    j = act.index(_ASSIGN)
    ctx = [_ST]
    for ln in lines:
        ctx += ln + [_NL]
    out = [(ctx + [_ACT] + act[:3] + act[j:] + [_NL, act[1], _COLON], act[3:j])]
    for k, ln in enumerate(lines):
        ctx = [_ST]
        for m, other in enumerate(lines):
            if m != k:
                ctx += other + [_NL]
        out.append((ctx + [_ACT] + act + [_NL, ln[0], _COLON], ln[2:]))
    return out


def slot_examples(seq):
    """-> [(bytes of prefix + target + <eos>, prefix length)]"""
    return [(bytes(p + t + [EOS]), len(p)) for p, t in slots(seq)]


def _slot_batch(items, dev):
    T = max(len(s) for s, _ in items)
    arr = np.full((len(items), T), PAD, np.uint8)
    for r, (s, _) in enumerate(items):
        arr[r, : len(s)] = np.frombuffer(s, np.uint8)
    x = torch.from_numpy(arr).to(dev).long()
    pos = torch.arange(T - 1, device=dev)[None]
    lens = torch.tensor([len(s) for s, _ in items], device=dev)[:, None]
    plen = torch.tensor([p for _, p in items], device=dev)[:, None]
    return x, (pos + 1 >= plen) & (pos + 1 < lens)  # mask over positions 0..T-2: is x[t+1] a target


@torch.no_grad()
def slot_probs(model, seqs, tok_budget=400_000):
    """-> for each compact step, the lowest probability the slot model gives to any token of any
    of the step's formulas when asked to write it from the rest of the step."""
    dev = next(model.parameters()).device
    cuda = dev.type == "cuda"
    items, owner = [], []
    for si, s in enumerate(seqs):
        for it in slot_examples(s):
            items.append(it)
            owner.append(si)
    out = [1.0] * len(seqs)
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
    i0 = 0
    while i0 < len(order):
        T = len(items[order[min(len(order) - 1, i0 + 8191)]][0])
        n = max(1, min(8192, tok_budget // max(1, T)))
        chunk = order[i0 : i0 + n]
        i0 += n
        x, mask = _slot_batch([items[i] for i in chunk], dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            logits = model.head(model.trunk(x[:, :-1])).float()
        lp = torch.log_softmax(logits, -1).gather(2, x[:, 1:, None])[:, :, 0]
        lo = torch.where(mask, lp, torch.zeros_like(lp)).amin(1).exp().tolist()
        for i, p in zip(chunk, lo):
            if p < out[owner[i]]:
                out[owner[i]] = p
    return out


SLOT_TAUS = (0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003)


def train_slot(pos, steps=8000, bs=1024, lr=6e-4, seed=0, log=print, cfg=None, holdout=0.02, test=None):
    """pos: compact steps (bytes) the checker accepted. -> (model, metrics). `test`: labelled
    compact steps [(bytes, 0/1)] never trained on, to read how many refused steps the model
    refuses and how many accepted steps it keeps at several thresholds."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = dev.type == "cuda"
    rng = random.Random(seed)
    torch.manual_seed(seed)
    idx = list(range(len(pos)))
    rng.shuffle(idx)
    n_ho = max(1, int(len(idx) * holdout))
    ho = [pos[i] for i in idx[:n_ho]]
    ex = [e for i in idx[n_ho:] for e in slot_examples(pos[i])]
    model = GPT(VOCAB, **(cfg or VER_CFG)).to(dev)
    model.kind = "slot"
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    t0 = time.time()
    model.train()
    acc_loss, queue = 0.0, []
    for s in range(steps):
        for gp in opt.param_groups:
            gp["lr"] = lr * min(1.0, (s + 1) / 200) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        if not queue:
            pool = [ex[rng.randrange(len(ex))] for _ in range(64 * bs)]
            pool.sort(key=lambda e: len(e[0]))
            i0 = 0
            while i0 < len(pool):
                i1 = i0 + 1
                while i1 < len(pool) and (i1 - i0 + 1) * len(pool[i1][0]) <= bs * 48:
                    i1 += 1
                queue.append(pool[i0:i1])
                i0 = i1
            rng.shuffle(queue)
        x, mask = _slot_batch(queue.pop(), dev)
        x = shift_names(x, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            logits = model.head(model.trunk(x[:, :-1])).float()
        loss = F.cross_entropy(logits[mask], x[:, 1:][mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        acc_loss += float(loss.detach())
        if (s + 1) % 1000 == 0 or s + 1 == steps:
            log(f"slot verifier step {s + 1}/{steps} loss {acc_loss / ((s % 1000) + 1):.5f} {time.time() - t0:.0f}s")
            acc_loss = 0.0
    model.eval()
    m = {"train_slots": len(ex), "holdout_n": len(ho)}
    p_ho = slot_probs(model, ho)
    for tau in SLOT_TAUS:
        m[f"holdout_pos_kept@{tau}"] = round(sum(p >= tau for p in p_ho) / max(1, len(p_ho)), 5)
    if test:
        pr = slot_probs(model, [c for c, _ in test])
        neg = [p for p, (_, lab) in zip(pr, test) if lab == 0]
        ps = [p for p, (_, lab) in zip(pr, test) if lab == 1]
        m["test_n"] = len(test)
        for tau in SLOT_TAUS:
            m[f"test_neg_caught@{tau}"] = round(sum(p < tau for p in neg) / max(1, len(neg)), 5)
            m[f"test_pos_kept@{tau}"] = round(sum(p >= tau for p in ps) / max(1, len(ps)), 5)
    return model, m


# ---- the masked slot verifier ("mslot"): the same consistency check, as infilling ----
#
# The plain slot model writes a formula left to right from nothing. Where a rule leaves part of a
# formula free, it has to guess that part, and it learned sharp habits about it: valid steps whose
# free part was unusual got probabilities like 1e-5 and were refused (2% of the valid steps on dev,
# concentrated in a few theorem families). The masked version shows the model the formula it is
# asked about with about half of its symbols hidden, and asks it to write the formula out:
#
#     <st> n5 : P <nl> n3 : ( ⊢ → ⊢ ) <nl> <act> have n9 : Q := n3 n5 ; <nl> n3 :   ->   ( P → Q )
#
# (`⊢` marks a hidden symbol; it never occurs in a step otherwise.) Only the hidden symbols are
# scored. Each formula of a step is scored twice, with a random half hidden and then the other
# half, so every symbol is hidden exactly once while its neighbours and the formula's length are
# in view. A symbol a rule fixes is still fixed by the other formulas; a symbol in a free part is
# now a choice among the few that fit the visible syntax, so none of them is improbable.

_MASK = TOK.stoi["⊢"]


def mslot_parts(seq):
    """compact step (ids) -> [(A, f, B)]: for each formula f of the step (the stated one, then each
    cited line's), the context before it and after it; the model input is A + template + B + f."""
    seq = list(seq)
    ia = seq.index(_ACT)
    lines, cur = [], []
    for t in seq[1:ia]:
        if t == _NL:
            lines.append(cur)
            cur = []
        else:
            cur.append(t)
    act = seq[ia + 1 :]
    j = act.index(_ASSIGN)
    ctx = [_ST]
    for ln in lines:
        ctx += ln + [_NL]
    out = [(ctx + [_ACT] + act[:3], act[3:j], act[j:] + [_NL, act[1], _COLON])]
    for k, ln in enumerate(lines):
        A, B = [_ST], []
        for m, other in enumerate(lines):
            if m < k:
                A += other + [_NL]
            elif m > k:
                B += other + [_NL]
        out.append((A + ln[:2], ln[2:], [_NL] + B + [_ACT] + act + [_NL, ln[0], _COLON]))
    return [(bytes(a), bytes(f), bytes(b)) for a, f, b in out if f]


def _mslot_batch(items, dev):
    """items: [(A, f, B, hidden)] with `hidden` a bool array over f. -> ids (B, T), and a mask over
    positions 0..T-2 marking where the next token is a hidden symbol of f."""
    T = max(len(a) + 2 * len(f) + len(b) for a, f, b, _ in items)
    arr = np.full((len(items), T), PAD, np.uint8)
    tgt = np.zeros((len(items), T - 1), bool)
    for r, (a, f, b, hid) in enumerate(items):
        n, la, lb = len(f), len(a), len(b)
        fa = np.frombuffer(f, np.uint8)
        arr[r, :la] = np.frombuffer(a, np.uint8)
        arr[r, la : la + n] = np.where(hid, _MASK, fa)
        arr[r, la + n : la + n + lb] = np.frombuffer(b, np.uint8)
        p0 = la + n + lb
        arr[r, p0 : p0 + n] = fa
        tgt[r, p0 - 1 : p0 - 1 + n] = hid
    return torch.from_numpy(arr).to(dev).long(), torch.from_numpy(tgt).to(dev)


@torch.no_grad()
def mslot_probs(model, seqs, tok_budget=400_000):
    """-> for each compact step, the lowest probability the model gives to any symbol of any of
    the step's formulas when that symbol is hidden (each symbol is hidden in one of two passes)."""
    import zlib

    dev = next(model.parameters()).device
    cuda = dev.type == "cuda"
    items, owner = [], []
    for si, s in enumerate(seqs):
        rng = np.random.default_rng(zlib.crc32(bytes(s)))
        for a, f, b in mslot_parts(s):
            hid = rng.random(len(f)) < 0.5
            for h in (hid, ~hid):
                if h.any():
                    items.append((a, f, b, h))
                    owner.append(si)
    out = [1.0] * len(seqs)
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]) + 2 * len(items[i][1]) + len(items[i][2]))
    i0 = 0
    while i0 < len(order):
        it = items[order[min(len(order) - 1, i0 + 8191)]]
        T = len(it[0]) + 2 * len(it[1]) + len(it[2])
        n = max(1, min(8192, tok_budget // max(1, T)))
        chunk = order[i0 : i0 + n]
        i0 += n
        x, mask = _mslot_batch([items[i] for i in chunk], dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            logits = model.head(model.trunk(x[:, :-1])).float()
        lp = torch.log_softmax(logits, -1).gather(2, x[:, 1:, None])[:, :, 0]
        lo = torch.where(mask, lp, torch.zeros_like(lp)).amin(1).exp().tolist()
        for i, p in zip(chunk, lo):
            if p < out[owner[i]]:
                out[owner[i]] = p
    return out


def train_mslot(pos, steps=8000, bs=1024, lr=6e-4, seed=0, log=print, cfg=None, holdout=0.02, test=None):
    """pos: compact steps (bytes) the checker accepted. -> (model, metrics); see `train_slot`."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = dev.type == "cuda"
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    idx = list(range(len(pos)))
    rng.shuffle(idx)
    n_ho = max(1, int(len(idx) * holdout))
    ho = [pos[i] for i in idx[:n_ho]]
    ex = [e for i in idx[n_ho:] for e in mslot_parts(pos[i])]
    model = GPT(VOCAB, **(cfg or VER_CFG)).to(dev)
    model.kind = "mslot"
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    t0 = time.time()
    model.train()
    acc_loss, queue = 0.0, []
    for s in range(steps):
        for gp in opt.param_groups:
            gp["lr"] = lr * min(1.0, (s + 1) / 200) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        if not queue:
            pool = [ex[rng.randrange(len(ex))] for _ in range(64 * bs)]
            pool.sort(key=lambda e: len(e[0]) + 2 * len(e[1]) + len(e[2]))
            i0 = 0
            while i0 < len(pool):
                i1 = i0 + 1
                e = pool[i1] if i1 < len(pool) else None
                while i1 < len(pool) and (i1 - i0 + 1) * (len(e[0]) + 2 * len(e[1]) + len(e[2])) <= bs * 64:
                    i1 += 1
                    e = pool[i1] if i1 < len(pool) else None
                queue.append(pool[i0:i1])
                i0 = i1
            rng.shuffle(queue)
        items = []
        for a, f, b in queue.pop():
            hid = nrng.random(len(f)) < nrng.uniform(0.3, 0.8)  # the hidden share varies per example
            if not hid.any():
                hid[nrng.integers(len(f))] = True
            items.append((a, f, b, hid))
        x, mask = _mslot_batch(items, dev)
        x = shift_names(x, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            logits = model.head(model.trunk(x[:, :-1])).float()
        loss = F.cross_entropy(logits[mask], x[:, 1:][mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        acc_loss += float(loss.detach())
        if (s + 1) % 1000 == 0 or s + 1 == steps:
            log(f"mslot verifier step {s + 1}/{steps} loss {acc_loss / ((s % 1000) + 1):.5f} {time.time() - t0:.0f}s")
            acc_loss = 0.0
    model.eval()
    m = {"train_slots": len(ex), "holdout_n": len(ho)}
    p_ho = mslot_probs(model, ho)
    for tau in SLOT_TAUS:
        m[f"holdout_pos_kept@{tau}"] = round(sum(p >= tau for p in p_ho) / max(1, len(p_ho)), 5)
    if test:
        pr = mslot_probs(model, [c for c, _ in test])
        neg = [p for p, (_, lab) in zip(pr, test) if lab == 0]
        ps = [p for p, (_, lab) in zip(pr, test) if lab == 1]
        m["test_n"] = len(test)
        for tau in SLOT_TAUS:
            m[f"test_neg_caught@{tau}"] = round(sum(p < tau for p in neg) / max(1, len(neg)), 5)
            m[f"test_pos_kept@{tau}"] = round(sum(p >= tau for p in ps) / max(1, len(ps)), 5)
    return model, m


def load_examples(run_dir, max_per_round=10**9, seed=0):
    """-> (list of (bytes, label)), per-round counts."""
    rng = random.Random(seed)
    ex, counts = [], {}
    for fn in sorted(glob.glob(os.path.join(run_dir, "ver_r*.pkl")),
                     key=lambda f: int(os.path.basename(f)[5:-4])):  # fmt: skip
        with open(fn, "rb") as f:
            d = pickle.load(f)
        n0 = len(ex)
        for lab, key in ((0, "neg"), (1, "pos")):
            items = d[key]
            if len(items) > max_per_round:
                items = rng.sample(items, max_per_round)
            for s, a in items:
                c = compact_bytes(s, a)
                if c is not None:
                    ex.append((c, lab))
        counts[os.path.basename(fn)] = len(ex) - n0
    return ex, counts


def x_positives(x_path, n, seed=0):
    """n accepted steps from the pretraining corpus (every step of an x proof is well typed)."""
    from .data import load_x

    tr, _, _ = load_x(x_path, os.path.join(os.path.dirname(x_path), "x_cache.pkl"))
    idx = np.flatnonzero(tr.lab[:, 1] == 1)
    rng = np.random.default_rng(seed)
    sel = rng.choice(idx, size=min(n, len(idx)), replace=False)
    out = []
    for i in sel:
        seq = tr.flat[tr.off[i] : tr.off[i + 1]]
        k = int(tr.nst[i])
        c = compact_bytes(bytes(seq[:k]), bytes(seq[k:-1]))
        if c is not None:
            out.append((c, 1))
    return out


_ATOMS = ("P", "Q", "R", "S")
_CONN = ("∧", "∨", "→")
_SWAP = {".1": ".2", ".2": ".1", "Or.inl": "Or.inr", "Or.inr": "Or.inl"}


def _label(lines, act):
    """The training-time checker's verdict on a compact step (1 well typed, 0 not, None unknown)."""
    from .common import LP, parse_ftoks

    try:
        scope, prem = {}, []
        for ln in lines:
            f, used = parse_ftoks(ln, 2)
            if used != len(ln) - 2:
                return None
            if ln[0][0] == "h":
                k = int(ln[0][1:])
                prem = [("atom", "P")] * (k - 1) + [f]
            else:
                scope[ln[0]] = f
        j = act.index(":=")
        f, used = parse_ftoks(act, 3)
        if 3 + used != j:
            return None
        pp = LP._P(list(act[j + 1 :]))
        LP._term(pp, scope, f, prem)
        return 1 if pp.peek() == ";" else None
    except LP._Reject:
        return 0
    except Exception:  # noqa: BLE001
        return None


def near_misses(comp_b, rng, n=2):
    """Small edits of one step (an atom or connective changed in the stated formula or in a cited
    line, `.1`/`.2` or `Or.inl`/`Or.inr` swapped, two cited names swapped), each labelled by the
    training-time checker. Most come out ill typed and differ from a well-typed step in one token:
    the cases a verifier trained only on the policy's own refused steps kept passing."""
    toks = [ITOS[i] for i in comp_b]
    ia = toks.index("<act>")
    lines, cur = [], []
    for t in toks[1:ia]:
        if t == "<nl>":
            lines.append(cur)
            cur = []
        else:
            cur.append(t)
    act = toks[ia + 1 :]
    if ":=" not in act:
        return []
    j = act.index(":=")
    out = []
    for _ in range(n * 3):
        if len(out) >= n:
            break
        L = [list(x) for x in lines]
        A = list(act)
        op = rng.random()
        if op < 0.45:  # one symbol of the stated formula
            pos = [i for i in range(3, j) if A[i] in _ATOMS or A[i] in _CONN]
            if not pos:
                continue
            i = rng.choice(pos)
            A[i] = rng.choice([x for x in (_ATOMS if A[i] in _ATOMS else _CONN) if x != A[i]])
        elif op < 0.75 and L:  # one symbol of a cited line
            ln = rng.choice(L)
            pos = [i for i in range(2, len(ln)) if ln[i] in _ATOMS or ln[i] in _CONN]
            if not pos:
                continue
            i = rng.choice(pos)
            ln[i] = rng.choice([x for x in (_ATOMS if ln[i] in _ATOMS else _CONN) if x != ln[i]])
        else:  # the term
            term = A[j + 1 : -1]
            sw = [i for i, t in enumerate(term) if t in _SWAP]
            names = [i for i, t in enumerate(term) if is_name(t)]
            if sw:
                i = rng.choice(sw)
                term[i] = _SWAP[term[i]]
            elif len(names) >= 2 and term[names[0]] != term[names[1]]:
                a, b = names[0], names[1]
                term[a], term[b] = term[b], term[a]
            else:
                continue
            A = A[: j + 1] + term + [";"]
        lab = _label(L, A)
        if lab is None:
            continue
        seq = ["<st>"]
        for ln in L:
            seq += ln + ["<nl>"]
        out.append((enc(seq + ["<act>"] + A), lab))
    return out


def _batch(ex, idx, dev):
    T = max(len(ex[i][0]) for i in idx)
    arr = np.full((len(idx), T), PAD, np.uint8)
    for r, i in enumerate(idx):
        arr[r, : len(ex[i][0])] = np.frombuffer(ex[i][0], np.uint8)
    x = torch.from_numpy(arr).to(dev).long()
    last = torch.tensor([len(ex[i][0]) - 1 for i in idx], device=dev)
    y = torch.tensor([ex[i][1] for i in idx], device=dev, dtype=torch.float32)
    return x, last, y


def _metrics(model, ex, idx, tag):
    probs = ver_probs(model, [ex[i][0] for i in idx], views=1)
    labs = [ex[i][1] for i in idx]
    neg = [p for p, lab in zip(probs, labs) if lab == 0]
    pos = [p for p, lab in zip(probs, labs) if lab == 1]
    m = {f"{tag}_n": len(idx)}
    for tau in (0.5, 0.9):
        m[f"{tag}_neg_caught@{tau}"] = round(sum(p < tau for p in neg) / max(1, len(neg)), 5)
        m[f"{tag}_pos_kept@{tau}"] = round(sum(p >= tau for p in pos) / max(1, len(pos)), 5)
    return m


def train_verifier(ex, steps=4000, bs=1024, lr=6e-4, seed=0, log=print, init=None, holdout=0.02,
                   cfg=None, mine_rounds=0, mine_steps=3000, test=None):  # fmt: skip
    """-> (model, metrics on a held-out share of the examples, and on `test` if given).

    `mine_rounds`: after the main run, score every training example, collect the ones the network
    gets wrong (a refused step it would pass, an accepted step it would refuse) and train on
    batches that are half such hard examples; repeated. The policy's proposals at test time are
    filtered by this network, so the rare steps it wrongly passes are exactly what gets through."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = dev.type == "cuda"
    rng = random.Random(seed)
    torch.manual_seed(seed)
    idx = list(range(len(ex)))
    rng.shuffle(idx)
    n_ho = max(1, int(len(idx) * holdout))
    ho, tr = idx[:n_ho], idx[n_ho:]
    model = GPT(VOCAB, **(cfg or VER_CFG)).to(dev)
    if init is not None:
        model.load_state_dict(init)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    t0 = time.time()

    def run(n_steps, peak, hard, tag):
        model.train()
        acc_loss, queue = 0.0, []
        for s in range(n_steps):
            for gp in opt.param_groups:
                gp["lr"] = peak * min(1.0, (s + 1) / 200) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * s / n_steps)))
            if not queue:  # length-bucketed batches: a pool sorted by length, cut by tokens
                pool = [tr[rng.randrange(len(tr))] for _ in range(64 * bs)]
                if hard:
                    pool = pool[: 32 * bs] + [hard[rng.randrange(len(hard))] for _ in range(32 * bs)]
                pool.sort(key=lambda i: len(ex[i][0]))
                i0 = 0
                while i0 < len(pool):
                    i1 = i0 + 1
                    while i1 < len(pool) and (i1 - i0 + 1) * len(ex[pool[i1]][0]) <= bs * 48:
                        i1 += 1
                    queue.append(pool[i0:i1])
                    i0 = i1
                rng.shuffle(queue)
            b = queue.pop()
            x, last, y = _batch(ex, b, dev)
            x = shift_names(x, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                h = model.trunk(x)
                logit = model.aux(h[torch.arange(len(b), device=dev), last])[:, 0].float()
            loss = F.binary_cross_entropy_with_logits(logit, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            acc_loss += float(loss.detach())
            if (s + 1) % 1000 == 0 or s + 1 == n_steps:
                log(f"verifier {tag} step {s + 1}/{n_steps} loss {acc_loss / ((s % 1000) + 1):.4f} {time.time() - t0:.0f}s")
                acc_loss = 0.0
        model.eval()

    run(steps, lr, None, "main")
    m = _metrics(model, ex, ho, "holdout")
    if test:
        m.update(_metrics(model, test, list(range(len(test))), "test"))
    log("after main: " + json.dumps(m))
    for r in range(mine_rounds):
        probs = ver_probs(model, [ex[i][0] for i in tr])
        hard = [i for i, p in zip(tr, probs) if (p >= 0.2) != (ex[i][1] == 1) or (0.2 <= p < 0.8)]
        n_fa = sum(1 for i in hard if ex[i][1] == 0)
        log(f"mining round {r + 1}: {len(hard)} hard examples ({n_fa} refused steps passed, {len(hard) - n_fa} accepted steps doubted)")
        if not hard:
            break
        run(mine_steps, lr / 3, hard, f"mine{r + 1}")
        m = _metrics(model, ex, ho, "holdout")
        if test:
            m.update(_metrics(model, test, list(range(len(test))), "test"))
        log(f"after mining round {r + 1}: " + json.dumps(m))
    m["train"] = len(tr)
    return model, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="an RL run directory with ver_r*.pkl files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_per_round", type=int, default=10**9)
    ap.add_argument("--near", type=int, default=0, help="near-miss edits per accepted step (checker-labelled)")
    ap.add_argument("--mine_rounds", type=int, default=0)
    ap.add_argument("--mine_steps", type=int, default=3000)
    ap.add_argument("--test_run", default=None, help="a directory of ver_r*.pkl files never trained on, for a test read")
    ap.add_argument("--n_layer", type=int, default=VER_CFG["n_layer"])
    ap.add_argument("--d", type=int, default=VER_CFG["d"])
    ap.add_argument("--x", default=None, help="pretraining corpus: add --x_pos accepted steps from it as positives")
    ap.add_argument("--x_pos", type=int, default=300_000)
    ap.add_argument("--kind", default="disc", choices=["disc", "slot", "mslot"],
                    help="disc: one logit per step, trained on refused and accepted steps; "
                         "slot: the generative consistency model, trained on accepted steps only")  # fmt: skip
    a = ap.parse_args()
    t0 = time.time()
    ex, counts = load_examples(a.run, a.max_per_round, a.seed)
    if a.x:
        xp = x_positives(a.x, a.x_pos, a.seed)
        counts["x_positives"] = len(xp)
        ex += xp
    if a.near:
        rng = random.Random(a.seed + 5)
        nm = [e for c, lab in ex if lab == 1 for e in near_misses(c, rng, a.near)]
        counts["near_misses"] = len(nm)
        counts["near_misses_ill_typed"] = sum(1 for _, lab in nm if lab == 0)
        ex += nm
    print(f"examples {len(ex)} from {len(counts)} rounds ({time.time() - t0:.0f}s): {json.dumps(counts)}", flush=True)
    cfg = dict(n_layer=a.n_layer, d=a.d, n_head=8)
    test = None
    if a.test_run:
        test, tc = load_examples(a.test_run, a.max_per_round, a.seed)
        print(f"test examples {len(test)}: {json.dumps(tc)}", flush=True)
    if a.kind in ("slot", "mslot"):
        pos = [c for c, lab in ex if lab == 1]
        fn = train_slot if a.kind == "slot" else train_mslot
        model, m = fn(pos, a.steps, a.bs, a.lr, a.seed, log=lambda s: print(s, flush=True), cfg=cfg, test=test)
    else:
        model, m = train_verifier(ex, a.steps, a.bs, a.lr, a.seed, log=lambda s: print(s, flush=True), cfg=cfg,
                                  mine_rounds=a.mine_rounds, mine_steps=a.mine_steps, test=test)  # fmt: skip
    print("holdout " + json.dumps(m), flush=True)
    torch.save({"cfg": cfg, "kind": a.kind, "state": model.state_dict(), "metrics": m, "rounds": counts,
                "args": vars(a)}, a.out)  # fmt: skip
    print("VERIFIER_DONE", flush=True)


if __name__ == "__main__":
    main()
