"""The verifier: a separate small network that judges one proof step.

Input: the hypothesis lines the step cites, copied from the proof state, then the step itself:

    <st> n3 : ( P → Q ) <nl> n5 : P <nl> <act> have n9 : Q := n3 n5 ;

Which lines a step cites is read off the step's text (the names in its term; `h<k>` for a premise
line). Whether the step is well typed is the network's judgement, read as one logit at the last
token. It is trained on steps the policy sampled during RL on y: refused by the per-step type
check (negatives) or accepted (positives), saved each round by `fr/rl.py` as `ver_r<round>.pkl`.
At test time nothing but this network screens a step (no Lean, no type check).

It is a separate network because training this judgement on the policy's own trunk damaged the
policy (PREREG amendment 2).

    python -m fr.verifier --run /workspace/runs/rl_s0 --out /workspace/runs/rl_s0/verifier.pt
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


@torch.no_grad()
def ver_probs(model, seqs, tok_budget=400_000):
    """seqs: list of id sequences (bytes) -> probability that each step is well typed."""
    if not seqs:
        return []
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
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            h = model.trunk(x)
            logit = model.aux(h[torch.arange(len(chunk), device=dev), last])[:, 0].float()
        for i, p in zip(chunk, torch.sigmoid(logit).tolist()):
            out[i] = p
    return out


def build_verifier(extra, device):
    """The verifier stored in a checkpoint's `extra['verifier']`, or None."""
    v = (extra or {}).get("verifier")
    if not v:
        return None
    m = GPT(VOCAB, **v["cfg"]).to(device)
    m.load_state_dict(v["state"])
    m.eval()
    return m


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


def _batch(ex, idx, dev):
    T = max(len(ex[i][0]) for i in idx)
    arr = np.full((len(idx), T), PAD, np.uint8)
    for r, i in enumerate(idx):
        arr[r, : len(ex[i][0])] = np.frombuffer(ex[i][0], np.uint8)
    x = torch.from_numpy(arr).to(dev).long()
    last = torch.tensor([len(ex[i][0]) - 1 for i in idx], device=dev)
    y = torch.tensor([ex[i][1] for i in idx], device=dev, dtype=torch.float32)
    return x, last, y


def train_verifier(ex, steps=4000, bs=1024, lr=6e-4, seed=0, log=print, init=None, holdout=0.02):
    """-> (model, metrics on a held-out share of the examples)."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = dev.type == "cuda"
    rng = random.Random(seed)
    torch.manual_seed(seed)
    idx = list(range(len(ex)))
    rng.shuffle(idx)
    n_ho = max(1, int(len(idx) * holdout))
    ho, tr = idx[:n_ho], idx[n_ho:]
    model = GPT(VOCAB, **VER_CFG).to(dev)
    if init is not None:
        model.load_state_dict(init)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    model.train()
    t0 = time.time()
    acc_loss = 0.0
    queue = []
    for s in range(steps):
        for gp in opt.param_groups:
            gp["lr"] = lr * min(1.0, (s + 1) / 200) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        if not queue:  # length-bucketed batches: a pool of examples sorted by length, cut by tokens
            pool = sorted((tr[rng.randrange(len(tr))] for _ in range(64 * bs)), key=lambda i: len(ex[i][0]))
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
        if (s + 1) % 500 == 0 or s + 1 == steps:
            log(f"verifier step {s + 1}/{steps} loss {acc_loss / (500 if (s + 1) % 500 == 0 else (steps % 500 or 500)):.4f} "
                f"{time.time() - t0:.0f}s")  # fmt: skip
            acc_loss = 0.0
    model.eval()
    probs = ver_probs(model, [ex[i][0] for i in ho])
    labs = [ex[i][1] for i in ho]
    m = {"holdout": len(ho), "train": len(tr), "pos_share": sum(labs) / len(labs)}
    for tau in (0.3, 0.5, 0.7, 0.9):
        neg = [p for p, lab in zip(probs, labs) if lab == 0]
        pos = [p for p, lab in zip(probs, labs) if lab == 1]
        m[f"neg_caught@{tau}"] = sum(p < tau for p in neg) / max(1, len(neg))
        m[f"pos_kept@{tau}"] = sum(p >= tau for p in pos) / max(1, len(pos))
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
    ap.add_argument("--x", default=None, help="pretraining corpus: add --x_pos accepted steps from it as positives")
    ap.add_argument("--x_pos", type=int, default=300_000)
    a = ap.parse_args()
    t0 = time.time()
    ex, counts = load_examples(a.run, a.max_per_round, a.seed)
    if a.x:
        xp = x_positives(a.x, a.x_pos, a.seed)
        counts["x_positives"] = len(xp)
        ex += xp
    print(f"examples {len(ex)} from {len(counts)} rounds ({time.time() - t0:.0f}s): {json.dumps(counts)}", flush=True)
    model, m = train_verifier(ex, a.steps, a.bs, a.lr, a.seed, log=lambda s: print(s, flush=True))
    print("holdout " + json.dumps(m), flush=True)
    torch.save({"cfg": VER_CFG, "state": model.state_dict(), "metrics": m, "rounds": counts,
                "args": vars(a)}, a.out)  # fmt: skip
    print("VERIFIER_DONE", flush=True)


if __name__ == "__main__":
    main()
