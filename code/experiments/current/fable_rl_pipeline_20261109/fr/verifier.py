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


VIEWS = 1  # verifier passes per step at test time; views beyond the first shift the names


@torch.no_grad()
def ver_probs(model, seqs, tok_budget=400_000, views=None):
    """seqs: list of id sequences (bytes) -> probability that each step is well typed. With
    `views` > 1 the step is judged again under random name offsets (the same statement to the
    network, different name tokens) and the lowest probability is returned."""
    views = VIEWS if views is None else views
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
        rows = torch.arange(len(chunk), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            logit = model.aux(model.trunk(x)[rows, last])[:, 0].float()
            for _ in range(views - 1):
                lg = model.aux(model.trunk(shift_names(x, low_bias=0.0))[rows, last])[:, 0].float()
                logit = torch.minimum(logit, lg)
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
    ap.add_argument("--mine_rounds", type=int, default=0)
    ap.add_argument("--mine_steps", type=int, default=3000)
    ap.add_argument("--test_run", default=None, help="a directory of ver_r*.pkl files never trained on, for a test read")
    ap.add_argument("--n_layer", type=int, default=VER_CFG["n_layer"])
    ap.add_argument("--d", type=int, default=VER_CFG["d"])
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
    cfg = dict(n_layer=a.n_layer, d=a.d, n_head=8)
    test = None
    if a.test_run:
        test, tc = load_examples(a.test_run, a.max_per_round, a.seed)
        print(f"test examples {len(test)}: {json.dumps(tc)}", flush=True)
    model, m = train_verifier(ex, a.steps, a.bs, a.lr, a.seed, log=lambda s: print(s, flush=True), cfg=cfg,
                              mine_rounds=a.mine_rounds, mine_steps=a.mine_steps, test=test)
    print("holdout " + json.dumps(m), flush=True)
    torch.save({"cfg": cfg, "state": model.state_dict(), "metrics": m, "rounds": counts,
                "args": vars(a)}, a.out)  # fmt: skip
    print("VERIFIER_DONE", flush=True)


if __name__ == "__main__":
    main()
