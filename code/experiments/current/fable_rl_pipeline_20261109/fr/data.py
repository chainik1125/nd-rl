"""Training examples and batches.

An example is one (state, action) sequence `state ids + action ids + <eos>` with three labels:
`pol` (1: train the policy on the action tokens), `ver` (verifier target at the last action token:
1 well typed, 0 not, -1 none), `val` (value target at `<act>`: 1 the attempt finished, 0 it did
not, -1 none). A `Store` packs many examples into flat arrays.
"""

from __future__ import annotations

import gzip
import json
import os
import pickle
import random
from multiprocessing import Pool

import numpy as np
import torch

from .common import EOS, MAXN, PAD, REF0, enc, nd_pairs, step_kind


class Store:
    def __init__(self, examples=None):
        """examples: iterable of (state bytes, action bytes, pol, ver, val)."""
        self.flat = np.zeros(0, np.uint8)
        self.off = np.zeros(1, np.int64)
        self.nst = np.zeros(0, np.int32)
        self.lab = np.zeros((0, 3), np.int8)
        if examples:
            self.extend(examples)

    def __len__(self):
        return len(self.nst)

    def extend(self, examples):
        bufs, lens, nst, lab = [], [], [], []
        for s, a, pol, ver, val in examples:
            bufs.append(bytes(s) + bytes(a) + bytes([EOS]))
            lens.append(len(s) + len(a) + 1)
            nst.append(len(s))
            lab.append((pol, ver, val))
        if not bufs:
            return
        self.flat = np.concatenate([self.flat, np.frombuffer(b"".join(bufs), np.uint8)])
        self.off = np.concatenate([self.off, self.off[-1] + np.cumsum(lens)])
        self.nst = np.concatenate([self.nst, np.asarray(nst, np.int32)])
        self.lab = np.concatenate([self.lab, np.asarray(lab, np.int8).reshape(-1, 3)])

    def lens(self, idx):
        return self.off[idx + 1] - self.off[idx]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump((self.flat, self.off, self.nst, self.lab), f, protocol=4)

    @classmethod
    def load(cls, path):
        s = cls()
        with open(path, "rb") as f:
            s.flat, s.off, s.nst, s.lab = pickle.load(f)
        return s


def proof_examples(prompt: str, nd: str):
    """An ND proof -> policy examples; atomic steps are also verifier positives."""
    pairs, _ = nd_pairs(prompt, nd)
    out = []
    for s, a in pairs:
        k = step_kind(a)
        out.append((enc(s), enc(a), 1, 1 if k in ("atom", "pr") else -1, -1))
    return out


def _x_worker(rows):
    out = []
    for prompt, proof in rows:
        try:
            out.append(proof_examples(prompt, proof))
        except Exception:
            out.append(None)
    return out


def load_x(path: str, cache: str, n_val: int = 1500, workers: int = 8):
    """The pretraining corpus -> (train Store, val Store). A fixed 1,500 proofs are kept out of
    training as a validation split (seed 0). Tokenised once and cached."""
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            a, b, meta = pickle.load(f)
        tr, va = Store(), Store()
        tr.flat, tr.off, tr.nst, tr.lab = a
        va.flat, va.off, va.nst, va.lab = b
        return tr, va, meta
    rows = []
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows.append((r["prompt"], r["proof"]))
    random.Random(0).shuffle(rows)
    chunks = [rows[i : i + 500] for i in range(0, len(rows), 500)]
    with Pool(workers) as pool:
        res = [x for ch in pool.map(_x_worker, chunks) for x in ch]
    skipped = sum(1 for x in res if x is None)
    res = [x for x in res if x is not None]
    va = Store([e for p in res[:n_val] for e in p])
    tr = Store([e for p in res[n_val:] for e in p])
    meta = {"proofs": len(rows), "skipped": skipped, "train_proofs": len(res) - n_val,
            "val_proofs": n_val, "train_pairs": len(tr), "val_pairs": len(va)}  # fmt: skip
    with open(cache, "wb") as f:
        pickle.dump(((tr.flat, tr.off, tr.nst, tr.lab), (va.flat, va.off, va.nst, va.lab), meta),
                    f, protocol=4)  # fmt: skip
    return tr, va, meta


def make_batch(picks, dev):
    """picks: list of (Store, index array) -> dict of device tensors for one padded batch."""
    seqs, nst, lab = [], [], []
    for st, idx in picks:
        for i in idx:
            seqs.append(st.flat[st.off[i] : st.off[i + 1]])
        nst.append(st.nst[idx])
        lab.append(st.lab[idx])
    lens = np.fromiter((len(s) for s in seqs), np.int64, len(seqs))
    T = int(lens.max())
    x = np.full((len(seqs), T), PAD, np.uint8)
    for r, s in enumerate(seqs):
        x[r, : len(s)] = s
    return {
        "x": torch.from_numpy(x).to(dev, non_blocking=True).long(),
        "len": torch.from_numpy(lens).to(dev),
        "nst": torch.from_numpy(np.concatenate(nst).astype(np.int64)).to(dev),
        "lab": torch.from_numpy(np.concatenate(lab).astype(np.int64)).to(dev),
    }


def shift_names(x, gen=None, low_bias=0.5, low_max=8):
    """Name augmentation: add one random offset per row to every name token, so the row's names
    start anywhere in n1..n64 (half of the time within the first `low_max` offsets, which is
    where sampling starts them)."""
    is_name = x >= REF0
    if not bool(is_name.any()):
        return x
    k = x - REF0 + 1
    mx = torch.where(is_name, k, torch.zeros_like(k)).amax(1)
    mn = torch.where(is_name, k, torch.full_like(k, MAXN + 1)).amin(1)
    has = mx > 0
    span = (mn - 1) + (MAXN - mx)  # offsets available: new minimum name in 1..span+1
    u = torch.rand(x.shape[0], generator=gen, device=x.device)
    low = torch.rand(x.shape[0], generator=gen, device=x.device) < low_bias
    lim = torch.where(low, span.clamp(max=low_max), span)
    new_mn = 1 + torch.floor(u * (lim + 1).float()).long().clamp(max=lim)
    s = torch.where(has, new_mn - mn, torch.zeros_like(mn))
    return torch.where(is_name, x + s[:, None], x)


class Mixer:
    """Draws padded batches from several Stores in fixed proportions (by example), bucketed by
    length so little of a batch is padding."""

    def __init__(self, sources, tok_budget=120_000, seed=0, chunk_batches=64, max_len=1400):
        """sources: list of (Store, weight); empty stores are dropped."""
        self.sources = [(s, w) for s, w in sources if len(s) > 0 and w > 0]
        tot = sum(w for _, w in self.sources)
        self.w = [w / tot for _, w in self.sources]
        self.rng = np.random.default_rng(seed)
        self.tok_budget = tok_budget
        self.chunk_batches = chunk_batches
        self.max_len = max_len
        self.queue = []
        ml = [float(np.mean(s.lens(self.rng.integers(0, len(s), 2000)))) for s, _ in self.sources]
        self.mean_len = sum(w * m for w, m in zip(self.w, ml))

    def _refill(self):
        n = int(self.chunk_batches * self.tok_budget / (self.mean_len * 1.15))
        items = []
        for k, ((st, _), w) in enumerate(zip(self.sources, self.w)):
            idx = self.rng.integers(0, len(st), max(1, int(round(n * w))))
            ln = st.lens(idx)
            ok = ln <= self.max_len
            items.append(np.stack([np.full(ok.sum(), k), idx[ok], ln[ok]], 1))
        items = np.concatenate(items)
        items = items[np.argsort(items[:, 2] + self.rng.random(len(items)), kind="stable")]
        batches = []
        i = 0
        while i < len(items):
            j = i + 1
            while j < len(items) and (j - i + 1) * items[j, 2] <= self.tok_budget:
                j += 1
            batches.append(items[i:j])
            i = j
        self.rng.shuffle(batches)
        self.queue = batches

    def next(self, dev):
        if not self.queue:
            self._refill()
        b = self.queue.pop()
        picks = []
        for k, (st, _) in enumerate(self.sources):
            idx = b[b[:, 0] == k, 1]
            if len(idx):
                picks.append((st, idx))
        return make_batch(picks, dev)
