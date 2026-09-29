#!/usr/bin/env python3
# imports the fork (via harness) and a checkpoint's pretrain.py, neither resolvable here
# pyright: reportMissingImports=false
"""Value head for test-time search (value_head_plan.md): collect resample rollouts from a frozen
checkpoint on rl_targets (never dev or holdout), label every prefix on a rollout's chain with the
rollout's outcome, and train an MLP on the frozen trunk's final hidden state to predict it.

The 1,500 rl_targets the checkpoint's own RL stage trained on (the harness samples them with the
run's seed) are excluded, so the head never learns from theorems the policy has proofs memorized
for.

  python3 value.py --run combo-c005_s0 --out value_s0.json   # writes value_s0.pt (the head) too

Value(model, head)(prompt_ids, gens) is what ttscale.py's search methods call at eval time.
"""

import argparse
import collections
import hashlib
import json
import random
import time

import harness
import torch
import torch.nn.functional as F
from engine import Gate, Node, check, sample_segments
from ttscale import SEG_MAX, load

N_VAL = 495  # rl_targets theorems held out from head training, by a hash of the key
D_HIDDEN = 256


class Head(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, D_HIDDEN), torch.nn.GELU(), torch.nn.Linear(D_HIDDEN, 1)
        )

    def forward(self, h):
        return self.net(h).squeeze(-1)  # logit of V


@torch.no_grad()
def hidden_at(model, seqs, positions, batch=64):
    """Final hidden state (after ln_f) of each right-padded sequence at the given positions: one
    causal forward per sequence yields every prefix's feature at once."""
    dev = next(model.parameters()).device
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    res = [None] * len(seqs)
    for s in range(0, len(order), batch):
        chunk = order[s : s + batch]
        T = max(len(seqs[i]) for i in chunk)
        idx = torch.zeros((len(chunk), T), dtype=torch.long, device=dev)
        for j, i in enumerate(chunk):
            idx[j, : len(seqs[i])] = torch.tensor(seqs[i], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            h = model.ln_f(model.features(idx)[0]).float()  # causal: pads after a row are unseen
        for j, i in enumerate(chunk):
            res[i] = h[j, positions[i]].cpu()
    return res


class Value:
    """V(prefix) for search: sigmoid(head(hidden at the prefix's last token))."""

    def __init__(self, model, path):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        self.model = model
        self.head = Head(ck["d"]).to(next(model.parameters()).device)
        self.head.load_state_dict(ck["state"])
        self.head.eval()

    @torch.no_grad()
    def __call__(self, prompt_ids, gens, stats):
        t0 = time.time()
        seqs = [p + g for p, g in zip(prompt_ids, gens, strict=True)]
        hs = hidden_at(self.model, seqs, [[len(s) - 1] for s in seqs])
        dev = next(self.head.parameters()).device
        v = torch.sigmoid(self.head(torch.cat(hs).to(dev))).tolist()
        stats["value_tokens"] += sum(len(s) for s in seqs)
        stats["value_secs"] += time.time() - t0
        return v


def collect(model, tok, recs, k, cap, gen, stats, batch):
    """Resample rollouts (round-1 method 1, without per-row stats). Per rollout: theorem index,
    final generated ids, segment boundaries (0 = root), outcome."""
    rows = [
        {"t": i, "node": Node.root(r["prompt"]), "cuts": [0], "rej": 0, "end": None}
        for i, r in enumerate(recs)
        for _ in range(k)
    ]
    pids = [tok.encode_prompt(r["prompt"]) for r in recs]
    live = rows
    while live:
        items = [
            (pids[w["t"]] + w["node"].gen, min(SEG_MAX, harness.MAX_NEW - len(w["node"].gen)))
            for w in live
        ]
        segs = sample_segments(model, tok, items, harness.TEMP, gen, batch, stats)
        for w, (seg, end, lp, _) in zip(live, segs, strict=True):
            verdict, child = check(w["node"], seg, end, lp, tok.itos)
            if verdict == "ok":
                w["node"] = child
                w["cuts"].append(len(child.gen))
            elif verdict == "done":
                w["node"], w["end"] = child, "done"
            else:
                w["rej"] += 1
                if w["rej"] > cap:
                    w["end"] = "cap"
            if w["end"] is None and len(w["node"].gen) >= harness.MAX_NEW:
                w["end"] = "max_new"
        live = [w for w in live if w["end"] is None]
    gate = Gate(tok, stats)
    fin = [w for w in rows if w["end"] == "done"]
    ok = gate([recs[w["t"]]["prompt"] for w in fin], [w["node"].gen for w in fin])
    for w, o in zip(fin, ok, strict=True):
        w["ok"] = o
    return [(w["t"], w["node"].gen, w["cuts"], int(w.get("ok", False))) for w in rows], pids


def auc(scores, labels):
    """Probability a random positive outranks a random negative (ties count half)."""
    pairs = sorted(zip(scores, labels, strict=True))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank_sum += (i + j + 1) / 2 * sum(lab for _, lab in pairs[i:j])
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2) / max(n_pos * n_neg, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt-dir")
    ap.add_argument("--limit", type=int, default=0, help="rl_targets theorems (0 = all 4,495)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--cap", type=int, default=10)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    model, tok, seed = load(a.run, a.ckpt_dir)
    recs = harness.read(f"{harness.FORK}/data/ladder/rl_targets.jsonl")
    heldout = {r["key"] for r in harness.read(f"{harness.FORK}/data/ladder/transfer.jsonl")}
    assert not heldout & {r["key"] for r in recs}, "rl_targets overlaps transfer (dev/holdout)"
    rl = random.Random(seed).sample(recs, min(harness.N_TARGETS, len(recs)))  # as harness.main
    rl_keys = {r["key"] for r in rl}
    recs = [r for r in recs if r["key"] not in rl_keys]
    rank = lambda r: hashlib.sha1(f"value:{r['key']}".encode()).hexdigest()  # noqa: E731
    recs = sorted(recs, key=rank)
    if a.limit:
        recs = recs[: a.limit]
    n_val = min(N_VAL, len(recs) // 9)
    gen = torch.Generator(device=harness.DEV)
    gen.manual_seed(seed * 1000 + 555)
    stats = collections.Counter()
    rollouts, pids = collect(model, tok, recs, a.k, a.cap, gen, stats, a.batch)
    stats["collect_secs"] = time.time() - t0
    seqs = [pids[t] + g for t, g, _, _ in rollouts]
    pos = [[len(pids[t]) + c - 1 for c in cuts] for t, _, cuts, _ in rollouts]
    hs = hidden_at(model, seqs, pos)
    X = torch.cat(hs)
    y = torch.tensor([o for _, _, cuts, o in rollouts for _ in cuts], dtype=torch.float)
    thm = torch.tensor([t for t, _, cuts, _ in rollouts for _ in cuts])
    depth = torch.tensor([d for _, _, cuts, _ in rollouts for d in range(len(cuts))])
    val = thm < n_val  # the first n_val theorems in hash order
    dev = harness.DEV
    head = Head(X.size(1)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    Xtr, ytr = X[~val].to(dev), y[~val].to(dev)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    for _step in range(a.steps):
        i = torch.randint(0, len(ytr), (4096,), generator=rng).to(dev)
        loss = F.binary_cross_entropy_with_logits(head(Xtr[i]), ytr[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        pv = torch.sigmoid(head(X[val].to(dev))).cpu()
    yv, dv = y[val], depth[val]
    bins = [(lo / 10, (lo + 1) / 10) for lo in range(10)]
    calib = [
        [lo, hi, int(m.sum()), float(yv[m].mean()) if m.any() else None]
        for lo, hi in bins
        for m in [(pv >= lo) & (pv < hi if hi < 1 else pv <= hi)]
    ]
    # Ranking where search uses it: among prefixes of the same theorem and depth.
    same, correct = 0, 0.0
    groups = collections.defaultdict(list)
    for p, lab, t, d in zip(pv.tolist(), yv.tolist(), thm[val].tolist(), dv.tolist(), strict=True):
        groups[(t, d)].append((p, lab))
    for g in groups.values():
        pos_ = [p for p, lab in g if lab]
        neg = [p for p, lab in g if not lab]
        for p in pos_:
            for q in neg:
                same += 1
                correct += 1.0 if p > q else 0.5 if p == q else 0.0
    torch.save(
        {"state": head.state_dict(), "d": X.size(1), "seed": seed},
        a.out.removesuffix(".json") + ".pt",
    )
    res = {
        "args": vars(a),
        "theorems": len(recs),
        "val_theorems": n_val,
        "rollouts": len(rollouts),
        "rollout_success": sum(o for *_, o in rollouts) / len(rollouts),
        "prefixes": len(y),
        "train_prefixes": int((~val).sum()),
        "positive_rate": float(y.mean()),
        "val_auc": auc(pv.tolist(), yv.tolist()),
        "val_auc_same_theorem_depth": correct / max(same, 1),
        "val_pairs_same_theorem_depth": same,
        "val_bce": float(F.binary_cross_entropy(pv.clamp(1e-6, 1 - 1e-6), yv)),
        "val_bce_constant": float(
            F.binary_cross_entropy(torch.full_like(yv, float(ytr.mean())), yv)
        ),
        "calibration": calib,
        "stats": dict(stats),
        "secs": time.time() - t0,
    }
    with open(a.out, "w") as f:
        json.dump(res, f)
    harness.log(
        f"value {a.run}: {len(y)} prefixes from {len(rollouts)} rollouts "
        f"(success {res['rollout_success']:.3f}); val AUC {res['val_auc']:.3f}, "
        f"same theorem+depth {res['val_auc_same_theorem_depth']:.3f}, {res['secs']:.0f}s"
    )


if __name__ == "__main__":
    main()
