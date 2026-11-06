#!/usr/bin/env python3
"""SAE features of the pretrain-to-RL change: steer the pretrain with n features, or remove n features from the RL model.

  python run_sae.py --seed 0 --round 16 --block 4 [--limit 8 --rollouts 40 --steps 200 --k 4]

One BatchTopK SAE per (round, block), trained on activations of both models on the same token sequences. The sequences
come from rollouts on Dan's generated held-out theorems, never the textbook problems scored here. Latents are ranked
by E[(z_r - z_0)^2]. Every edit adds a decoded difference and keeps the SAE's reconstruction error, so an empty edit
is the unmodified model. Arms, all edited at every token position:
  steer_mean  pretrain + sum_S mean(z_r - z_0) d
  steer_map   pretrain with each z in S moved by a per-latent least-squares map z_0 -> z_r
  ablate_map  round r with each z in S moved by the reverse map z_r -> z_0 (gap lost as n grows)
  *_rand      the same with n random live latents (controls), and meandiff: the raw activation mean difference
"""
import argparse
import json
import random
import time

import torch
import torch.nn.functional as F

import common as C
import state_sample


@torch.no_grad()
def rollouts(model, tok, prompts, seed):
    """Sampled token sequences (prompt + action) from env rollouts, without Lean."""
    seqs, real = [], state_sample.generate_ids_fast

    def spy(m, t, pids, **kw):
        outs = real(m, t, pids, **kw)
        for p, o in zip(pids, outs):
            o = o.tolist() if hasattr(o, "tolist") else list(o)
            act = []
            for x in o:
                if x == t.pad:
                    break
                act.append(x)
                if x == t.eos:
                    break
            seqs.append((list(p), act))
        return outs

    state_sample.generate_ids_fast = spy
    try:
        state_sample.env_generate(model, tok, prompts, greedy=False, temperature=1.0, max_action=256, max_steps=48,
                                  batch=1024, seed=seed, stats={}, gate=False)
    finally:
        state_sample.generate_ids_fast = real
    return seqs


@torch.no_grad()
def paired_acts(m0, m1, block, seqs, tok, per_seq, gen, max_len=1000):
    """Activations after `block` of both models at sampled positions of each sequence. Half the sample is action tokens."""
    store = {0: [], 1: []}
    cap = {}
    hooks = [m.blocks[block].register_forward_hook(lambda _m, _i, o, j=j: cap.__setitem__(j, o))
             for j, m in enumerate((m0, m1))]
    seqs = sorted(seqs, key=lambda s: len(s[0]) + len(s[1]))
    for i in range(0, len(seqs), 128):
        chunk = seqs[i:i + 128]
        L = min(max_len, max(len(p) + len(a) for p, a in chunk))
        idx = torch.full((len(chunk), L), tok.pad, dtype=torch.long, device=C.DEVICE)
        rows, cols = [], []
        for b, (p, a) in enumerate(chunk):
            s = (p + a)[:L]
            idx[b, :len(s)] = torch.tensor(s, device=C.DEVICE)
            na, np_ = max(0, len(s) - len(p)), min(len(p), len(s))
            pick = random.Random(int(torch.randint(1 << 30, (1,), generator=gen))).sample
            cols_a = [np_ + j for j in pick(range(na), min(na, per_seq // 2))]
            cols_p = pick(range(np_), min(np_, per_seq - len(cols_a)))
            for c in cols_a + cols_p:
                rows.append(b)
                cols.append(c)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=C.DEVICE == "cuda"):
            m0(idx), m1(idx)
        for j in (0, 1):
            store[j].append(cap[j][rows, cols].float().cpu())
    for h in hooks:
        h.remove()
    return torch.cat(store[0]), torch.cat(store[1])


class SAE(torch.nn.Module):
    def __init__(self, d, n, k, center, scale):
        super().__init__()
        self.k, self.scale = k, scale
        self.register_buffer("center", center)
        dec = F.normalize(torch.randn(n, d), dim=1)
        self.dec = torch.nn.Parameter(dec.clone())
        self.enc = torch.nn.Parameter(dec.clone())
        self.register_buffer("theta", torch.zeros(()))

    def pre(self, x):
        return (x - self.center) / self.scale @ self.enc.T

    def batch_topk(self, a):
        a = F.relu(a)
        keep = torch.topk(a.flatten(), self.k * a.size(0)).indices
        z = torch.zeros_like(a).flatten()
        z[keep] = a.flatten()[keep]
        return z.view_as(a)

    def encode(self, x):
        return F.relu(self.pre(x)) * (self.pre(x) > self.theta)

    def decode_scaled(self, z):
        return z @ self.dec


def train_sae(x0, x1, n_latents, k, steps, seed):
    torch.manual_seed(seed)
    x = torch.cat((x0, x1)).to(C.DEVICE)
    center = x.mean(0)
    scale = (x - center).norm(dim=1).mean().item() / x.size(1) ** 0.5  # unit per-dimension RMS
    sae = SAE(x.size(1), n_latents, k, center, scale).to(C.DEVICE)
    opt = torch.optim.Adam(sae.parameters(), lr=3e-4)
    target = (x - center) / scale
    fired = torch.zeros(n_latents, device=C.DEVICE)
    for step in range(steps):
        i = torch.randint(x.size(0), (4096,), device=C.DEVICE)
        z = sae.batch_topk(sae.pre(x[i]))
        loss = F.mse_loss(sae.decode_scaled(z), target[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            sae.dec.data = F.normalize(sae.dec.data, dim=1)
            if step >= steps // 2:
                fired += (z > 0).sum(0)
    with torch.no_grad():  # the global threshold that keeps k latents per position on average
        i = torch.randint(x.size(0), (8192,), device=C.DEVICE)
        a = F.relu(sae.pre(x[i])).flatten()
        sae.theta.fill_(torch.topk(a, k * i.numel()).values[-1])
        zz = sae.encode(x[i])
        rec = sae.decode_scaled(zz)
        fve = 1 - (rec - target[i]).pow(2).sum() / target[i].pow(2).sum()
        info = {"fve": fve.item(), "l0": (zz > 0).sum(1).float().mean().item(), "live": int((fired > 0).sum())}
    return sae, info


@torch.no_grad()
def latent_stats(sae, x0, x1):
    """Per-latent mean difference, rank score and both least-squares maps, from paired positions."""
    n = sae.dec.size(0)
    s = {k: torch.zeros(n, device=C.DEVICE) for k in ("z0", "z1", "z00", "z11", "z01", "sq")}
    N = 0
    for i in range(0, x0.size(0), 8192):
        a, b = sae.encode(x0[i:i + 8192].to(C.DEVICE)), sae.encode(x1[i:i + 8192].to(C.DEVICE))
        s["z0"] += a.sum(0); s["z1"] += b.sum(0); s["z00"] += (a * a).sum(0); s["z11"] += (b * b).sum(0)
        s["z01"] += (a * b).sum(0); s["sq"] += (b - a).pow(2).sum(0)
        N += a.size(0)
    s = {k: v / N for k, v in s.items()}
    cov = s["z01"] - s["z0"] * s["z1"]
    fwd = cov / (s["z00"] - s["z0"] ** 2).clamp(min=1e-8)
    back = cov / (s["z11"] - s["z1"] ** 2).clamp(min=1e-8)
    return {"delta": s["z1"] - s["z0"], "score": s["sq"], "fwd_a": fwd, "fwd_b": s["z1"] - fwd * s["z0"],
            "back_a": back, "back_b": s["z0"] - back * s["z1"]}


class Edit:
    """Forward hook on a block. mode constant: add `vec`. mode map: move latents S by z -> a z + b."""

    def __init__(self, sae=None, S=None, a=None, b=None, vec=None):
        self.vec = vec
        if S is not None:
            self.enc, self.dec, self.a, self.b = sae.enc[S], sae.dec[S], a[S], b[S]
            self.center, self.scale, self.theta = sae.center, sae.scale, sae.theta

    def __call__(self, _m, _i, x):
        if self.vec is not None:
            return x + self.vec.to(x.dtype)
        y = x.float()
        act = (y - self.center) / self.scale @ self.enc.T
        z = F.relu(act) * (act > self.theta)
        return (y + ((self.a - 1) * z + self.b) @ self.dec * self.scale).to(x.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--ns", type=int, nargs="+", default=[1, 4, 16, 64, 256, 1024, 8192])
    ap.add_argument("--latents", type=int, default=8192)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--rollouts", type=int, default=12000)
    ap.add_argument("--per_seq", type=int, default=16)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--train_set", default=str(C.FORK / "data/p2/heldout.jsonl"))
    a = ap.parse_args()
    out = C.RESULTS / "sae" / f"s{a.seed}_r{a.round}_b{a.block}"
    if C.done(out / "DONE"):
        return
    recs = C.problems(a.limit)
    m0, tok, _ = C.load(a.seed, 0)
    m1, _, _ = C.load(a.seed, a.round)
    pool = [json.loads(l)["prompt"] for l in open(a.train_set) if l.strip()]
    names = {r["prompt"] for r in C.problems()}
    pool = [p for p in pool if p not in names]
    rng = random.Random(a.seed)
    prompts = [rng.choice(pool) for _ in range(a.rollouts)]
    t = time.time()
    half = len(prompts) // 2
    seqs = rollouts(m0, tok, prompts[:half], a.seed) + rollouts(m1, tok, prompts[half:], a.seed + 1)
    gen = torch.Generator().manual_seed(a.seed)
    x0, x1 = paired_acts(m0, m1, a.block, seqs, tok, a.per_seq, gen)
    print(f"{len(seqs)} sequences, {x0.size(0)} positions in {time.time() - t:.0f}s", flush=True)
    t = time.time()
    sae, info = train_sae(x0, x1, a.latents, a.topk, a.steps, a.seed)
    st = latent_stats(sae, x0, x1)
    order = st["score"].argsort(descending=True)
    live = (st["score"] > 0).nonzero().flatten()
    meandiff = (x1 - x0).mean(0).to(C.DEVICE)
    info.update(seqs=len(seqs), positions=x0.size(0), train_s=time.time() - t, live_scored=len(live),
                train_set=a.train_set, meandiff_rel=(meandiff.norm() / x0.norm(dim=1).mean()).item())
    print(json.dumps(info), flush=True)
    C.save(out / "sae_info.json", info)
    torch.save({"enc": sae.enc.detach().cpu(), "dec": sae.dec.detach().cpu(), "center": sae.center.cpu(), "scale": sae.scale,
                "theta": sae.theta.cpu(), "order": order.cpu(), **{k: v.cpu() for k, v in st.items()}}, out / "sae.pt")

    def run(tag, model, edit, **meta):
        path = out / f"{tag}.json"
        if C.done(path):
            return
        h = model.blocks[a.block].register_forward_hook(edit) if edit else None
        t0 = time.time()
        res = C.evaluate(model, tok, recs, a.k, a.batch)
        if h:
            h.remove()
        C.save(path, {**meta, "seed": a.seed, "round": a.round, "block": a.block, "k": a.k, "wall_s": time.time() - t0, **res})
        print(f"{tag}: solved-any {sum(r['n_ok'] > 0 for r in res['rows'])}/{len(recs)} in {time.time() - t0:.0f}s", flush=True)

    run("ref_r0", m0, None, ref=True)
    run("ref_r", m1, None, ref=True)
    run("meandiff", m0, Edit(vec=meandiff), arm="meandiff")
    one, zero = torch.ones_like(st["delta"]), torch.zeros_like(st["delta"])
    rand = live[torch.randperm(len(live), generator=torch.Generator().manual_seed(a.seed)).to(live.device)]
    for n in a.ns:
        S = order[:n]
        run(f"steer_mean_n{n}", m0, Edit(sae, S, one, st["delta"]), arm="steer_mean", n=n)
        run(f"steer_map_n{n}", m0, Edit(sae, S, st["fwd_a"], st["fwd_b"]), arm="steer_map", n=n)
        run(f"ablate_map_n{n}", m1, Edit(sae, S, st["back_a"], st["back_b"]), arm="ablate_map", n=n)
    for n in (64, 1024):
        S = rand[:n]
        run(f"steer_map_rand_n{n}", m0, Edit(sae, S, st["fwd_a"], st["fwd_b"]), arm="steer_map_rand", n=n)
        run(f"ablate_map_rand_n{n}", m1, Edit(sae, S, st["back_a"], st["back_b"]), arm="ablate_map_rand", n=n)
    (out / "DONE").touch()


if __name__ == "__main__":
    main()
