#!/usr/bin/env python3
"""One BatchTopK crosscoder for a (base, checkpoint) pair at one residual layer, then its latent
statistics.
A port of Leon's circuit_diffing_20261101/crosscoder.py (branch leon/circuit-diffing) to the long-RL
models: the
model class, the training loop, the evaluation (decoder norms and cosines, firing, energy, Latent
Scaling betas,
splice loss) and the planted-feature control are his; the registry and his sampler are replaced by
explicit paths.

  python crosscoder.py --a v4_untied-ei_s0:pretrain --b v4_untied-ei_s0:rl_round_005 [--layer 3]
  [--plant]

Writes CL_ART/crosscoders/<b>_L<layer>_n<latents>_k<k>_seed<seed>/{crosscoder.pt, meta.json,
eval.json, latents.pt,
train_log.jsonl}. Sources: b_roll = B's own rollouts, a_roll = A's rollouts, pretrain = pretraining
records
(collect.py); 3% of each is held out for evaluation.
"""

import argparse
import json
import random
import time
from pathlib import Path

import common as C
import numpy as np
import torch
import torch.nn.functional as F

CC = C.ART / "crosscoders"
SEQ = C.ART / "seqs"
SOURCES = ("b_roll", "a_roll", "pretrain")
#: Planted-feature positive control (`--plant`), Leon's: (m, a, b, r, magnitude). Feature i fires on
#: a token with id t
#: at position q when (a t + b q) mod m == r, and adds magnitude x v_i (a fixed random unit vector,
#: in scaled units)
#: to model B's activation only. A random function of token and position mod m is nothing the base
#: represents.
PLANT = (
    (29, 7, 3, 0, 2.0),
    (29, 11, 5, 1, 4.0),
    (29, 13, 2, 2, 8.0),
    (331, 17, 3, 3, 2.0),
    (331, 19, 7, 4, 4.0),
    (331, 23, 5, 5, 8.0),
)


class Crosscoder(torch.nn.Module):
    """Shared latents f, one encoder reading both models' activations, one decoder per model."""

    def __init__(self, d, n, k, n_models=2, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        dec = torch.randn(n, n_models, d, generator=g)
        dec = dec / dec.norm(dim=-1, keepdim=True) * 0.1
        self.W_dec = torch.nn.Parameter(dec)
        self.W_enc = torch.nn.Parameter(dec.permute(1, 2, 0).clone())  # (m, d, n)
        self.b_enc = torch.nn.Parameter(torch.zeros(n))
        self.b_dec = torch.nn.Parameter(torch.zeros(n_models, d))
        self.k = k
        self.register_buffer("theta", torch.tensor(float("nan")))

    def dec_norms(self):
        return self.W_dec.norm(dim=-1)  # (n, m)

    def pre(self, a):
        """a: (B, m, d) -> relu pre-activations (B, n)."""
        return F.relu(torch.einsum("bmd,mdn->bn", a, self.W_enc) + self.b_enc)

    def encode(self, a, train=False):
        """-> (f, the smallest selected score of the batch; nan at inference)."""
        p = self.pre(a)
        score = p * self.dec_norms().sum(-1)
        if train or not torch.isfinite(self.theta):
            kk = min(self.k * a.shape[0], score.numel())
            top = torch.topk(score.flatten(), kk, sorted=False)
            mask = torch.zeros_like(score.flatten(), dtype=torch.bool)
            mask[top.indices] = True
            mask = mask.view_as(score) & (score > 0)
            return p * mask, top.values.min().detach()
        return p * (score > self.theta), torch.tensor(float("nan"))

    def decode(self, f):
        return torch.einsum("bn,nmd->bmd", f, self.W_dec) + self.b_dec


def load_seqs(path):
    d = np.load(path)
    t, o = d["tokens"], d["offsets"]
    return [t[o[i] : o[i + 1]] for i in range(len(o) - 1)], d["gen_start"]


def split(n, seed, frac=0.03):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    h = max(1, int(frac * n))
    return sorted(idx[h:]), sorted(idx[:h])


class Acts:
    """Paired activations of models A and B at one layer, for batches of token sequences."""

    def __init__(self, ma, mb, layer, device):
        self.models, self.layer, self.device = (ma, mb), layer, device
        self.scale = torch.ones(2, device=device)
        self.plant_on, self.plant_v = False, None

    def set_plant(self, d, seed=99):
        g = torch.Generator().manual_seed(seed)
        v = torch.randn(len(PLANT), d, generator=g)
        self.plant_v = (v / v.norm(dim=-1, keepdim=True)).to(self.device)
        self.plant_on = True

    @staticmethod
    def indicators(seqs):
        """(N, n_plant) bool, in the token order of `run`'s output."""
        rows = []
        for s in seqs:
            t = s.astype(np.int64)
            q = np.arange(len(s))
            rows.append(np.stack([(a * t + b * q) % m == r for m, a, b, r, _ in PLANT], 1))
        return np.concatenate(rows)

    @torch.no_grad()
    def run(self, seqs, max_tokens=65536):
        """-> (activations (N, 2, d), scaled, in the order of `seqs`; the lengths). Right-padded
        batches: the trunk
        is causal, so padding after a sequence does not touch its positions."""
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        out = [None] * len(seqs)
        i = 0
        while i < len(order):
            j = i
            while j < len(order) and (j - i + 1) * len(seqs[order[j]]) <= max_tokens:
                j += 1
            j = max(j, i + 1)
            batch = [order[q] for q in range(i, j)]
            T = max(len(seqs[q]) for q in batch)
            ids = torch.zeros(len(batch), T, dtype=torch.long)
            for r, q in enumerate(batch):
                ids[r, : len(seqs[q])] = torch.from_numpy(seqs[q].astype(np.int64))
            ids = ids.to(self.device)
            hs = []
            for m in self.models:
                with C.capture(m, [self.layer]) as cap, C.autocast():
                    m(ids)
                hs.append(cap[self.layer].float())
            h = torch.stack(hs, 2)  # (b, T, 2, d)
            for r, q in enumerate(batch):
                out[q] = h[r, : len(seqs[q])]
            i = j
        lens = [len(s) for s in seqs]
        x = torch.cat(out) * self.scale[None, :, None]
        if self.plant_on:
            ind = torch.from_numpy(self.indicators(seqs)).to(self.device).float()
            mag = torch.tensor([p[4] for p in PLANT], device=self.device)
            x[:, 1] += (ind * mag) @ self.plant_v
        return x, lens


def batches(n, bs, gen, device):
    perm = torch.randperm(n, generator=gen).to(device)
    for i in range(0, n - bs + 1, bs):
        yield perm[i : i + bs]


def parse(spec):
    run, stem = spec.split(":")
    return run, stem


def train(a):
    dev = C.DEV
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    run_a, stem_a = parse(a.a)
    run_b, stem_b = parse(a.b)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ma, _ = C.load_model(C.ckpt_path(run_a, stem_a), dev)
    mb, _ = C.load_model(C.ckpt_path(run_b, stem_b), dev)
    paths = {
        "b_roll": SEQ / f"{a.b_roll or run_b + '__' + stem_b}.npz",
        "a_roll": SEQ / f"{run_a}__{stem_a}.npz",
        "pretrain": SEQ / "pretrain.npz",
    }
    train_seqs, held, gen_starts = [], {}, {}
    for si, (src, p) in enumerate(paths.items()):
        s, gs = load_seqs(p)
        tr, ho = split(len(s), 1234 + si)
        held[src], gen_starts[src] = ho, gs
        train_seqs += [s[i] for i in tr]
    n_tok = sum(len(s) for s in train_seqs)
    print(f"train sequences {len(train_seqs)}, {n_tok / 1e6:.2f}M tokens", flush=True)
    acts = Acts(ma, mb, a.layer, dev)
    rng = random.Random(a.seed)
    sample = rng.sample(train_seqs, min(3000, len(train_seqs)))
    x, _ = acts.run(sample)
    d = x.shape[-1]
    acts.scale = (d / (x**2).sum(-1).mean(0)).sqrt()  # E||a_m||^2 = d per model
    mean0 = (x * acts.scale[None, :, None]).mean(0)
    del x
    if a.plant:
        acts.set_plant(d)
    cc = Crosscoder(d, a.n_latents, a.k, seed=a.seed).to(dev)
    with torch.no_grad():
        cc.b_dec.copy_(mean0)
    opt = torch.optim.Adam(cc.parameters(), lr=a.lr, betas=(0.9, 0.999))
    steps = int(a.tokens // a.batch)
    warm, decay0 = min(500, steps // 10), int(0.8 * steps)

    def lr_at(s):
        if s < warm:
            return a.lr * (s + 1) / warm
        if s >= decay0:
            return a.lr * max(0.0, (steps - s) / (steps - decay0))
        return a.lr

    last_fire = torch.zeros(a.n_latents, device=dev)
    thetas = []
    step, seen, t0 = 0, 0, time.time()
    gen = torch.Generator().manual_seed(a.seed)
    order = list(range(len(train_seqs)))
    pos = len(order)
    with open(out / "train_log.jsonl", "w") as log:
        while step < steps:
            buf_seqs, tok = [], 0
            while tok < a.buffer:
                if pos >= len(order):
                    rng.shuffle(order)
                    pos = 0
                s = train_seqs[order[pos]]
                pos += 1
                buf_seqs.append(s)
                tok += len(s)
            buf, _ = acts.run(buf_seqs)
            for idx in batches(buf.shape[0], a.batch, gen, dev):
                if step >= steps:
                    break
                x = buf[idx]
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                f, tmin = cc.encode(x, train=True)
                xh = cc.decode(f)
                err = x - xh
                mse = (err**2).sum(-1).sum(-1).mean()
                fired = (f > 0).any(0)
                last_fire += x.shape[0]
                last_fire[fired] = 0
                dead = last_fire > a.dead_tokens
                aux = torch.zeros((), device=dev)
                if dead.sum() > 0:  # AuxK: dead latents reconstruct the residual error
                    p = cc.pre(x)[:, dead]
                    ka = min(a.k_aux, int(dead.sum()))
                    tv, ti = torch.topk(p, ka, dim=-1)
                    fa = torch.zeros_like(p).scatter_(-1, ti, tv)
                    xa = torch.einsum("bn,nmd->bmd", fa, cc.W_dec[dead])
                    aux = ((err.detach() - xa) ** 2).sum(-1).sum(-1).mean()
                loss = mse + aux / 32
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(cc.parameters(), 1.0)
                opt.step()
                if step >= decay0:
                    thetas.append(float(tmin))
                seen += x.shape[0]
                if step % 500 == 0 or step == steps - 1:
                    fvu = float(
                        ((err**2).sum((0, 2)) / ((x - x.mean(0)) ** 2).sum((0, 2))).mean().detach()
                    )
                    row = dict(
                        step=step,
                        tokens=seen,
                        mse=float(mse.detach()),
                        aux=float(aux.detach()),
                        fvu=fvu,
                        dead=float(dead.float().mean()),
                        l0=float((f > 0).float().sum(-1).mean()),
                        secs=time.time() - t0,
                    )
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    print(row, flush=True)
                step += 1
            del buf
    cc.theta.fill_(float(np.mean(thetas)) if thetas else float("nan"))
    meta = dict(
        a=a.a,
        b=a.b,
        rl_secs_a=C.rl_seconds(run_a, stem_a),
        rl_secs_b=C.rl_seconds(run_b, stem_b),
        layer=a.layer,
        n_latents=a.n_latents,
        k=a.k,
        seed=a.seed,
        tokens=seen,
        steps=steps,
        lr=a.lr,
        batch=a.batch,
        scale=acts.scale.tolist(),
        theta=float(cc.theta),
        sources={k: str(v) for k, v in paths.items()},
        plant=[list(p) for p in PLANT] if a.plant else None,
        train_secs=time.time() - t0,
    )
    torch.save({"state": cc.state_dict(), "meta": meta, "held_out": held}, out / "crosscoder.pt")
    (out / "meta.json").write_text(json.dumps(meta))
    return evaluate(cc, acts, paths, held, gen_starts, out, a)


@torch.no_grad()
def evaluate(cc, acts, paths, held, gen_starts, out, a):
    dev = acts.device
    n = cc.W_dec.shape[0]
    D = cc.W_dec  # (n, 2, d)
    norms = cc.dec_norms()  # (n, 2)
    res = {"per_source": {}}
    z = lambda *shape: torch.zeros(*shape, device=dev, dtype=torch.float64)  # noqa: E731
    acc = {
        "ff": z(n),
        "fire": {s: z(n) for s in SOURCES},
        "fire_gen": {s: z(n) for s in SOURCES},
        "energy": {s: z(n, 2) for s in SOURCES},
        "f_sum": {s: z(n) for s in SOURCES},
    }
    # Latent Scaling: sum_x f_j <d_j^ref, y_m(x)> for ref in (A, B), m in (A, B), y in (err, rec)
    ls = {(ref, m, y): z(n) for ref in (0, 1) for m in (0, 1) for y in ("err", "rec")}
    tot = {"err": torch.zeros(2, device=dev), "var": torch.zeros(2, device=dev), "N": 0}
    pl = None
    if acts.plant_on:
        cosv = F.normalize(D[:, 1], dim=-1) @ acts.plant_v.T  # (n, n_plant)
        cand = cosv.abs().topk(3, dim=0).indices  # (3, n_plant)
        pl = {
            "cand": cand,
            "both": z(3, len(PLANT)),
            "fire": z(3, len(PLANT)),
            "ind": z(len(PLANT)),
        }
    n_tok = {s: 0 for s in SOURCES}
    n_gen = {s: 0 for s in SOURCES}
    for src, p in paths.items():
        seqs, _ = load_seqs(p)
        ho = held[src]
        hs = [seqs[i] for i in ho]
        gs = gen_starts[src][ho]
        se = {"err": torch.zeros(2, device=dev), "var": torch.zeros(2, device=dev), "N": 0}
        x_all, lens = acts.run(hs)
        ind_all = torch.from_numpy(acts.indicators(hs)).to(dev) if pl is not None else None
        is_gen = torch.cat(
            [torch.arange(L, device=dev) >= int(g) for L, g in zip(lens, gs, strict=True)]
        )
        mu = x_all.mean(0)
        for i in range(0, x_all.shape[0], 8192):
            x = x_all[i : i + 8192]
            gm = is_gen[i : i + 8192]
            f, _ = cc.encode(x)
            xh = cc.decode(f)
            err = x - xh
            se["err"] += (err**2).sum((0, 2))
            se["var"] += ((x - mu) ** 2).sum((0, 2))
            se["N"] += x.shape[0]
            acc["ff"] += (f.double() ** 2).sum(0)
            acc["fire"][src] += (f > 0).double().sum(0)
            acc["fire_gen"][src] += (f[gm] > 0).double().sum(0)
            acc["f_sum"][src] += f.double().sum(0)
            acc["energy"][src] += (f.double() ** 2)[:, :, None].sum(0) * (norms.double() ** 2)
            rec = torch.einsum("bn,nmd->bmd", f, D)  # without b_dec
            for ref in (0, 1):
                dref = D[:, ref]
                for m in (0, 1):
                    ls[(ref, m, "err")] += (f * (err[:, m] @ dref.T)).double().sum(0)
                    ls[(ref, m, "rec")] += (f * (rec[:, m] @ dref.T)).double().sum(0)
            if pl is not None:
                ib = ind_all[i : i + 8192].double()  # (b, n_plant)
                fc = (f[:, pl["cand"]] > 0).double()  # (b, 3, n_plant)
                pl["both"] += (fc * ib[:, None, :]).sum(0)
                pl["fire"] += fc.sum(0)
                pl["ind"] += ib.sum(0)
            n_tok[src] += x.shape[0]
            n_gen[src] += int(gm.sum())
        res["per_source"][src] = {
            "fvu": (se["err"] / se["var"]).tolist(),
            "var": se["var"].tolist(),
            "tokens": se["N"],
            "gen_tokens": n_gen[src],
        }
        for key in ("err", "var"):
            tot[key] += se[key]
        tot["N"] += se["N"]
        del x_all
    res["fvu"] = (tot["err"] / tot["var"]).tolist()
    ff = acc["ff"]
    dd = norms.double() ** 2
    cross = (D[:, 0] * D[:, 1]).sum(-1).double()
    beta = {}
    for ref in (0, 1):
        for m in (0, 1):
            own = dd[:, ref] if m == ref else cross
            denom = dd[:, ref] * ff
            beta[f"err_ref{ref}_m{m}"] = (ls[(ref, m, "err")] + own * ff) / denom
            beta[f"rec_ref{ref}_m{m}"] = ls[(ref, m, "rec")] / denom
    lat = {
        "norm_a": norms[:, 0].tolist(),
        "norm_b": norms[:, 1].tolist(),
        "cos": F.cosine_similarity(D[:, 0], D[:, 1], dim=-1).tolist(),
        "ff": ff.tolist(),
        **{f"fire_{s}": (acc["fire"][s] / max(1, n_tok[s])).tolist() for s in SOURCES},
        **{f"fire_gen_{s}": (acc["fire_gen"][s] / max(1, n_gen[s])).tolist() for s in SOURCES},
        **{f"energy_a_{s}": acc["energy"][s][:, 0].tolist() for s in SOURCES},
        **{f"energy_b_{s}": acc["energy"][s][:, 1].tolist() for s in SOURCES},
        **{k: v.tolist() for k, v in beta.items()},
    }
    if pl is not None:
        cosv = F.normalize(D[:, 1], dim=-1) @ acts.plant_v.T
        det = []
        for i, spec in enumerate(PLANT):
            rows = []
            for c in range(3):
                j = int(pl["cand"][c, i])
                both, fire, ind = (
                    float(pl["both"][c, i]),
                    float(pl["fire"][c, i]),
                    float(pl["ind"][i]),
                )
                na_, nb_ = float(norms[j, 0]), float(norms[j, 1])
                rows.append(
                    dict(
                        latent=j,
                        cos_b=float(cosv[j, i]),
                        dnorm=0.5 * ((na_ - nb_) / max(na_, nb_) + 1),
                        precision=both / fire if fire else None,
                        recall=both / ind if ind else None,
                    )
                )
            det.append(
                dict(
                    spec=list(spec), freq=float(pl["ind"][i]) / sum(n_tok.values()), candidates=rows
                )
            )
        res["planted"] = det
    res["n_tokens"] = n_tok
    res["var_total"] = tot["var"].tolist()
    res["loss"] = splice_loss(cc, acts, paths, held, gen_starts, a.n_held)
    res["l0"] = float(sum(acc["fire"][s].sum() for s in SOURCES) / max(1, sum(n_tok.values())))
    res["dead_frac"] = float((sum(acc["fire"][s] for s in SOURCES) == 0).float().mean())
    (out / "eval.json").write_text(json.dumps(res, indent=1))
    torch.save(lat, out / "latents.pt")
    print(f"eval {json.dumps({k: res[k] for k in ('fvu', 'l0', 'dead_frac')})}")
    print(f"loss {json.dumps(res['loss'])}", flush=True)
    return res


@torch.no_grad()
def splice_loss(cc, acts, paths, held, gen_starts, n=1500):
    """Next-token loss on the sampled tokens of held-out rollouts, per model: clean, with the layer
    replaced by the
    crosscoder reconstruction, and by the mean activation. `recovered` = (mean - recon) / (mean -
    clean)."""
    dev = acts.device
    out = {}
    for src in ("b_roll", "a_roll"):
        seqs, _ = load_seqs(paths[src])
        gs = gen_starts[src]
        ho = held[src][:n]
        sums = {f"m{m}_{c}": 0.0 for m in (0, 1) for c in ("clean", "recon", "mean")}
        cnt, mean_act = 0, None
        for i in range(0, len(ho), 64):
            batch = ho[i : i + 64]
            T = max(len(seqs[q]) for q in batch)
            ids = torch.zeros(len(batch), T, dtype=torch.long)
            lm = torch.zeros(len(batch), T, dtype=torch.bool)
            for r, q in enumerate(batch):
                s = seqs[q]
                ids[r, : len(s)] = torch.from_numpy(s.astype(np.int64))
                lm[r, int(gs[q]) - 1 : len(s) - 1] = True  # positions that predict sampled ids
            ids, lm = ids.to(dev), lm.to(dev)
            hs = []
            for m in acts.models:
                with C.capture(m, [acts.layer]) as cap, C.autocast():
                    m(ids)
                hs.append(cap[acts.layer].float())
            x = torch.stack(hs, 2) * acts.scale[None, None, :, None]  # (b, T, 2, d)
            b_, T_ = x.shape[:2]
            f, _ = cc.encode(x.view(b_ * T_, 2, -1))
            xh = cc.decode(f).view(b_, T_, 2, -1)
            if mean_act is None:
                mean_act = x[lm].mean(0)
            tgt = ids[:, 1:]
            for m, model in enumerate(acts.models):
                for cond, rep in (
                    ("clean", None),
                    ("recon", xh[:, :, m] / acts.scale[m]),
                    ("mean", (mean_act[m] / acts.scale[m]).expand(b_, T_, -1)),
                ):
                    hook = None
                    if rep is not None:
                        hook = model.blocks[acts.layer - 1].register_forward_hook(
                            lambda mod, inp, o, _r=rep: _r.to(o.dtype)
                        )
                    try:
                        with C.autocast():
                            logits = model(ids).float()
                    finally:
                        if hook is not None:
                            hook.remove()
                    ce = F.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.shape[-1]),
                        tgt.reshape(-1),
                        reduction="none",
                    ).view(b_, T_ - 1)
                    sums[f"m{m}_{cond}"] += float(ce[lm[:, :-1]].sum())
            cnt += int(lm[:, :-1].sum())
        r = {k: v / max(cnt, 1) for k, v in sums.items()}
        for m in (0, 1):
            c, rc, mn = r[f"m{m}_clean"], r[f"m{m}_recon"], r[f"m{m}_mean"]
            r[f"m{m}_recovered"] = (mn - rc) / (mn - c) if mn > c else float("nan")
        r["tokens"] = cnt
        out[src] = r
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="run:stem of the base")
    ap.add_argument("--b", required=True, help="run:stem of the RL checkpoint")
    ap.add_argument(
        "--b_roll",
        default=None,
        help="npz stem whose rollouts stand in for B's (the null: A's own)",
    )
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--n_latents", type=int, default=8192)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--k_aux", type=int, default=256)
    ap.add_argument("--dead_tokens", type=float, default=2e6)
    ap.add_argument("--tokens", type=float, default=40e6)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--buffer", type=int, default=2_000_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plant", action="store_true", help="planted-feature positive control")
    ap.add_argument("--n_held", type=int, default=1500, help="splice-loss sequences per source")
    ap.add_argument("--tag", default="", help="suffix on the output directory")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.out is None:
        a.out = str(
            CC / f"{a.b.replace(':', '__')}_L{a.layer}_n{a.n_latents}_k{a.k}_seed{a.seed}{a.tag}"
        )
    if (Path(a.out) / "eval.json").exists():
        print(f"exists: {a.out}")
        return 0
    print(f"start {vars(a)}", flush=True)
    train(a)
    print(f"done {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
