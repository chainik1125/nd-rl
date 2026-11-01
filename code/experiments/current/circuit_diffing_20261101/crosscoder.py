"""Stage 1: a BatchTopK crosscoder on one residual-stream layer of two models (A = base, B = an
RL checkpoint), trained on paired activations of identical token sequences.

    python -m code.experiments.current.circuit_diffing_20261101.crosscoder --gpu 0 \
        --a base:0 --b L3:0 --layer 3 [--n_latents 8192 --k 32 --tokens 40e6 --seed 0]

Data: B's own rollouts, A's rollouts and the Stage-1 pretraining sample (`rollouts.py`); 3% of
the sequences of each source are held out for evaluation. Activations of both models (fp32, the
residual stream after block `layer`) are computed on the fly into a shuffled GPU buffer. Each
model's activations are scaled by one scalar so that E||a||^2 = d (Lindsey et al. 2024).

Crosscoder (Lindsey et al. 2024; BatchTopK as in Bussmann et al. 2024 and Minder et al. 2025):
  pre = sum_m a_m W_enc[m] + b_enc;  f = BatchTopK(relu(pre)) over the batch, selecting on
  relu(pre_j) * (||d_j^A|| + ||d_j^B||) so that the choice does not depend on how a latent's
  scale is split between f and d;  a_m_hat = f W_dec[:, m] + b_dec[m];
  loss = sum_m ||a_m - a_m_hat||^2 + aux / 32 (AuxK on latents dead for 2M tokens, k_aux 256).
At inference a latent is on when its selection score exceeds theta, the mean over the last 20% of
training of each batch's smallest selected score.

Evaluation on the held-out sequences (`eval.json`): per model FVU and next-token loss on sampled
tokens with the layer spliced by the reconstruction (and by the mean, for loss recovered);
L0, dead latents; per latent decoder norms, decoder cosine, firing rates per source, energy, and
the Latent Scaling betas of Minder et al. 2025 (`latent_scaling`).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .common import ART, assert_gpu_free, capture, load_model, pt_now, registry

SEQ = ART / "seqs"
CC = ART / "crosscoders"
SOURCES = ("b_roll", "a_roll", "pretrain")
#: Planted-feature positive control (`--plant`): (m, a, b, r, magnitude). Feature i fires on a
#: token with id t at position q when (a t + b q) mod m == r, and adds magnitude x v_i (a fixed
#: random unit vector, in scaled units) to model B's activation only. A random function of token
#: and position mod m is not something the base can represent linearly.
PLANT = (
    (29, 7, 3, 0, 2.0),
    (29, 11, 5, 1, 4.0),
    (29, 13, 2, 2, 8.0),
    (331, 17, 3, 3, 2.0),
    (331, 19, 7, 4, 4.0),
    (331, 23, 5, 5, 8.0),
)


class Crosscoder(torch.nn.Module):
    def __init__(self, d: int, n: int, k: int, n_models: int = 2, seed: int = 0) -> None:
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

    def dec_norms(self) -> torch.Tensor:
        return self.W_dec.norm(dim=-1)  # (n, m)

    def pre(self, a: torch.Tensor) -> torch.Tensor:
        """a: (B, m, d) -> relu pre-activations (B, n)."""
        return F.relu(torch.einsum("bmd,mdn->bn", a, self.W_enc) + self.b_enc)

    def encode(self, a: torch.Tensor, train: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
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

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bn,nmd->bmd", f, self.W_dec) + self.b_dec


def load_seqs(path: Path) -> list[np.ndarray]:
    d = np.load(path)
    t, o = d["tokens"], d["offsets"]
    return [t[o[i] : o[i + 1]] for i in range(len(o) - 1)]


def load_meta(path: Path) -> dict[str, np.ndarray]:
    d = np.load(path)
    return {k: d[k] for k in d.files if k not in ("tokens",)}


def split(n: int, seed: int, frac: float = 0.03) -> tuple[list[int], list[int]]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    h = max(1, int(frac * n))
    return sorted(idx[h:]), sorted(idx[:h])


class Acts:
    """Paired activations of models A and B at one layer, for batches of token sequences."""

    def __init__(self, ma: Any, mb: Any, layer: int, device: str) -> None:
        self.models, self.layer, self.device = (ma, mb), layer, device
        self.scale = torch.ones(2, device=device)
        self.plant_on = False
        self.plant_v: torch.Tensor | None = None

    def set_plant(self, d: int, seed: int = 99) -> None:
        g = torch.Generator().manual_seed(seed)
        v = torch.randn(len(PLANT), d, generator=g)
        self.plant_v = (v / v.norm(dim=-1, keepdim=True)).to(self.device)
        self.plant_on = True

    @staticmethod
    def indicators(seqs: list[np.ndarray]) -> np.ndarray:
        """(N, n_plant) bool, in the token order of `run`'s output."""
        rows = []
        for s in seqs:
            t = s.astype(np.int64)
            q = np.arange(len(s))
            rows.append(np.stack([(a * t + b * q) % m == r for m, a, b, r, _ in PLANT], 1))
        return np.concatenate(rows)

    @torch.no_grad()
    def run(
        self, seqs: list[np.ndarray], max_tokens: int = 65536
    ) -> tuple[torch.Tensor, list[int]]:
        """-> (activations (N, 2, d), scaled; the order of `seqs` kept; the lengths)."""
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        out: list[torch.Tensor | None] = [None] * len(seqs)
        i = 0
        while i < len(order):
            L = len(seqs[order[i]])
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
                with capture(m, [self.layer]) as cap:
                    m.features(ids)
                hs.append(cap[self.layer])
            h = torch.stack(hs, 2)  # (b, T, 2, d)
            for r, q in enumerate(batch):
                out[q] = h[r, : len(seqs[q])]
            i = j
            del L
        lens = [len(s) for s in seqs]
        x = torch.cat([x for x in out if x is not None]) * self.scale[None, :, None]
        if self.plant_on:
            assert self.plant_v is not None
            ind = torch.from_numpy(self.indicators(seqs)).to(self.device).float()
            mag = torch.tensor([p[4] for p in PLANT], device=self.device)
            x[:, 1] += (ind * mag) @ self.plant_v
        return x, lens


def batches(n: int, bs: int, gen: torch.Generator, device: str):
    perm = torch.randperm(n, generator=gen).to(device)
    for i in range(0, n - bs + 1, bs):
        yield perm[i : i + bs]


def train(a: argparse.Namespace) -> dict[str, Any]:
    dev = a.device
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    reg = registry()
    ca = reg[(a.a.split(":")[0], int(a.a.split(":")[1]))]
    cb = reg[(a.b.split(":")[0], int(a.b.split(":")[1]))]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ma, _, _ = load_model(ca.path, dev)
    mb, _, _ = load_model(cb.path, dev)
    paths = {
        "b_roll": SEQ / f"{cb.label}_s{cb.seed}.npz",
        "a_roll": SEQ / f"{ca.label}_s{ca.seed}.npz",
        "pretrain": SEQ / "pretrain.npz",
    }
    train_seqs: list[np.ndarray] = []
    held: dict[str, list[int]] = {}
    for si, (src, p) in enumerate(paths.items()):
        s = load_seqs(p)
        tr, ho = split(len(s), 1234 + si)
        held[src] = ho
        train_seqs += [s[i] for i in tr]
    n_tok = sum(len(s) for s in train_seqs)
    print(f"{pt_now()} train sequences {len(train_seqs)}, {n_tok / 1e6:.2f}M tokens", flush=True)
    acts = Acts(ma, mb, a.layer, dev)
    rng = random.Random(a.seed)
    # scale: E||a_m||^2 = d on a sample
    sample = rng.sample(train_seqs, min(3000, len(train_seqs)))
    x, _ = acts.run(sample)
    d = x.shape[-1]
    acts.scale = (d / (x**2).sum(-1).mean(0)).sqrt()
    mean0 = (x * acts.scale[None, :, None]).mean(0)
    del x
    if a.plant:
        acts.set_plant(d)
    cc = Crosscoder(d, a.n_latents, a.k, seed=a.seed).to(dev)
    with torch.no_grad():
        cc.b_dec.copy_(mean0)
    opt = torch.optim.Adam(cc.parameters(), lr=a.lr, betas=(0.9, 0.999))
    steps = int(a.tokens // a.batch)
    warm, decay0 = 500, int(0.8 * steps)

    def lr_at(s: int) -> float:
        if s < warm:
            return a.lr * (s + 1) / warm
        if s >= decay0:
            return a.lr * max(0.0, (steps - s) / (steps - decay0))
        return a.lr

    last_fire = torch.zeros(a.n_latents, device=dev)  # tokens since the latent last fired
    thetas: list[float] = []
    step, seen, t0 = 0, 0, time.time()
    gen = torch.Generator().manual_seed(a.seed)
    log = open(out / "train_log.jsonl", "w")  # noqa: SIM115 (closed after the loop)
    order = list(range(len(train_seqs)))
    pos = len(order)
    while step < steps:
        # fill a buffer of about a.buffer tokens
        buf_seqs = []
        tok = 0
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
            if dead.sum() > 0:
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
                print(f"{pt_now()} {row}", flush=True)
            step += 1
        del buf
    log.close()
    cc.theta.fill_(float(np.mean(thetas)))
    meta = dict(
        a=a.a,
        b=a.b,
        a_path=str(ca.path),
        b_path=str(cb.path),
        R_b=cb.R,
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
        held_out=held,
        sources={k: str(v) for k, v in paths.items()},
        plant=[list(p) for p in PLANT] if a.plant else None,
        train_secs=time.time() - t0,
    )
    torch.save({"state": cc.state_dict(), "meta": meta}, out / "crosscoder.pt")
    (out / "meta.json").write_text(json.dumps({k: v for k, v in meta.items() if k != "held_out"}))
    res = evaluate(cc, acts, paths, held, out, a)
    return res


@torch.no_grad()
def evaluate(
    cc: Crosscoder,
    acts: Acts,
    paths: dict[str, Path],
    held: dict[str, list[int]],
    out: Path,
    a: argparse.Namespace,
) -> dict[str, Any]:
    dev = acts.device
    n = cc.W_dec.shape[0]
    D = cc.W_dec  # (n, 2, d)
    norms = cc.dec_norms()  # (n, 2)
    res: dict[str, Any] = {"per_source": {}}
    acc = {
        "ff": torch.zeros(n, device=dev, dtype=torch.float64),
        "fire": {s: torch.zeros(n, device=dev, dtype=torch.float64) for s in SOURCES},
        "fire_gen": {s: torch.zeros(n, device=dev, dtype=torch.float64) for s in SOURCES},
        "energy": {s: torch.zeros(n, 2, device=dev, dtype=torch.float64) for s in SOURCES},
        "f_sum": {s: torch.zeros(n, device=dev, dtype=torch.float64) for s in SOURCES},
    }
    # Latent Scaling: sum_x f_j <d_j^ref, y_m(x)> for ref in (A, B), m in (A, B), y in (err, rec)
    ls = {
        (ref, m, y): torch.zeros(n, device=dev, dtype=torch.float64)
        for ref in (0, 1)
        for m in (0, 1)
        for y in ("err", "rec")
    }
    tot = {"err": torch.zeros(2, device=dev), "var": torch.zeros(2, device=dev), "N": 0}
    pl = None
    if acts.plant_on:
        assert acts.plant_v is not None
        cosv = F.normalize(D[:, 1], dim=-1) @ acts.plant_v.T  # (n, n_plant)
        cand = cosv.abs().topk(3, dim=0).indices  # (3, n_plant)
        pl = {
            "cand": cand,
            "both": torch.zeros(3, len(PLANT), device=dev, dtype=torch.float64),
            "fire": torch.zeros(3, len(PLANT), device=dev, dtype=torch.float64),
            "ind": torch.zeros(len(PLANT), device=dev, dtype=torch.float64),
        }
    n_tok = {s: 0 for s in SOURCES}
    n_gen = {s: 0 for s in SOURCES}
    for src, p in paths.items():
        seqs = load_seqs(p)
        meta = load_meta(p)
        ho = held[src]
        hs = [seqs[i] for i in ho]
        gs = meta["gen_start"][ho]
        se = {"err": torch.zeros(2, device=dev), "var": torch.zeros(2, device=dev), "N": 0}
        x_all, lens = acts.run(hs)
        ind_all = None
        if pl is not None:
            ind_all = torch.from_numpy(acts.indicators(hs)).to(dev)
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
                dref = D[:, ref]  # (n, d)
                for m in (0, 1):
                    ls[(ref, m, "err")] += (f * (err[:, m] @ dref.T)).double().sum(0)
                    ls[(ref, m, "rec")] += (f * (rec[:, m] @ dref.T)).double().sum(0)
            if pl is not None and ind_all is not None:
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
    dd = norms.double() ** 2  # (n, 2)
    cross = (D[:, 0] * D[:, 1]).sum(-1).double()  # <d^A, d^B>
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
        assert acts.plant_v is not None
        cosv = F.normalize(D[:, 1], dim=-1) @ acts.plant_v.T
        det = []
        for i, spec in enumerate(PLANT):
            rows = []
            for c in range(3):
                j = int(pl["cand"][c, i])
                both, fire, ind = (
                    float(pl[k][c, i]) if k != "ind" else float(pl[k][i])
                    for k in ("both", "fire", "ind")
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
    res["loss"] = splice_loss(cc, acts, paths, held, a.n_held)
    res["l0"] = float(sum(acc["fire"][s].sum() for s in SOURCES) / max(1, sum(n_tok.values())))
    res["dead_frac"] = float((sum(acc["fire"][s] for s in SOURCES) == 0).float().mean())
    (out / "eval.json").write_text(json.dumps(res, indent=1))
    torch.save(lat, out / "latents.pt")
    print(f"{pt_now()} eval {json.dumps({k: res[k] for k in ('fvu', 'l0', 'dead_frac')})}")
    print(f"{pt_now()} loss {json.dumps(res['loss'])}", flush=True)
    return res


@torch.no_grad()
def splice_loss(
    cc: Crosscoder, acts: Acts, paths: dict[str, Path], held: dict[str, list[int]], n: int = 1500
) -> dict[str, Any]:
    """Next-token loss on sampled tokens of held-out rollout sequences, per model: clean, with
    the layer replaced by the crosscoder reconstruction, and by the mean activation."""
    dev = acts.device
    out: dict[str, Any] = {}
    for src in ("b_roll", "a_roll"):
        seqs = load_seqs(paths[src])
        meta = load_meta(paths[src])
        ho = held[src][:n]
        sums = {f"m{m}_{c}": 0.0 for m in (0, 1) for c in ("clean", "recon", "mean")}
        cnt = 0
        mean_act = None
        for i in range(0, len(ho), 64):
            batch = ho[i : i + 64]
            T = max(len(seqs[q]) for q in batch)
            ids = torch.zeros(len(batch), T, dtype=torch.long)
            lm = torch.zeros(len(batch), T, dtype=torch.bool)
            for r, q in enumerate(batch):
                s = seqs[q]
                ids[r, : len(s)] = torch.from_numpy(s.astype(np.int64))
                lm[r, int(meta["gen_start"][q]) - 1 : len(s) - 1] = True  # predicts sampled ids
            ids, lm = ids.to(dev), lm.to(dev)
            hs = []
            for m in acts.models:
                with capture(m, [acts.layer]) as cap:
                    m.features(ids)
                hs.append(cap[acts.layer])
            x = torch.stack(hs, 2) * acts.scale[None, None, :, None]  # (b, T, 2, d)
            b_, T_ = x.shape[:2]
            f, _ = cc.encode(x.view(b_ * T_, 2, -1))
            xh = cc.decode(f).view(b_, T_, 2, -1)
            if mean_act is None:
                mean_act = x[lm].mean(0)  # (2, d)
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
                        logits = model(ids)
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
        r = {k: v / cnt for k, v in sums.items()}
        for m in (0, 1):
            c, rc, mn = r[f"m{m}_clean"], r[f"m{m}_recon"], r[f"m{m}_mean"]
            r[f"m{m}_recovered"] = (mn - rc) / (mn - c) if mn > c else float("nan")
        r["tokens"] = cnt
        out[src] = r
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="base:0")
    ap.add_argument("--b", required=True)
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
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--device", default="cuda", help="cpu: smoke tests only")
    ap.add_argument("--n_held", type=int, default=1500, help="splice-loss sequences per source")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.device == "cuda":
        assert a.gpu is not None, "--gpu"
        assert_gpu_free(a.gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    if a.out is None:
        b = a.b.replace(":", "_s")
        a.out = str(CC / f"{b}_L{a.layer}_n{a.n_latents}_k{a.k}_seed{a.seed}")
    if (Path(a.out) / "eval.json").exists():
        print(f"exists: {a.out}")
        return 0
    print(f"{pt_now()} start {vars(a)}", flush=True)
    train(a)
    print(f"{pt_now()} done {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
