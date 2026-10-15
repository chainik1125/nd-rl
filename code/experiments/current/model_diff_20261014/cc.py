# ruff: noqa: E501,SIM115  (research scripts: long result lines, bare open(), short loop names)
"""BatchTopK crosscoder (n models = 2) or SAE (n = 1) on one residual-stream layer, trained on corpus.pt proof tokens.

  python cc.py --models g0_pre,g0_fin --layer 4 [--m 8192 --k 32 --tokens 60e6] --out runs/g0_L4

Architecture (Lindsey et al. 2024 crosscoders; BatchTopK as in Bussmann et al. 2024 / Minder et al. 2025):
  f = BatchTopK( ReLU( sum_M x^M W_enc^M + b_enc ) ), selection score f_i * sum_M ||d_i^M||  (decoder-norm weighted)
  x_hat^M = f W_dec^M + b_dec^M; loss = sum_M ||x^M - x_hat^M||^2 + aux (AuxK on dead latents, Gao et al. 2024)
Inputs are scaled per model by a scalar so E||x||^2 = D. Eval uses a fixed threshold (EMA of the smallest selected
score), so a latent's activation on a token does not depend on the batch.

After training, over every proof token of the corpus:
  - per latent: decoder norms per model, cosine between decoders, firing counts by depth / kind / src / pop_k / close_k,
    mean activation, top-24 activating (token index, value)
  - latent scaling (Minder et al. 2025): beta^M = <f d_j, t^M> / <f d_j, f d_j> fitted over tokens where j fires, with
    d_j the LAST model's decoder, for t^M = x^M (activation) and t^M = x^M - (x_hat^M - f_j d_j^M) (error without j)
  - splice eval: replace the layer's residual by each model's reconstruction, continue the forward: proof-token CE and
    pop accuracy (k >= 1) clean vs spliced, per model
Saves <out>/cc.pt (weights + scales + stats) and <out>/log.txt.
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from tiny import D, Tiny

p = argparse.ArgumentParser()
p.add_argument("--models", required=True)
p.add_argument("--layer", type=int, required=True)
p.add_argument("--m", type=int, default=8192)
p.add_argument("--k", type=int, default=32)
p.add_argument("--tokens", type=float, default=60e6)
p.add_argument("--bs", type=int, default=4096)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--corpus", default="out/corpus.pt")
p.add_argument("--ckdir", default="ckpt")
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--diff", action="store_true", help="SAE on x_fin - x_pre (models = pre,fin)")
p.add_argument("--eval_only", action="store_true", help="load <out>/weights.pt, skip training")
p.add_argument("--oversample", default="", help="src:times, e.g. lad3:3")
p.add_argument("--eval_chunks", type=int, default=0, help="0 = all (smoke tests use a few)")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
LOG = open(f"{args.out}/log.txt", "a")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


CK = {
    "g0": "v3_untied-grpo_s0",
    "g1": "v3_untied-grpo_s1",
    "e0": "v2_untied-ei_s0",
    "e1": "v2_untied-ei_s1",
}
names = args.models.split(",")
n = 1 if args.diff else len(names)  # dictionary input streams


def ckpath(name):
    run, stage = name.split("_")
    return f"{args.ckdir}/runs/long-rl-20261014-{CK[run]}/{'pretrain' if stage == 'pre' else 'final'}.pt"


models = []
for nm in names:
    c = torch.load(ckpath(nm), map_location="cpu")
    models.append(Tiny(c.get("state", c)).to(dev))
C = torch.load(args.corpus)
ids_all = C["ids"].long()
starts = C["starts"].tolist() + [len(ids_all)]
n_prompt = C["n_prompt"].tolist()
R = len(starts) - 1
log(
    f"models {names} layer {args.layer} m {args.m} k {args.k} | corpus records {R} tokens {len(ids_all):,}"
)


def batch_records(idx):
    """pad a list of record indices -> (x (B,T) on dev, mask of proof positions (B,T), global token index (B,T))"""
    L = [starts[i + 1] - starts[i] for i in idx]
    T = max(L)
    x = torch.zeros(len(idx), T, dtype=torch.long)
    g = torch.full((len(idx), T), -1, dtype=torch.long)
    mk = torch.zeros(len(idx), T, dtype=torch.bool)
    for b, i in enumerate(idx):
        x[b, : L[b]] = ids_all[starts[i] : starts[i + 1]]
        g[b, : L[b]] = torch.arange(starts[i], starts[i + 1])
        mk[b, n_prompt[i] : L[b]] = True
    return x.to(dev), mk.to(dev), g.to(dev)


@torch.no_grad()
def acts(idx):
    """(N, n, D) layer activations at proof positions of the given records, and their global token indices"""
    x, mk, g = batch_records(idx)
    a = torch.stack([m.resid(x)[args.layer][mk] for m in models], 1)
    if args.diff:
        a = (a[:, 1] - a[:, 0])[:, None]
    return a, g[mk]


# order records by length so batches pad little; chunks of ~120k tokens
rep = [1] * R
if args.oversample:
    o_src, o_t = args.oversample.split(":")
    for i in range(R):
        if C["SRCS"][int(C["src"][i])] == o_src:
            rep[i] = int(o_t)
order = sorted(
    (i for i in range(R) for _ in range(rep[i])), key=lambda i: starts[i + 1] - starts[i]
)
chunks, cur, tot = [], [], 0
for i in order:
    cur.append(i)
    tot += starts[i + 1] - starts[i]
    if tot > 40_000 or len(cur) >= 256:
        chunks.append(cur)
        cur, tot = [], 0
if cur:
    chunks.append(cur)
# eval chunks: every record once (no oversampling)
eval_chunks, cur, tot = [], [], 0
for i in sorted(range(R), key=lambda i: starts[i + 1] - starts[i]):
    cur.append(i)
    tot += starts[i + 1] - starts[i]
    if tot > 40_000 or len(cur) >= 256:
        eval_chunks.append(cur)
        cur, tot = [], 0
if cur:
    eval_chunks.append(cur)

# scale so E||x||^2 = D per model
with torch.no_grad():
    sample = torch.cat([acts(chunks[j])[0] for j in torch.randperm(len(chunks))[:8].tolist()])
    scale = (D / sample.pow(2).sum(-1).mean(0)).sqrt()  # (n,)
log("scales", scale.tolist())

m, k = args.m, args.k
W_dec = torch.randn(m, n, D, device=dev)
W_dec = W_dec / W_dec.norm(dim=-1, keepdim=True) * 0.1
W_enc = W_dec.clone().permute(1, 2, 0).contiguous()  # (n, D, m)
W_dec = torch.nn.Parameter(W_dec)
W_enc = torch.nn.Parameter(W_enc)
b_enc = torch.nn.Parameter(torch.zeros(m, device=dev))
b_dec = torch.nn.Parameter(torch.zeros(n, D, device=dev))
params = [W_enc, W_dec, b_enc, b_dec]
opt = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))
steps = int(args.tokens // args.bs)


def sched(s):
    return min(1, s / 500) * (1 if s < 0.8 * steps else (steps - s) / (0.2 * steps))


thr = torch.zeros((), device=dev)
last_fire = torch.zeros(m, device=dev)  # tokens since a latent last fired
DEAD = 5_000_000 if args.tokens >= 2e7 else 1_000_000


def encode(x):  # x (B, n, D) scaled
    pre = F.relu(torch.einsum("bnd,ndm->bm", x, W_enc) + b_enc)
    return pre


def dnorm():
    return W_dec.norm(dim=-1)  # (m, n)


def topk_batch(pre, kk):
    score = pre * dnorm().sum(-1)
    flat = score.flatten()
    top = flat.topk(kk * pre.shape[0])
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[top.indices] = True
    return pre * mask.view_as(pre), top.values.min()


def decode(f):
    return torch.einsum("bm,mnd->bnd", f, W_dec) + b_dec


# training with a shuffled activation buffer (skipped with --eval_only)
if args.eval_only:
    wts = torch.load(f"{args.out}/weights.pt", map_location=dev)
    with torch.no_grad():
        for p_, k_ in zip(params, ("W_enc", "W_dec", "b_enc", "b_dec"), strict=False):
            p_.copy_(wts[k_])
    scale = wts["scale"].to(dev)
    steps = 0
buf = torch.empty(0, n, D, device=dev)
ci = torch.randperm(len(chunks)).tolist()
cp = 0
t0 = time.time()
for step in range(steps):
    while buf.shape[0] < args.bs * 64:
        a, _ = acts(chunks[ci[cp % len(ci)]])
        cp += 1
        if cp % len(ci) == 0:
            ci = torch.randperm(len(chunks)).tolist()
        buf = torch.cat([buf, a * scale.view(1, n, 1)])
        buf = buf[torch.randperm(buf.shape[0], device=dev)]
    x, buf = buf[: args.bs], buf[args.bs :]
    for g_ in opt.param_groups:
        g_["lr"] = args.lr * sched(step)
    pre = encode(x)
    f, minsel = topk_batch(pre, k)
    xh = decode(f)
    err = x - xh
    mse = err.pow(2).sum(-1).mean(0)  # (n,)
    loss = mse.sum()
    # AuxK: dead latents reconstruct the residual error
    fired = (f > 0).any(0)
    last_fire = torch.where(fired, torch.zeros_like(last_fire), last_fire + args.bs)
    dead = last_fire > DEAD
    if dead.any():
        kaux = min(512, int(dead.sum()))
        pa = pre * dead
        fa, _ = topk_batch(pa, max(1, kaux // 8))
        ea = torch.einsum("bm,mnd->bnd", fa, W_dec)
        aux = (err.detach() - ea).pow(2).sum(-1).mean(0).sum()
        loss = loss + aux / 32
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    thr = minsel.detach() if step == 0 else 0.999 * thr + 0.001 * minsel.detach()
    if step % 500 == 0 or step == steps - 1:
        fvu = (mse / x.pow(2).sum(-1).mean(0)).tolist()
        log(
            f"step {step}/{steps} loss {loss.item():.2f} fvu {[round(v, 4) for v in fvu]} dead {int(dead.sum())} "
            f"L0 {float((f > 0).sum(-1).float().mean()):.1f} thr {float(thr):.4f} {time.time() - t0:.0f}s"
        )


torch.set_grad_enabled(
    False
)  # everything below is analysis: no autograd graphs (they leak across chunks)
if not args.eval_only:
    torch.save(
        {
            "W_enc": W_enc.detach(),
            "W_dec": W_dec.detach(),
            "b_enc": b_enc.detach(),
            "b_dec": b_dec.detach(),
            "scale": scale,
            "args": vars(args),
            "names": names,
        },
        f"{args.out}/weights.pt",
    )
    log("saved weights")
del buf
if dev == "cuda":
    torch.cuda.empty_cache()
# calibrate the eval threshold so the mean L0 on fresh activations is k (BatchTopK's threshold in expectation)
with torch.no_grad():
    sc = []
    for j in torch.randperm(len(chunks))[:6].tolist():
        a, _ = acts(chunks[j])
        for s0 in range(0, a.shape[0], 8192):
            pre = encode(a[s0 : s0 + 8192] * scale.view(1, n, 1))
            sc.append((pre * dnorm().sum(-1)).flatten().topk(k * pre.shape[0]).values.min().view(1))
    log(f"thr EMA {float(thr):.4f} -> calibrated {float(torch.cat(sc).mean()):.4f}")
    thr = torch.cat(sc).mean()


# ---------------- analysis over all proof tokens ----------------
@torch.no_grad()
def enc_eval(x):
    pre = encode(x)
    score = pre * dnorm().sum(-1)
    return pre * (score > thr)


labels = {k_: C[k_].long().to(dev) for k_ in ("depth", "kind", "pop_k", "close_k", "in_f")}
tok_src = torch.repeat_interleave(
    C["src"].long(), torch.tensor([starts[i + 1] - starts[i] for i in range(R)])
).to(dev)
NB = {"depth": 6, "kind": len(C["KIND"]), "pop_k": 5, "close_k": 5, "src": len(C["SRCS"])}
cnt = {k_: torch.zeros(NB[k_], m, device=dev) for k_ in NB}  # firing counts per label value
tot = {k_: torch.zeros(NB[k_], device=dev) for k_ in NB}
fsum = torch.zeros(m, device=dev)
ffreq = torch.zeros(m, device=dev)
TOPN = 24
topv = torch.zeros(TOPN, m, device=dev)
topi = torch.full((TOPN, m), -1, device=dev, dtype=torch.long)
num = torch.zeros(2, n, m, device=dev)
den = torch.zeros(m, device=dev)
fvu_num = torch.zeros(n, device=dev)
fvu_den = torch.zeros(n, device=dev)
dn = W_dec.detach()
dlast = dn[:, -1]  # (m, D) last model's decoder
for ch in (
    eval_chunks[:: max(1, len(eval_chunks) // args.eval_chunks)]
    if args.eval_chunks
    else eval_chunks
):
    a_all, g_all = acts(ch)
    for s0 in range(0, a_all.shape[0], 4096):
        a, g = a_all[s0 : s0 + 4096], g_all[s0 : s0 + 4096]
        x = a * scale.view(1, n, 1)
        f = enc_eval(x)
        xh = decode(f)
        fvu_num += (x - xh).pow(2).sum(-1).sum(0)
        fvu_den += (x - x.mean(0)).pow(2).sum(-1).sum(0)
        on = (f > 0).float()
        gl = g
        for k_ in NB:
            v = tok_src[gl] if k_ == "src" else labels[k_][gl]
            ok = v >= 0
            cnt[k_].index_add_(0, v[ok], on[ok])
            tot[k_].index_add_(0, v[ok], torch.ones(int(ok.sum()), device=dev))
        fsum += f.sum(0)
        ffreq += on.sum(0)
        ct = f.topk(min(TOPN, f.shape[0]), dim=0)
        cv = torch.cat([topv, ct.values])
        ciu = torch.cat([topi, g[ct.indices]])
        tv = cv.topk(TOPN, dim=0)
        topv, topi = tv.values, ciu.gather(0, tv.indices)
        # latent scaling with d_j = last model's decoder: proj_j(t) = <d_j, t>
        for M in range(n):
            proj_x = x[:, M] @ dlast.T  # (N, m) <x^M, d_j>
            # error without j: x - xh + f_j d_j^M ; <., d_j> = <x - xh, d_j> + f_j <d_j^M, d_j>
            proj_e = (x[:, M] - xh[:, M]) @ dlast.T + f * (dn[:, M] * dlast).sum(-1)
            num[0, M] += (f * proj_x).sum(0)
            num[1, M] += (f * proj_e).sum(0)
        den += (f.pow(2)).sum(0) * dlast.pow(2).sum(-1)
beta = num / den.clamp_min(1e-9)  # (2 targets, n, m)
log("eval FVU per model", (fvu_num / fvu_den).tolist())


# splice eval on records with pops at k >= 1
@torch.no_grad()
def splice_eval(max_recs=1500 if not args.eval_chunks else 64):
    pk = C["pop_k"].long()
    has = [i for i in range(R) if (pk[starts[i] : starts[i + 1]] >= 1).any()]
    pk = pk.to(dev)
    has = has[:: max(1, len(has) // max_recs)]
    res = {}
    for M, mod in enumerate(models):
        s = {
            "ce_clean": 0.0,
            "ce_splice": 0.0,
            "ce_zero": 0.0,
            "n_tok": 0,
            "pop_clean": 0,
            "pop_splice": 0,
            "n_pop": 0,
        }
        for j in range(0, len(has), 16):
            idx = has[j : j + 16]
            x, mk, g = batch_records(idx)
            h = mod.resid(x)
            hl = h[args.layer]
            B, T, _ = hl.shape
            xs = torch.zeros(B, T, n, D, device=dev)
            for M2, mod2 in enumerate(models):
                xs[:, :, M2] = (h if M2 == M else mod2.resid(x))[args.layer]
            z = xs.view(-1, n, D) * scale.view(1, n, 1)
            rec = (decode(enc_eval(z))[:, M] / scale[M]).view(B, T, D)
            rec = torch.where(mk[..., None], rec, hl)  # splice proof positions only
            lc = mod.logits_from(hl, args.layer)
            ls = mod.logits_from(rec, args.layer)
            tgt = x[:, 1:]
            m2 = mk[:, 1:] & mk[:, :-1]
            for key, lg in (("ce_clean", lc), ("ce_splice", ls)):
                s[key] += float(F.cross_entropy(lg[:, :-1][m2], tgt[m2], reduction="sum"))
            s["n_tok"] += int(m2.sum())
            pkb = torch.full((B, T), -1, dtype=torch.long, device=dev)
            gg = g.clamp_min(0)
            pkb = torch.where(g >= 0, pk[gg], pkb)
            pm = pkb[:, :-1] >= 1
            s["pop_clean"] += int((lc[:, :-1].argmax(-1) == tgt)[pm].sum())
            s["pop_splice"] += int((ls[:, :-1].argmax(-1) == tgt)[pm].sum())
            s["n_pop"] += int(pm.sum())
        res[names[M]] = {
            "ce_clean": s["ce_clean"] / s["n_tok"],
            "ce_splice": s["ce_splice"] / s["n_tok"],
            "pop_acc_clean": s["pop_clean"] / s["n_pop"],
            "pop_acc_splice": s["pop_splice"] / s["n_pop"],
            "n_pop": s["n_pop"],
        }
    return res


@torch.no_grad()
def splice_diff(max_recs=1500):
    """pre + reconstructed diff through pre's later blocks, vs pre (clean) and pre + full diff (= fin's residual)"""
    pk = C["pop_k"].long().to(dev)
    has = [i for i in range(R) if (C["pop_k"][starts[i] : starts[i + 1]] >= 1).any()]
    has = has[:: max(1, len(has) // max_recs)]
    hit = {"pre": 0, "full": 0, "sparse": 0}
    npop = 0
    for j in range(0, len(has), 64):
        x, mk, g = batch_records(has[j : j + 64])
        hp, hf = models[0].resid(x)[args.layer], models[1].resid(x)[args.layer]
        B, T, _ = hp.shape
        d = (hf - hp).reshape(-1, 1, D) * scale.view(1, 1, 1)
        rec = (decode(enc_eval(d))[:, 0] / scale[0]).view(B, T, D)
        rec = torch.where(mk[..., None], rec, torch.zeros_like(rec))
        tgt = x[:, 1:]
        pm = torch.where(g >= 0, pk[g.clamp_min(0)], torch.full_like(g, -1))[:, :-1] >= 1
        for key, h in (("pre", hp), ("full", hf), ("sparse", hp + rec)):
            hit[key] += int(
                (models[0].logits_from(h, args.layer)[:, :-1].argmax(-1) == tgt)[pm].sum()
            )
        npop += int(pm.sum())
    return {k_: v / npop for k_, v in hit.items()} | {"n_pop": npop}


sp = splice_diff() if args.diff else splice_eval()
log("splice", json.dumps(sp))
nrm = dn.norm(dim=-1)  # (m, n)
out = {
    "args": vars(args),
    "names": names,
    "scale": scale.cpu(),
    "thr": float(thr),
    "W_enc": W_enc.detach().cpu(),
    "W_dec": dn.cpu(),
    "b_enc": b_enc.detach().cpu(),
    "b_dec": b_dec.detach().cpu(),
    "dec_norm": nrm.cpu(),
    "freq": (ffreq / tot["src"].sum()).cpu(),
    "fsum": fsum.cpu(),
    "cnt": {k_: v.cpu() for k_, v in cnt.items()},
    "tot": {k_: v.cpu() for k_, v in tot.items()},
    "topv": topv.cpu(),
    "topi": topi.cpu(),
    "beta": beta.cpu(),
    "splice": sp,
    "fvu": (fvu_num / fvu_den).cpu(),
}
if n == 2:
    out["cos"] = F.cosine_similarity(dn[:, 0], dn[:, 1], dim=-1).cpu()
    out["rel"] = (nrm[:, 1] / nrm.sum(-1).clamp_min(1e-9)).cpu()
    r = out["rel"]
    log("rel-norm histogram (0=only A, 1=only B):", torch.histc(r, 20, 0, 1).int().tolist())
torch.save(out, f"{args.out}/cc.pt")
log("saved", f"{args.out}/cc.pt", f"{time.time() - t0:.0f}s")
