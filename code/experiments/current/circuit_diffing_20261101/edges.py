"""Stage 3 part B: edges between crosscoder latents at layer 3 and layer 5 (and layer 5 -> the
metric), and teacher-forced path patching in the L3 end (PREREG Stage 3).

    python -m code.experiments.current.circuit_diffing_20261101.edges score --gpu 2 --x L3
    python -m code.experiments.current.circuit_diffing_20261101.edges patch --gpu 2

Pairs: base vs X with the pair's own crosscoders `<X>_s0_L{3,5}_n8192_k32_seed0`. Sequences:
`stage3/seqs_L3.npz` (`traces.py`), the same tokens in every model. The joint crosscoder code f
(both models' activations) is the same in both models, so only the wiring differs.

Metric M_m: the summed log-prob of the accepted action tokens under model m.

score (pass 1, nodes): per model m, A^m_j = sum_tokens f_j <dM_m/da_m, d_j^m / s_m> at layers 3
  and 5, per problem. U3, U5 = the union over both models of the top `--top_nodes` latents by
  |A^m| summed over G.
score (pass 2, edges 3 -> 5): per model m and sender i in U3, the JVP of model m's blocks 4-5 at
  its own layer-3 residual, in the direction f3_i(t) d3_i^m / s3_m at every position t, read at
  layer 5 by the common reader w5_j = W_enc[base]_j + W_enc[X]_j (in model m's units):
  dpre5_j(t'; i, m). EAP edge e^m_ij = sum_t' dpre5_j(t') dM_m/df5_j(t') 1(j active at t');
  the unweighted edge sums dpre5_j over j's active positions. Both per problem.
patch (pass 3, L3 pair only): the edge sets of PREREG, from per-problem edges of the other fold
  (G) or of all G (S); in the driver's forward each receiver j in U5 gets
  pre5_j += sum_{i:(i,j) in E} (dpre5_j(t'; i, other) - dpre5_j(t'; i, driver)), is re-gated with
  the crosscoder's inference rule, and the driver's layer-5 residual moves by
  sum_j (f5'_j - f5_j) d5_j^driver / s5. Writes per-sequence M for every condition.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .common import ART, assert_gpu_free, pt_now, registry

OUT = Path(os.environ.get("CIRCUITS_STAGE3", str(ART / "stage3")))
CC = ART / "crosscoders"
PAIRS = {"L3": "L3", "cpt": "cpt_lr0.00015", "kp16": "kp16_st1"}


# ----------------------------------------------------------------------------- model pieces
def attn(b: Any, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """PeriLNBlock.attn without a cache, with explicit softmax (forward-mode AD friendly)."""
    B, T, D = x.shape
    H = b.h
    q, k, v = b.qkv(x).view(B, T, 3, H, D // H).permute(2, 0, 3, 1, 4)
    s = q @ k.transpose(-1, -2) / math.sqrt(D // H) + bias
    y = torch.softmax(s, -1) @ v
    return b.proj(y.transpose(1, 2).reshape(B, T, D))


def block_f(b: Any, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    x = x + b.ln_attn_out(attn(b, b.ln1(x), bias))
    return x + b.ln_mlp_out(b.fc2(F.gelu(b.fc1(b.ln2(x)))))


def alibi(model: Any, T: int, device: str) -> torch.Tensor:
    """ALiBiGPT.features' causal bias for an unpadded batch (key position = query position)."""
    pos = torch.arange(T, device=device)
    rel = pos[:, None] - pos[None, :]  # (T, T)
    bias = rel.float()[None, None] * (-model.slopes.float())  # (1, H, T, T)
    return bias.masked_fill(rel[None, None] < 0, -1e9)


def run_blocks(model: Any, x: torch.Tensor, lo: int, hi: int, bias: torch.Tensor) -> torch.Tensor:
    """Residual after block hi from the residual after block lo (blocks are 1-indexed)."""
    for b in model.blocks[lo:hi]:
        x = block_f(b, x, bias)
    return x


def metric_from5(
    model: Any, a5: torch.Tensor, bias: torch.Tensor, ids: torch.Tensor, met: torch.Tensor
) -> torch.Tensor:
    """Per sequence: summed log-prob of the metric tokens, from the residual after block 5."""
    h = run_blocks(model, a5, 5, 6, bias)
    logits = model.head(model.ln_f(h))[:, :-1].float()
    lp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None])[..., 0]
    return (lp * met[:, 1:]).sum(1)


# ----------------------------------------------------------------------------- data
def load_seqs() -> dict[str, Any]:
    d = np.load(OUT / "seqs_L3.npz")
    o = d["offsets"]
    seqs = [d["tokens"][o[i] : o[i + 1]] for i in range(len(o) - 1)]
    mets = [d["metric"][o[i] : o[i + 1]] for i in range(len(o) - 1)]
    info = json.loads((OUT / "seqs_L3.json").read_text())
    return dict(
        seqs=seqs,
        mets=mets,
        prob=d["prob"],
        cat=d["cat"],
        attempt=d["attempt"],
        n_prob=len(info["problems"]),
        pcat=np.array([0 if p["cat"] == "G" else 1 for p in info["problems"]]),
        names=[p["name"] for p in info["problems"]],
    )


def make_batches(lens: list[int], idx: list[int], max_tokens: int) -> list[list[int]]:
    order = sorted(idx, key=lambda i: lens[i])
    out, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * max(lens[i], max(lens[q] for q in cur)) > max_tokens:
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


def tensors(data: dict[str, Any], batch: list[int], dev: str) -> tuple[torch.Tensor, ...]:
    T = max(len(data["seqs"][i]) for i in batch)
    ids = torch.zeros(len(batch), T, dtype=torch.long)
    valid = torch.zeros(len(batch), T, dtype=torch.bool)
    met = torch.zeros(len(batch), T)
    for r, i in enumerate(batch):
        n = len(data["seqs"][i])
        ids[r, :n] = torch.from_numpy(data["seqs"][i].astype(np.int64))
        valid[r, :n] = True
        met[r, :n] = torch.from_numpy(data["mets"][i].astype(np.float32))
    pid = torch.as_tensor(data["prob"][batch], dtype=torch.long)
    return ids.to(dev), valid.to(dev), met.to(dev), pid.to(dev)


# ----------------------------------------------------------------------------- pair
class Pair:
    def __init__(self, x: str, dev: str) -> None:
        from .common import load_model
        from .patching import load_cc

        reg = registry()
        self.x, self.dev = x, dev
        lab = PAIRS[x]
        self.models = [
            load_model(reg[("base", 0)].path, dev)[0],
            load_model(reg[(lab, 0)].path, dev)[0],
        ]
        self.cc, self.scale, self.dirs = {}, {}, {}
        for layer in (3, 5):
            d = CC / f"{lab}_s0_L{layer}_n8192_k32_seed0"
            cc, meta = load_cc(d, dev)
            assert meta["a"] == "base:0" and meta["b"] == f"{lab}:0", meta
            self.cc[layer], self.dirs[layer] = cc, d
            self.scale[layer] = torch.tensor(meta["scale"], device=dev, dtype=torch.float32)
        self.normsum5 = self.cc[5].dec_norms().sum(-1)

    def dec(self, layer: int, m: int) -> torch.Tensor:
        """(n, d): latent writes in model m's raw units."""
        return self.cc[layer].W_dec[:, m] / self.scale[layer][m]

    def reader5(self, m: int) -> torch.Tensor:
        """(d, n): the common layer-5 reader applied to model m's raw residual."""
        W = self.cc[5].W_enc
        return (W[0] + W[1]) * self.scale[5][m]

    @torch.no_grad()
    def clean(self, ids: torch.Tensor) -> dict[str, Any]:
        T = ids.shape[1]
        out: dict[str, Any] = {"bias": alibi(self.models[0], T, self.dev), "a3": [], "a5": []}
        for m in self.models:
            x = m.emb(ids)
            x = run_blocks(m, x, 0, 3, out["bias"])
            out["a3"].append(x)
            out["a5"].append(run_blocks(m, x, 3, 5, out["bias"]))
        for layer in (3, 5):
            cc, s = self.cc[layer], self.scale[layer]
            a = out[f"a{layer}"]
            xx = torch.stack([a[0] * s[0], a[1] * s[1]], 2)  # (B, T, 2, d)
            raw = torch.einsum("btmd,mdn->btn", xx, cc.W_enc) + cc.b_enc
            p = F.relu(raw)
            f = p * (p * cc.dec_norms().sum(-1) > cc.theta)
            out[f"f{layer}"] = f
            if layer == 5:
                out["raw5"] = raw
        return out

    def grads(
        self, m: int, cl: dict[str, Any], ids: torch.Tensor, met: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> (dM/da3, dM/da5, M per sequence) for model m."""
        model = self.models[m]
        a3 = cl["a3"][m].detach().requires_grad_(True)
        with torch.enable_grad():
            a5 = run_blocks(model, a3, 3, 5, cl["bias"])
            M = metric_from5(model, a5, cl["bias"], ids, met)
            g3, g5 = torch.autograd.grad(M.sum(), [a3, a5])
        return g3.detach(), g5.detach(), M.detach()

    def dpre5(
        self, m: int, cl: dict[str, Any], senders: torch.Tensor, U5: torch.Tensor
    ) -> torch.Tensor:
        """(c, B, T, |U5|): dpre5_j(t'; i, m) for the senders i (latent ids)."""
        model = self.models[m]
        a3 = cl["a3"][m].detach()
        bias = cl["bias"]
        V = (
            cl["f3"][..., senders].permute(2, 0, 1)[..., None]
            * self.dec(3, m)[senders][:, None, None, :]
        )

        def fn(x: torch.Tensor) -> torch.Tensor:
            return run_blocks(model, x, 3, 5, bias)

        dA5 = torch.func.vmap(lambda v: torch.func.jvp(fn, (a3,), (v,))[1])(V)
        return dA5 @ self.reader5(m)[:, U5]


# ----------------------------------------------------------------------------- score
def score(a: argparse.Namespace) -> None:
    dev = a.device
    data = load_seqs()
    pair = Pair(a.x, dev)
    P, n = data["n_prob"], pair.cc[3].W_dec.shape[0]
    lens = [len(s) for s in data["seqs"]]
    allidx = list(range(len(lens)))
    t0 = time.time()
    # pass 1: nodes
    A = {lay: torch.zeros(2, P, n, device=dev, dtype=torch.float64) for lay in (3, 5)}
    Mseq = torch.zeros(2, len(lens), device=dev, dtype=torch.float64)
    for batch in make_batches(lens, allidx, a.max_tokens1):
        ids, valid, met, pid = tensors(data, batch, dev)
        cl = pair.clean(ids)
        for m in (0, 1):
            g3, g5, M = pair.grads(m, cl, ids, met)
            Mseq[m, batch] = M.double()
            for lay, g in ((3, g3), (5, g5)):
                attr = (cl[f"f{lay}"] * (g @ pair.dec(lay, m).T) * valid[..., None]).sum(1)
                A[lay][m].index_add_(0, pid, attr.double())
        del cl
    print(f"{pt_now()} {a.x} pass 1 (nodes) {time.time() - t0:.0f}s", flush=True)
    g_mask = torch.as_tensor(data["pcat"] == 0, device=dev)
    U = {}
    for lay in (3, 5):
        tot = A[lay][:, g_mask].sum(1).abs()  # (2, n)
        sel = torch.cat([tot[m].topk(a.top_nodes).indices for m in (0, 1)]).unique()
        U[lay] = sel.sort().values
        cov = [float(tot[m][U[lay]].sum() / tot[m].sum()) for m in (0, 1)]
        print(f"{pt_now()} U{lay}: {len(U[lay])} latents, |A| coverage {cov}", flush=True)
    U3, U5 = U[3], U[5]
    u3, u5 = len(U3), len(U5)
    # pass 2: edges 3 -> 5
    E = torch.zeros(2, P, u3, u5, device=dev)
    Eu = torch.zeros(2, P, u3, u5, device=dev)
    t1 = time.time()
    batches = make_batches(lens, allidx, a.max_tokens2)
    for bi, batch in enumerate(batches):
        ids, valid, met, pid = tensors(data, batch, dev)
        cl = pair.clean(ids)
        act5 = (cl["f5"][..., U5] > 0) & valid[..., None]
        for m in (0, 1):
            _, g5, _ = pair.grads(m, cl, ids, met)
            gt5 = (g5 @ pair.dec(5, m)[U5].T) * act5
            for i0 in range(0, u3, a.chunk):
                sl = slice(i0, min(u3, i0 + a.chunk))
                dp = pair.dpre5(m, cl, U3[sl], U5)  # (c, B, T, u5)
                e = torch.einsum("cbtj,btj->bcj", dp, gt5)
                eu = torch.einsum("cbtj,btj->bcj", dp, act5.float())
                E[m, :, sl].index_add_(0, pid, e)
                Eu[m, :, sl].index_add_(0, pid, eu)
                del dp
        del cl
        if bi % 10 == 0 or bi == len(batches) - 1:
            el = time.time() - t1
            print(
                f"{pt_now()} {a.x} pass 2 batch {bi + 1}/{len(batches)}, {el:.0f}s, "
                f"eta {el / (bi + 1) * (len(batches) - bi - 1):.0f}s",
                flush=True,
            )
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            x=a.x,
            pair=PAIRS[a.x],
            U3=U3.cpu(),
            U5=U5.cpu(),
            A3=A[3].float().cpu(),
            A5=A[5].float().cpu(),
            E=E.cpu(),
            Eu=Eu.cpu(),
            M=Mseq.cpu(),
            pcat=data["pcat"],
            names=data["names"],
            args=vars(a),
            secs=time.time() - t0,
        ),
        OUT / f"edges_{a.x}.pt",
    )
    print(f"{pt_now()} {a.x} done in {time.time() - t0:.0f}s -> {OUT / f'edges_{a.x}.pt'}")


# ----------------------------------------------------------------------------- edge sets
def edge_sets(
    eb: np.ndarray,
    ex: np.ndarray,
    K: int = int(os.environ.get("CIRCUITS_TOPK", "1000")),
    eps: float = 0.2,
    n_rand: int = 3,
) -> dict[str, np.ndarray]:
    """PREREG Stage 3 sets on (u3, u5) edge matrices summed over a problem set."""
    flat_x, flat_b = ex.ravel(), eb.ravel()
    top = np.zeros(flat_x.size, dtype=bool)
    top[np.argsort(-np.abs(flat_x))[:K]] = True
    new = top & (np.abs(flat_b) < eps * np.abs(flat_x))
    rest = np.flatnonzero(top & ~new)
    out = {"top": top, "new": new}

    def prefix(order: np.ndarray, w: np.ndarray, target: float) -> np.ndarray:
        m = np.zeros(flat_x.size, dtype=bool)
        if target <= 0 or len(order) == 0:
            return m
        cum = np.cumsum(w[order])
        k = int(np.searchsorted(cum, target))
        if k >= len(order):
            k = len(order) - 1
        elif k > 0 and abs(cum[k - 1] - target) <= abs(cum[k] - target):
            k -= 1
        m[order[: k + 1]] = True
        return m

    ax, ad = np.abs(flat_x), np.abs(flat_x - flat_b)
    for s in range(n_rand):
        out[f"rand{s}"] = prefix(np.random.default_rng(s).permutation(rest), ax, ax[new].sum())
    out["retuned"] = prefix(rest[np.argsort(-ad[rest])], ad, ad[new].sum())
    return {k: v.reshape(ex.shape) for k, v in out.items()}


# ----------------------------------------------------------------------------- patch
def patch(a: argparse.Namespace) -> None:
    dev = a.device
    data = load_seqs()
    pair = Pair("L3", dev)
    ed = torch.load(OUT / "edges_L3.pt", weights_only=False)
    U3, U5 = ed["U3"].to(dev), ed["U5"].to(dev)
    u3 = len(U3)
    Ep = ed["E"].double().numpy()  # (2, P, u3, u5)
    pcat = data["pcat"]
    G = [p for p in range(data["n_prob"]) if pcat[p] == 0]
    g_sh = G[:]
    random.Random(0).shuffle(g_sh)
    fold = {p: (0 if i < len(g_sh) // 2 else 1) for i, p in enumerate(g_sh)}

    def sets_on(probs: list[int]) -> dict[str, np.ndarray]:
        return edge_sets(Ep[0][probs].sum(0), Ep[1][probs].sum(0))

    groups = {
        "G0": sets_on([p for p in G if fold[p] == 1]),
        "G1": sets_on([p for p in G if fold[p] == 0]),
        "S": sets_on(G),
    }
    from .stage2 import latent_sets

    live = torch.from_numpy(latent_sets(pair.dirs[3])["live"]).to(dev)
    rl_conds = ["new", "rand0", "rand1", "rand2", "retuned", "top", "allU"]
    base_conds = ["base_new", "base_rand0"]
    masks: dict[str, dict[str, torch.Tensor]] = {}
    set_info: dict[str, Any] = {}
    for g, st in groups.items():
        st["allU"] = np.ones_like(st["top"])
        st["base_new"], st["base_rand0"] = st["new"], st["rand0"]
        masks[g] = {c: torch.from_numpy(st[c]).float().to(dev) for c in rl_conds + base_conds}
        eb = Ep[0][[p for p in G if (g == "S" or fold[p] != int(g[1]))]].sum(0)
        ex = Ep[1][[p for p in G if (g == "S" or fold[p] != int(g[1]))]].sum(0)
        set_info[g] = {
            c: dict(
                n=int(st[c].sum()),
                mass_x=float(np.abs(ex)[st[c]].sum()),
                change=float(np.abs(ex - eb)[st[c]].sum()),
            )
            for c in rl_conds
        }
        print(f"{pt_now()} sets {g}: {json.dumps(set_info[g])}", flush=True)
    lens = [len(s) for s in data["seqs"]]
    seq_group = []
    for i in range(len(lens)):
        p = int(data["prob"][i])
        seq_group.append("S" if pcat[p] == 1 else f"G{fold[p]}")
    names = ["base", "L3", "swap45", "revert3", *rl_conds, *base_conds]
    Mout = {c: np.full(len(lens), np.nan) for c in names}
    t0 = time.time()
    n_done = 0
    for g in groups:
        idx = [i for i in range(len(lens)) if seq_group[i] == g]
        for batch in make_batches(lens, idx, a.max_tokens2):
            ids, valid, met, pid = tensors(data, batch, dev)
            cl = pair.clean(ids)
            with torch.no_grad():
                bias = cl["bias"]
                Mout["base"][batch] = metric_from5(
                    pair.models[0], cl["a5"][0], bias, ids, met
                ).cpu()
                Mout["L3"][batch] = metric_from5(pair.models[1], cl["a5"][1], bias, ids, met).cpu()
                h = run_blocks(pair.models[0], cl["a3"][1], 3, 5, bias)
                Mout["swap45"][batch] = metric_from5(pair.models[1], h, bias, ids, met).cpu()
                d3 = pair.dec(3, 1)[live] - pair.dec(3, 0)[live]
                a3r = cl["a3"][1] - cl["f3"][..., live] @ d3
                h = run_blocks(pair.models[1], a3r, 3, 5, bias)
                Mout["revert3"][batch] = metric_from5(pair.models[1], h, bias, ids, met).cpu()
                delta = {c: torch.zeros_like(cl["raw5"][..., U5]) for c in rl_conds + base_conds}
                for i0 in range(0, u3, a.chunk):
                    sl = slice(i0, min(u3, i0 + a.chunk))
                    dd = pair.dpre5(0, cl, U3[sl], U5) - pair.dpre5(1, cl, U3[sl], U5)
                    for c in rl_conds:
                        delta[c] += torch.einsum("cbtj,cj->btj", dd, masks[g][c][sl])
                    for c in base_conds:
                        delta[c] -= torch.einsum("cbtj,cj->btj", dd, masks[g][c][sl])
                    del dd
                raw = cl["raw5"][..., U5]
                f_old = cl["f5"][..., U5]
                ns, th = pair.normsum5[U5], pair.cc[5].theta
                for c in rl_conds + base_conds:
                    drv = 0 if c.startswith("base_") else 1
                    p_new = F.relu(raw + delta[c])
                    f_new = p_new * (p_new * ns > th)
                    a5 = cl["a5"][drv] + (f_new - f_old) @ pair.dec(5, drv)[U5]
                    Mout[c][batch] = metric_from5(pair.models[drv], a5, bias, ids, met).cpu()
            n_done += len(batch)
            del cl, delta
            print(f"{pt_now()} patch {g}: {n_done}/{len(lens)} sequences, {time.time() - t0:.0f}s")
    res = dict(
        M={c: v.tolist() for c, v in Mout.items()},
        prob=data["prob"].tolist(),
        attempt=data["attempt"].tolist(),
        pcat=pcat.tolist(),
        fold={str(p): f for p, f in fold.items()},
        sets=set_info,
        args=vars(a),
        secs=time.time() - t0,
    )
    (OUT / "patch_L3.json").write_text(json.dumps(res))
    print(f"{pt_now()} patch done in {time.time() - t0:.0f}s", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("score", "patch"))
    ap.add_argument("--x", default="L3", choices=tuple(PAIRS))
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--device", default="cuda", help="cpu: smoke tests only")
    ap.add_argument("--top_nodes", type=int, default=256)
    ap.add_argument("--max_tokens1", type=int, default=16384, help="pass-1 batch tokens")
    ap.add_argument("--max_tokens2", type=int, default=4096, help="pass-2/3 batch tokens")
    ap.add_argument("--chunk", type=int, default=16, help="senders per vmapped JVP")
    a = ap.parse_args(argv)
    if a.device == "cuda":
        assert a.gpu is not None, "--gpu"
        assert_gpu_free(a.gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"{pt_now()} start {vars(a)}", flush=True)
    score(a) if a.cmd == "score" else patch(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
