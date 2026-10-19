"""Collapse audit, probe 6: which weights carry the `:` attractor (CPU, read-only).

Run from the repo root, with M = code.experiments.current.thinking_traces_20261005.collapse_audit:

    code/.venv/bin/python -m $M.patch_probe \
        --good s0_it39 --bad s0_it49 [--n 120]

On the teacher-forced 13-20 reference steps whose state holds >= 10 `:` tokens (B0 layout, as
`teacher_force.cond_ids`), measures the mean P(':') at non-':' action targets and the action NLL for
the good and bad checkpoints, then for each tensor group G: good + bad's G ("transplant": does G
alone bring the attractor in?) and bad + good's G ("repair": does reverting G alone remove it?).
Groups: emb, head, ln_f, and per block its attention (ln1, qkv, proj, ln_attn_out) and MLP (ln2,
fc1, fc2, ln_mlp_out).
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
import torch

from code.experiments.current.thinking_traces_20261005.collapse_audit.teacher_force import (
    CKPTS,
    build,
    cond_ids,
)
from code.thinking.ckpt import load_ckpt
from code.thinking.sample import phase_bans


def groups(sd: dict[str, torch.Tensor], n_layer: int) -> dict[str, list[str]]:
    g: dict[str, list[str]] = {"emb": ["emb.weight"], "head": ["head.weight"]}
    g["ln_f"] = [k for k in sd if k.startswith("ln_f.")]
    for i in range(n_layer):
        pre = f"blocks.{i}."
        att = [
            k
            for k in sd
            if k.startswith(pre) and any(s in k for s in ("ln1.", "qkv.", "proj.", "ln_attn_out."))
        ]
        mlp = [
            k
            for k in sd
            if k.startswith(pre) and any(s in k for s in ("ln2.", "fc1.", "fc2.", "ln_mlp_out."))
        ]
        g[f"b{i}.attn"] = att
        g[f"b{i}.mlp"] = mlp
    covered = {k for v in g.values() for k in v}
    rest = [k for k in sd if k not in covered]
    if rest:
        g["other"] = rest
    return g


@torch.inference_mode()
def measure(model, tok, steps) -> tuple[float, float]:
    _, ba = phase_bans(tok, model.cfg["vocab"], True)
    colon = tok.stoi[":"]
    pcs, nll = [], []
    seqs = [cond_ids(tok, s, "B0") for s in steps]
    for i in range(0, len(seqs), 64):
        ch = seqs[i : i + 64]
        L = max(len(p) + len(t) for p, t, _ in ch)
        x = torch.zeros((len(ch), L), dtype=torch.long)
        for r, (p, t, _) in enumerate(ch):
            x[r, : len(p) + len(t)] = torch.tensor(p + t)
        z = model(x[:, :-1]).float()
        for r, (p, t, _) in enumerate(ch):
            q0 = len(p) - 1
            lp = torch.log_softmax(z[r, q0 : q0 + len(t)].masked_fill(ba, -float("inf")), -1)
            tg = torch.tensor(t)
            m = tg != colon
            pcs.append(lp[m, colon].exp().numpy())
            nll.append(-lp.gather(1, tg[:, None])[:, 0].numpy())
    return float(np.concatenate(pcs).mean()), float(np.concatenate(nll).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--good", default="s0_it39")
    ap.add_argument("--bad", default="s0_it49")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--min_colons", type=int, default=10)
    a = ap.parse_args()
    torch.set_num_threads(8)
    good, tok, _ = load_ckpt(CKPTS[a.good], "cpu")
    bad, _, _ = load_ckpt(CKPTS[a.bad], "cpu")
    colon = tok.stoi[":"]
    steps = [
        s for s in build("t1320", a.n) if sum(1 for v in s["state"] if v == colon) >= a.min_colons
    ]
    print(f"{len(steps)} steps with >= {a.min_colons} ':' in the state")
    sg, sb = good.state_dict(), bad.state_dict()
    pg, ng = measure(good, tok, steps)
    pb, nb = measure(bad, tok, steps)
    print(f"{a.good}: P(':') {100 * pg:.3f}%  NLL {ng:.4f}")
    print(f"{a.bad}: P(':') {100 * pb:.3f}%  NLL {nb:.4f}")
    work = copy.deepcopy(good)
    # row-only patches: the `:` row of the head (its logit weights) and of the embedding
    for key in ("head.weight", "emb.weight"):
        t = sg[key].clone()
        t[colon] = sb[key][colon]
        work.load_state_dict({**sg, key: t})
        pt, nt = measure(work, tok, steps)
        t = sb[key].clone()
        t[colon] = sg[key][colon]
        work.load_state_dict({**sb, key: t})
        pr, nr = measure(work, tok, steps)
        print(
            f"  {key[:4]}[':'] transplant: P(':') {100 * pt:7.3f}%  NLL {nt:.4f} | "
            f"repair: P(':') {100 * pr:7.3f}%  NLL {nr:.4f}"
        )
    for name, keys in groups(sg, good.cfg["n_layer"]).items():
        work.load_state_dict({**sg, **{k: sb[k] for k in keys}})
        pt, nt = measure(work, tok, steps)
        work.load_state_dict({**sb, **{k: sg[k] for k in keys}})
        pr, nr = measure(work, tok, steps)
        print(
            f"  {name:10s} transplant (good + bad's): P(':') {100 * pt:7.3f}%  NLL {nt:.4f} | "
            f"repair (bad + good's): P(':') {100 * pr:7.3f}%  NLL {nr:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
