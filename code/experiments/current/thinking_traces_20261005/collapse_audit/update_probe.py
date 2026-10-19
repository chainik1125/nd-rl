"""Collapse audit, probe 7: was the attractor a first-order effect of the updates or a cliff?
(CPU, read-only.)

Run from the repo root, with M = code.experiments.current.thinking_traces_20261005.collapse_audit:

    code/.venv/bin/python -m $M.update_probe

For a checkpoint pair (A -> B) with weight change d = theta_B - theta_A, on the teacher-forced
long-state steps (>= 10 `:` in the state; 13-20 references, B0 layout): the colon score
c(theta) = mean log P(':') over the non-':' action targets, its gradient g at A, the first-order
prediction g . d against the realised c(B) - c(A), and c and mean P(':') along the line
theta_A + alpha d.
"""

from __future__ import annotations

import torch

from code.experiments.current.thinking_traces_20261005.collapse_audit.teacher_force import (
    CKPTS,
    build,
    cond_ids,
)
from code.thinking.ckpt import load_ckpt
from code.thinking.sample import phase_bans

PAIRS = [
    ("s0_it39", "s0_it49"),
    ("s0_it49", "s0_it50"),
    ("s0_it50", "s0_it52"),
    ("s1_it39", "s1_it49"),
    ("s1_it49", "s1_it50"),
]
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)


def colon_score(model, tok, seqs, grad: bool = False) -> tuple[torch.Tensor, float]:
    """-> (mean log P(':') at non-':' action targets, mean P(':') there)."""
    _, ba = phase_bans(tok, model.cfg["vocab"], True)
    colon = tok.stoi[":"]
    tot = torch.zeros(())
    n = 0
    ps = 0.0
    for i in range(0, len(seqs), 64):
        ch = seqs[i : i + 64]
        L = max(len(p) + len(t) for p, t, _ in ch)
        x = torch.zeros((len(ch), L), dtype=torch.long)
        for r, (p, t, _) in enumerate(ch):
            x[r, : len(p) + len(t)] = torch.tensor(p + t)
        with torch.set_grad_enabled(grad):
            z = model(x[:, :-1]).float()
            for r, (p, t, _) in enumerate(ch):
                q0 = len(p) - 1
                lp = torch.log_softmax(z[r, q0 : q0 + len(t)].masked_fill(ba, -float("inf")), -1)
                m = torch.tensor(t) != colon
                v = lp[m, colon]
                tot = tot + v.sum()
                n += int(m.sum())
                ps += float(v.detach().exp().sum())
    return tot / n, ps / n


def main() -> int:
    torch.set_num_threads(8)
    _, tok, _ = load_ckpt(CKPTS["s0_it39"], "cpu")
    colon = tok.stoi[":"]
    steps = [s for s in build("t1320", 200) if sum(1 for v in s["state"] if v == colon) >= 10]
    seqs = [cond_ids(tok, s, "B0") for s in steps]
    print(f"{len(seqs)} long-state steps")
    for a, b in PAIRS:
        ma, _, _ = load_ckpt(CKPTS[a], "cpu")
        mb, _, _ = load_ckpt(CKPTS[b], "cpu")
        sa = {k: v.detach().clone() for k, v in ma.state_dict().items()}
        sb = {k: v.detach().clone() for k, v in mb.state_dict().items()}
        ma.zero_grad(set_to_none=True)
        ca, pa = colon_score(ma, tok, seqs, grad=True)
        ca.backward()
        named = dict(ma.named_parameters())
        pred = 0.0
        gn = dn = 0.0
        for k, p in named.items():
            if p.grad is None:
                continue
            d = (sb[k] - sa[k]).float()
            pred += float((p.grad * d).sum())
            gn += float((p.grad**2).sum())
            dn += float((d**2).sum())
        cb, pb = colon_score(mb, tok, seqs)
        cos = pred / max(1e-12, (gn**0.5) * (dn**0.5))
        print(
            f"\n{a} -> {b}: c {float(ca):.3f} -> {float(cb):.3f} (realised {float(cb - ca):+.3f}), "
            f"first-order g.d {pred:+.3f}, cos(g, d) {cos:+.4f}, |d| {dn**0.5:.3f}; "
            f"P(':') {100 * pa:.3f}% -> {100 * pb:.3f}%"
        )
        line = []
        work, _, _ = load_ckpt(CKPTS[a], "cpu")
        for al in ALPHAS:
            work.load_state_dict({k: sa[k] + al * (sb[k] - sa[k]) for k in sa})
            c, p = colon_score(work, tok, seqs)
            line.append(f"a={al:g}: {100 * p:.3f}% ({float(c):.2f})")
        print("  along A + a d: P(':') (mean log P): " + "  ".join(line))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
