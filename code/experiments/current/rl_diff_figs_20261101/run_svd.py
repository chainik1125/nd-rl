#!/usr/bin/env python3
"""Keep the top n singular components of each weight matrix's change (round r minus pretrain), score the result.

  python run_svd.py --seed 0 --rounds 1 2 4 8 16 --ns 1 4 16 64 [--others base] [--k 32] [--limit 8] [--pretrain 1600]

Matrices are all 2-D parameters (26: qkv, proj, fc1, fc2 per block, plus emb and head). `--others` sets what the 1-D
parameters (LayerNorms, biases) do: base holds them at the pretrain, full moves them to round r.
"""
import argparse
import time

import torch

import common as C


@torch.no_grad()
def svds(s0, s1):
    out = {}
    for name, v in s0.items():
        if v.ndim == 2 and v.is_floating_point():
            U, S, Vh = torch.linalg.svd((s1[name] - v).float(), full_matrices=False)
            out[name] = (U, S, Vh)
    return out


@torch.no_grad()
def truncated(s0, s1, dec, n, others):
    state = {}
    for name, v in s0.items():
        if name in dec:
            U, S, Vh = dec[name]
            m = min(n, len(S))
            state[name] = v + ((U[:, :m] * S[:m]) @ Vh[:m]).to(v.dtype)
        else:
            state[name] = s1[name] if others == "full" else v
    return state


def energy_kept(dec, n):
    kept = sum((S[:n] ** 2).sum().item() for _, S, _ in dec.values())
    return kept / sum((S ** 2).sum().item() for _, S, _ in dec.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, nargs="+", required=True)
    ap.add_argument("--ns", type=int, nargs="+", required=True)
    ap.add_argument("--others", default="base", choices=("base", "full"))
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pretrain", type=int, default=None, help="Stage-1 step the ladder started from (rl-from-ckpt)")
    a = ap.parse_args()
    recs = C.problems(a.limit)
    out = C.RESULTS / "svd" / (f"s{a.seed}_p{a.pretrain}" if a.pretrain else f"s{a.seed}")
    model, tok, _ = C.load(a.seed, 0, a.pretrain)
    s0 = {k: v.clone() for k, v in model.state_dict().items()}

    def run(path, state, **meta):
        if C.done(path):
            return
        model.load_state_dict(state)
        t = time.time()
        res = C.evaluate(model, tok, recs, a.k, a.batch)
        C.save(path, {**meta, "seed": a.seed, "k": a.k, "wall_s": time.time() - t, **res})
        ok = sum(r["n_ok"] > 0 for r in res["rows"])
        print(f"{path.name}: solved-any {ok}/{len(recs)} in {time.time() - t:.0f}s", flush=True)

    run(out / "ref_r0.json", s0, round=0, ref=True)
    for r in a.rounds:
        s1 = torch.load(C.ckpt_path(a.seed, r, a.pretrain), map_location=C.DEVICE)["state"]
        run(out / f"ref_r{r}.json", s1, round=r, ref=True)
        dec = svds(s0, s1)
        for n in a.ns:
            run(out / f"r{r}_n{n}_{a.others}.json", truncated(s0, s1, dec, n, a.others), round=r, n=n,
                others=a.others, energy=energy_kept(dec, n))


if __name__ == "__main__":
    main()
