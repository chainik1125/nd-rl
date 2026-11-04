"""Dan's held-out read with his own code: a k 32 draw on his transfer_track832, counted on the 106.

    python danread.py --ckpt CKPT --seed S --round R --out OUT.json [--only106] [--batch 1024]

This is `state_ladder_ei.py` (Dan's fork at 13e41bbd, exported to KD/dan_code_13e41bbd) step 2,
"transfer, sampled": `generate` (state_sample.env_generate: T 0.8, max_action 256, max_steps 48)
on every transfer prompt × k 32 with noise seed `S*1000 + R + 500`, then `eval_set.judge` (Lean
alone decides). His heldout_cum counts a strict-106 member (data/coe/sigma_heldout_strict.txt)
from the first round it is solved (coe/coe_rounds.py L172–183); this writes the members solved in
this draw, so a cumulative count is the union over reads. With his seed, batch and set (the
default) it repeats his round-R draw; `--only106` samples just the 106 (an equivalent but different
draw, 8 × cheaper). Also written: per-member proofs' σ flags (coe/coe_sigma.py), the draw's token
counts (gen and state prefill, from the env's stats) and timings. Proof texts stay in OUT only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

KD = Path(os.environ.get("KD", "/scratch/cowifornia/nd-rl/knockout_danbase"))
DAN = KD / "dan_code_13e41bbd"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only106", action="store_true")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--k", type=int, default=32)
    a = ap.parse_args()
    os.chdir(DAN)
    sys.path.insert(0, str(DAN))
    sys.path.insert(0, str(DAN / "coe"))
    import torch  # noqa: F401
    from coe_sigma import flags  # type: ignore[import-not-found]
    from eval_set import judge  # type: ignore[import-not-found]
    from model import load_ckpt  # type: ignore[import-not-found]
    from state_ladder_ei import ENV, generate, read  # type: ignore[import-not-found]

    strict = [x.strip() for x in Path("data/coe/sigma_heldout_strict.txt").read_text().split()]
    transfer = read("data/coe/transfer_track832.jsonl")
    if a.only106:
        s = set(strict)
        transfer = [t for t in transfer if t["name"] in s]
    ENV["max_action"], ENV["max_steps"], ENV["stats"] = 256, 48, {}
    t0 = time.time()
    model, tok, _ = load_ckpt(a.ckpt, "cuda")
    seed = a.seed * 1000 + a.round
    prompts = [t["prompt"] for t in transfer for _ in range(a.k)]
    flat = generate(model, tok, prompts, greedy=False, temperature=0.8, batch=a.batch,
                    seed=seed + 500)  # fmt: skip
    t_gen = time.time() - t0
    outs = [flat[i * a.k : (i + 1) * a.k] for i in range(len(transfer))]
    rows = judge(transfer, outs, "n_lines")
    by = {r["name"]: r for r in rows}
    solved = sorted(n for n in strict if n in by and by[n]["solved"])
    first32, e32 = solved, None
    if a.k > 32:  # Dan's measure is a k 32 draw: the first 32 attempts; all k give the E32
        rows32 = judge(transfer, [o[:32] for o in outs], "n_lines")
        first32 = sorted(r["name"] for r in rows32 if r["name"] in set(strict) and r["solved"])
        e32 = sum(1.0 - math.comb(a.k - by[n]["n_ok"], 32) / math.comb(a.k, 32)
                  for n in strict if n in by)  # fmt: skip
    t_all = time.time() - t0
    sig = {}
    for n in solved:
        fl = [f["w"] for f in (flags(p) for p in by[n]["proofs"]) if f]
        sig[n] = {"n_ok": by[n]["n_ok"],
                  "sigma": any(f["sigma_chain"] or f["sigma_strict"] for f in fl),
                  "dn_derived": any(f["dn_derived"] for f in fl)}  # fmt: skip
    st = ENV["stats"]
    res = {
        "ckpt": a.ckpt, "seed": a.seed, "round": a.round, "noise_seed": seed + 500, "k": a.k,
        "batch": a.batch, "set": "106" if a.only106 else "transfer_track832",
        "n_prompts": len(transfer), "strict_solved": solved, "strict_n": len(solved),
        "first32_solved": first32, "first32_n": len(first32), "e32": e32,
        "transfer_solved": sum(r["solved"] for r in rows), "sigma": sig,
        "env_stats": {k: (dict(v) if hasattr(v, "items") else v) for k, v in st.items()},
        "secs_gen": t_gen, "secs": t_all, "host": os.uname().nodename,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }  # fmt: skip
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    with open(Path(a.out).with_suffix(".proofs.jsonl"), "w") as f:
        for n in solved:
            f.write(json.dumps({"name": n, "proofs": by[n]["proofs"]}) + "\n")
    print(f"{a.ckpt}: strict {len(solved)} / 106; transfer {res['transfer_solved']} / "
          f"{len(transfer)}; {t_all:.0f} s")  # fmt: skip


if __name__ == "__main__":
    main()
