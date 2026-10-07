#!/usr/bin/env python3
"""Sampling-temperature sweep for Dan's la_T1_best12_s0_r16 prover (runs on the pod).

Reuses Dmitry's selection-sprint code unchanged (nd-rl dmitry/main,
proof_selection_20261006/code, uploaded to ./dm): `collect_pools.collect` samples and records
every attempt, `collect_pools.judge` asks Lean, and `score_loglik.py` teacher-forces every attempt
at temperature 1. Only the sampling temperature varies, so the T=0.8 pools should reproduce his
numbers.

  python sweep_remote.py --out /workspace/tempsweep/out [--temps 0 0.8] [--k 8] [--limit 4]
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "dm"))
import collect_pools as CP  # noqa: E402
import pool_lib as PL  # noqa: E402
from model import load_ckpt  # noqa: E402

TEMPS = [0, 0.1, 0.3, 0.5, 0.8, 1, 1.5, 2]
SETS = ["textbook72", "holdout250"]


def log(s):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {s}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--temps", type=float, nargs="*", default=TEMPS)
    ap.add_argument("--sets", nargs="*", default=SETS)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="first N theorems only (smoke test)")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "pools"), exist_ok=True)
    model, tok, _ = load_ckpt(PL.CKPT, "cuda")
    sample = CP.generate_ids_fast
    for T in a.temps:
        greedy = T == 0
        # collect() hardcodes greedy=False; T=0 is argmax decoding, so one attempt per theorem
        CP.generate_ids_fast = (
            (lambda *x, **kw: sample(*x, **{**kw, "greedy": True})) if greedy else sample
        )
        k = 1 if greedy else a.k
        # above T=1 most actions run to the 512-token cap, and 2048 such rows may not fit the KV
        # cache in 48 GB
        batch = CP.BATCH if T <= 1 else 512
        for s in a.sets:
            stem = f"{s}__s0_r16_T{T:g}__k{k}"
            pool = os.path.join(a.out, "pools", stem + ".jsonl")
            summary = pool[: -len(".jsonl")] + ".summary.json"
            if os.path.exists(summary):
                log(f"{stem}: done already, skipping")
                continue
            theorems = PL.load_set(s)[: a.limit or None]
            prompts = [th["prompt"] for th in theorems for _ in range(k)]
            log(f"{stem}: {len(theorems)} theorems x k={k}, batch {batch}")
            t0 = time.time()
            recs, info = CP.collect(
                model, tok, prompts, seed=0, temperature=T or 1.0, batch=batch, log=log
            )
            sample_s = time.time() - t0
            PL.write_jsonl(
                pool,
                [
                    dict(
                        name=th["name"],
                        prompt=th["prompt"],
                        set=s,
                        gen=PL.GEN,
                        k=k,
                        sampling=dict(
                            temperature=T, max_action=CP.MAX_ACTION, max_steps=CP.MAX_STEPS, seed=0
                        ),
                        attempts=[
                            dict({x: v for x, v in recs[t * k + j].items() if x != "_why"}, i=j)
                            for j in range(k)
                        ],
                    )
                    for t, th in enumerate(theorems)
                ],
            )
            valid, reason, lean_s = CP.judge(prompts, recs)
            PL.write_jsonl(
                pool[: -len(".jsonl")] + ".labels.jsonl",
                [
                    dict(
                        name=th["name"],
                        valid=valid[t * k : (t + 1) * k],
                        reason=reason[t * k : (t + 1) * k],
                    )
                    for t, th in enumerate(theorems)
                ],
            )
            t1 = time.time()
            subprocess.run(
                [
                    sys.executable,
                    os.path.join(HERE, "dm", "score_loglik.py"),
                    "--pool",
                    pool,
                    "--out_dir",
                    os.path.join(a.out, "scores"),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            n = len(recs)
            summ = dict(
                stem=stem,
                set=s,
                temperature=T,
                k=k,
                n_theorems=len(theorems),
                attempts=n,
                finished_rate=sum(r["done"] for r in recs) / n,
                validity_rate=sum(valid) / n,
                pass_at_k=sum(any(valid[t * k : (t + 1) * k]) for t in range(len(theorems)))
                / len(theorems),
                gen_tokens=sum(r["gen_tokens"] for r in recs),
                sample_s=sample_s,
                lean_s=lean_s,
                score_s=time.time() - t1,
                **info,
            )
            with open(summary, "w") as f:
                json.dump(summ, f, indent=1)
            log(json.dumps(summ))


if __name__ == "__main__":
    main()
