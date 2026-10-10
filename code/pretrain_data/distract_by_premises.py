"""Does any model solve distracted K12 problems once the prompt has four or more premises? A premise-stratified sample of
the k12_distract10 slice (120 rows at 4 premises, 60 at 5, 120 at <= 3), pass@32 per row, on Robbie's lean_seq base or
on one of Dan's checkpoints (his environment sampler); every proof judged by the pinned verifier.

    cd code && uv run python pretrain_data/distract_by_premises.py --model robbie
    cd code && ND_REGISTRY=0 uv run python pretrain_data/distract_by_premises.py --model dan --ckpt la_T1_best12_s0_r8

Writes data/pretrain_data/screens/distract_by_prem_<label>.json (per-row hit counts by premise count) and prints the split.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "textbook_problem")]
import pool_common as PC  # noqa: E402

OUT = os.path.join(PC.DATA, "screens")


def sample_rows(seed=1, n4=120, n5=60, n3=120):
    rows = [r for r in PC.read(os.path.join(PC.DATA, "pools", "scan", "k12_distract10.jsonl.gz")) if r.get("slice")]
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for r in rows:
        by[min(r["n_prem"], 5) if r["n_prem"] >= 4 else 3].append(r)
    out = rng.sample(by[4], n4) + rng.sample(by[5], n5) + rng.sample(by[3], n3)
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["robbie", "dan"], required=True)
    ap.add_argument("--ckpt", default="la_T1_best12_s0_r8")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = sample_rows()
    t0 = time.time()
    if a.model == "robbie":
        import screen_ckpt as SC
        import eval_ckpt as EC
        model, tok, meta, fork_sample, lean_tok = EC.load_robbie(SC.RUN, "pretrain", "cpu")
        label = SC.RUN
        import k_pools as K
        hits = K.pass_hits(model, tok, fork_sample, rows, a.k, a.seed)
    else:
        import screen_ckpt_dan as SD
        import textbook72_distract_eval_dan as TD
        SD.load_dan()
        model, tok, extra = SD.load_ckpt(os.path.join(SD.CKPTS, a.ckpt + ".pt"))
        label = a.ckpt
        hits = TD.pass_hits(model, tok, rows, a.k, a.seed)
    by = collections.defaultdict(lambda: dict(n=0, solved=0, samples=0))
    for r, h in zip(rows, hits):
        b = by[r["n_prem"]]
        b["n"] += 1
        b["solved"] += (h or 0) > 0
        b["samples"] += h or 0
    res = {str(k): dict(v, pass32=round(v["solved"] / v["n"], 3), pass1=round(v["samples"] / (a.k * v["n"]), 3)) for k, v in sorted(by.items())}
    print(label, json.dumps(res), f"{time.time() - t0:.0f}s")
    json.dump(dict(label=label, k=a.k, seed=a.seed, by_prem=res,
                   rows=[dict(n_prem=r["n_prem"], n_lines=r["n_lines"], hits=h) for r, h in zip(rows, hits)]),
              open(os.path.join(OUT, f"distract_by_prem_{label}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
