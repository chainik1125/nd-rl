"""Score Dan's checkpoints on the sealed 72 and on their distracted copy (two unused premises each), paired, through his
environment sampler with the pinned verifier as judge (the Dan-checkpoint screens' protocol). Aggregates only go to the
run folder; per-problem sample counts go to the gitignored docs/textbook-problem/textbook72/.

    cd code && ND_REGISTRY=0 nohup uv run python pretrain_data/textbook72_distract_eval_dan.py --ckpt la_T1_best12_s0_r8 \
        --out ../experiments/pretrain-data/textbook72_distract2/run > ../experiments/pretrain-data/textbook72_distract2/run/dan_s0.log 2>&1 &
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
import screen_ckpt_dan as SD  # noqa: E402
from data import load_textbook72  # noqa: E402

T72 = os.path.join(PC.ROOT, "docs", "textbook-problem", "textbook72")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    SD.load_dan()
    model, tok, extra = SD.load_ckpt(os.path.join(SD.CKPTS, a.ckpt + ".pt"))
    print("loaded", a.ckpt, sum(p.numel() for p in model.parameters()), "params", flush=True)
    orig = [dict(name=r["name"], prompt=r["prompt"]) for r in load_textbook72()]
    dist = [json.loads(l) for l in open(os.path.join(T72, "textbook72_distract2.jsonl"))]
    by_name = {r["name"]: r for r in dist}
    dist = [by_name[r["name"]] for r in orig if r["name"] in by_name]
    orig = [r for r in orig if r["name"] in by_name]
    summary = dict(ckpt=a.ckpt, k=a.k, seed=a.seed, n=len(orig))
    per = {}
    for label, rows in (("original", orig), ("distract2", dist)):
        t0 = time.time()
        hits = SD.pass_hits(model, tok, rows, a.k, a.seed) if hasattr(SD, "pass_hits") else None
        if hits is None:
            hits = pass_hits(model, tok, rows, a.k, a.seed)
        per[label] = hits
        summary[label] = dict(avg_solve_rate=round(sum(hits) / (a.k * len(rows)), 4),
                              any_of_k=sum(1 for h in hits if h > 0), secs=round(time.time() - t0))
        print(label, summary[label], flush=True)
    # paired, by original premise count (aggregate counts per bin only)
    bins = collections.defaultdict(lambda: dict(n=0, orig_solved=0, dist_solved=0, both=0))
    for r, ho, hd in zip(dist, per["original"], per["distract2"]):
        b = bins[min(3, r["n_prem_orig"])]
        b["n"] += 1; b["orig_solved"] += ho > 0; b["dist_solved"] += hd > 0; b["both"] += (ho > 0 and hd > 0)
    summary["by_orig_premises"] = {str(k): v for k, v in sorted(bins.items())}
    cross = [(ho, hd) for r, ho, hd in zip(dist, per["original"], per["distract2"]) if r["n_prem"] >= 4]
    summary["crossing_to_4plus"] = dict(n=len(cross), orig_solved=sum(1 for ho, _ in cross if ho > 0), dist_solved=sum(1 for _, hd in cross if hd > 0))
    stay = [(ho, hd) for r, ho, hd in zip(dist, per["original"], per["distract2"]) if r["n_prem"] <= 3]
    summary["staying_le3"] = dict(n=len(stay), orig_solved=sum(1 for ho, _ in stay if ho > 0), dist_solved=sum(1 for _, hd in stay if hd > 0))
    with open(os.path.join(a.out, "metrics.jsonl"), "a") as f:
        f.write(json.dumps(summary) + "\n")
    with open(os.path.join(T72, f"distract2_hits_{a.ckpt}_k{a.k}.json"), "w") as f:
        json.dump(dict(names=[r["name"] for r in dist], original=per["original"], distract2=per["distract2"]), f)
    print(json.dumps(summary))


def pass_hits(model, tok, rows, k, seed):
    """per-row verified samples out of k through Dan's env sampler; mirrors screen_ckpt_dan.pass_at_k"""
    from strict_verify import safe_verify
    prompts = [r["prompt"] for r in rows]
    flat = [(i, j) for i in range(len(rows)) for j in range(k)]
    bodies = SD.env_generate(model, tok, [prompts[i] for i, _ in flat], greedy=False, temperature=1.0, max_action=256, max_steps=48,
                             seed=seed, batch=256)
    hits = collections.Counter()
    for (i, j), body in zip(flat, bodies):
        if body and not body.startswith("LEANPARSE") and safe_verify(prompts[i], body)[0]:
            hits[i] += 1
    return [hits[i] for i in range(len(rows))]


if __name__ == "__main__":
    main()
