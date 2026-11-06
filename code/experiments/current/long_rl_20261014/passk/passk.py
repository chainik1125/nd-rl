#!/usr/bin/env python3
# imports the fork (via harness), which is not resolvable here
# pyright: reportMissingImports=false
"""On the pod: per-theorem success counts at k samples, for pass@k curves, on textbook72 and holdout250.

  ND_ARCH=untied ND_RL=ei python3 passk/passk.py <ckpt.pt> --tag NAME --out results/NAME.json [--k 256] [--sets textbook72] [--temp 1 --chunk 1024]

Pools are the autoresearch ones (../autoresearch/score_holdout.py): textbook72 = 58 dev + 14 train textbook problems;
holdout250 = the fixed 250 of the transfer holdout half, ranked by sha1("passk:" + key). Same fork commit, so the same
250. Sampling and judging are this harness's own: T = 0.8 unless --temp, the Lean-only gate, n_loops = EVAL_LEVEL.
"""

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import harness  # noqa: E402  (overlay tokenizer + Lean-only gate, fork sampler)

TEXTBOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "textbook72")


def textbook72():
    # reference_lines (dev58 only) count total lines and are upper bounds; they only set the sampling cap here.
    # train14 has none, so it gets a 20-line cap (64 + 30 * 20 tokens), above every dev58 reference but three.
    return [{"name": r["name"], "prompt": r["prompt"], "body": r.get("reference_lines") or 20, "split": split}
            for split in ("dev", "train") for r in harness.read(f"{TEXTBOOK}/textbook_{split}.jsonl")]


def holdout250():
    transfer = harness.read(f"{harness.FORK}/data/ladder/transfer.jsonl")
    half = [r for r in transfer if int(hashlib.sha1(r["key"].encode()).hexdigest(), 16) % 2 == 1]
    top = sorted(half, key=lambda r: hashlib.sha1(f"passk:{r['key']}".encode()).hexdigest())[:250]
    return [{"name": r["name"], "prompt": r["prompt"], "body": r["n_lines"] - r["n_prem"]} for r in top]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--sets", default="textbook72,holdout250", help="comma-separated, from textbook72, holdout250")
    ap.add_argument("--temp", type=float, default=harness.TEMP)
    ap.add_argument("--chunk", type=int, help="sample k in chunks of this many per theorem (seeds 7777, 7778, ...), "
                                              "saving after each, so a large k fits in host memory")
    a = ap.parse_args()
    harness.TEMP = a.temp  # sample() reads the module global
    assert harness.DEV == "cuda", "no CUDA on this pod"
    agent = __import__("pretrain")
    train = harness.read(f"{harness.DATA}/pretrain.jsonl.gz")
    model, tok, _ = agent.pretrain(train, 1e-9, harness.DEV, 0, harness.log)  # zero steps: builds model + tokenizer
    ck = harness.torch.load(a.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ck.get("state", ck))  # pretrain.pt/final.pt wrap the state; rl_round_NNN.pt is bare
    setattr(model, harness.KNOB, harness.EVAL_LEVEL)
    model.eval()
    out = {"tag": a.tag, "ckpt": os.path.basename(a.ckpt), "sha256": harness.sha(a.ckpt), "k": a.k,
           "temp": a.temp, "sets": {}}
    for name in a.sets.split(","):
        pool = {"textbook72": textbook72, "holdout250": holdout250}[name]()
        t0, chunk, rows = time.time(), a.chunk or a.k, None
        for i, done in enumerate(range(0, a.k, chunk)):
            part = harness.judge(pool, harness.sample(model, tok, pool, min(chunk, a.k - done), 7777 + i, harness.MAX_NEW))
            for r, p in zip(part, pool, strict=True):
                r.pop("proofs")
                r["split"] = p.get("split")
            if rows is None:
                rows = part
            else:  # distinct proofs are per chunk, so written/pruned can repeat a proof across chunks
                for r, q in zip(rows, part, strict=True):
                    r["solved"] |= q["solved"]
                    for f in ("n_ok", "n_tried", "written", "pruned"):
                        r[f] += q[f]
            out["sets"][name] = {"secs": time.time() - t0, "rows": rows}
            n = done + min(chunk, a.k - done)
            harness.log(f"{a.tag} {name}: {n}/{a.k} samples, solved {sum(r['solved'] for r in rows)}/{len(rows)}, "
                        f"pass@1 {sum(r['n_ok'] for r in rows) / (n * len(rows)):.3f}, {time.time() - t0:.0f}s")
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            with open(a.out, "w") as f:
                json.dump(out, f)


if __name__ == "__main__":
    main()
