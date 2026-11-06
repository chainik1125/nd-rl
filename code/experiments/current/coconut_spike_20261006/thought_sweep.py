#!/usr/bin/env python3
# imports the fork (via harness), which is not resolvable here
# pyright: reportMissingImports=false
"""On the pod: score one checkpoint at several test-time thought counts (eval only, no training).

  python thought_sweep.py <run-dir> --stage final --thoughts 0 1 2 4 8 16 32

The looped spike's loop_sweep.py with `model.n_thoughts` as the knob: same checkpoint rebuild
(the run's own pretrain() at a ~zero budget, then a strict load), same pools, sampler, judge,
temperature and seed:

  textbook72  the 72 textbook problems, k = 256
  dev_long    the dev transfer problems with true length >= 10 (329), k = 64
  rl_unsolved the run's RL targets that no round solved, k = 128

pretrain.pt is scored on textbook72 only: the other two pools are empty or undefined before RL.
A thought costs one extra position per sample, so compute is nearly flat in the thought count.
Writes <run-dir>/thought_sweep_<stage>.json after every pool.
"""

import argparse
import json
import os
import random
import sys
import time

import harness  # frozen harness: fork imports, constants, sampler, judge

HERE = os.path.dirname(os.path.abspath(__file__))
POOL_K = {"textbook72": 256, "dev_long": 64, "rl_unsolved": 128}
LONG = 10


def pools(run_dir, stage, seed):
    out = {
        "textbook72": [
            {**r, "n_lines": r.get("reference_lines")}
            for split in ("dev", "train")
            for r in harness.read(f"{HERE}/textbook72/textbook_{split}.jsonl")
        ]
    }
    if stage == "pretrain":
        return out
    dev = harness.dev_split(harness.read(f"{harness.FORK}/data/ladder/transfer.jsonl"))
    out["dev_long"] = [r for r in dev if r["n_lines"] >= LONG]
    found = {r["name"] for r in harness.read(f"{run_dir}/found_targets.jsonl")}
    # the harness's own draw of its 1,500 targets
    targets = harness.read(f"{harness.FORK}/data/ladder/rl_targets.jsonl")
    targets = random.Random(seed).sample(targets, harness.N_TARGETS)
    out["rl_unsolved"] = [t for t in targets if t["name"] not in found]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--stage", default="final")
    ap.add_argument("--thoughts", type=int, nargs="*", default=[0, 1, 2, 4, 8, 16, 32])
    a = ap.parse_args()
    assert harness.DEV == "cuda", "no CUDA on this pod"
    ckpt = harness.torch.load(f"{a.run_dir}/{a.stage}.pt", map_location="cpu", weights_only=True)
    seed = ckpt["seed"]
    sys.path.insert(0, f"{a.run_dir}/source")  # this run's pretrain.py
    import pretrain

    train_recs = harness.read(f"{harness.FORK}/data/train.jsonl")
    model, tok, _ = pretrain.pretrain(train_recs, 1e-6, harness.DEV, seed, lambda m: None)
    model.load_state_dict(ckpt["state"])  # strict: fails loudly on an architecture mismatch
    model.eval()
    out = {"run_dir": a.run_dir, "stage": a.stage, "seed": seed, "results": []}
    for n in a.thoughts:
        model.n_thoughts = n
        for name, pool in pools(a.run_dir, a.stage, seed).items():
            rows, samp = harness.sample_and_judge(
                model, tok, pool, POOL_K[name], harness.TEMP, seed * 1000 + 777,
                harness.SAMPLE_BATCH, harness.MAX_NEW,
            )  # fmt: skip
            res = {
                "thoughts": n,
                "pool": name,
                "k": POOL_K[name],
                "n": len(pool),
                "solved": sum(r["solved"] for r in rows),
                "secs": samp["secs"],
                "samples_per_sec": samp["samples_per_sec"],
                "problems": {r["name"]: {"ref": r["n_lines"], "n_ok": r["n_ok"]} for r in rows},
            }
            out["results"].append(res)
            harness.log(
                f"{a.stage} thoughts {n} {name}: solved {res['solved']}/{res['n']} at "
                f"k={res['k']}, {res['samples_per_sec']:.0f} samples/s, {res['secs']:.0f}s"
            )
            with open(f"{a.run_dir}/thought_sweep_{a.stage}.json", "w") as f:
                json.dump(out, f)


if __name__ == "__main__":
    t0 = time.time()
    main()
    harness.log(f"thought sweep done in {time.time() - t0:.0f}s")
