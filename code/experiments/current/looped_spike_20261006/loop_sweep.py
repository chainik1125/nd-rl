#!/usr/bin/env python3
# imports the fork (via harness), which is not resolvable here
# pyright: reportMissingImports=false
"""On the pod: score one checkpoint at several test-time loop counts (eval only, no training).

  python loop_sweep.py <run-dir> --stage final --loops 1 2 3 4 6 8 12

<run-dir> is a harness output directory (source/pretrain.py and <stage>.pt). The model is rebuilt
the way ../autoresearch/score_holdout.py does it (the run's own pretrain() at a ~zero budget, then
a strict load), and for each loop count `model.n_loops` is set and two pools are sampled with the
harness's sampler, judge, temperature and seed:

  textbook72  the 72 textbook problems, k = 256
  dev_long    the dev transfer problems with true length >= 10 (329), k = 64
  rl_unsolved the run's RL targets that no round solved (<run-dir>/found_targets.jsonl lists the
              solved ones), k = 128; not for pretrain.pt, which precedes RL

Every problem's success count is recorded, so pass@m for any m <= k is exact afterwards and "more
loops" can be compared with "more samples" at equal compute. A model without `n_loops` (an
untied baseline) is scored once, as loops 0. Writes <run-dir>/loop_sweep_<stage>.json after
every loop count. Needs remote.sh's environment (fork checkout, Lean, LEAN_GATE_WORKERS).
"""

import argparse
import json
import os
import random
import sys
import time

import harness  # frozen harness: fork imports, constants, sampler, judge

HERE = os.path.dirname(os.path.abspath(__file__))
POOL_K = {"textbook72": 256, "dev_long": 64, "rl_unsolved": 128, "dev": 64}
LONG = 10


def pools(run_dir, stage, seed):
    textbook = [
        {**r, "n_lines": r.get("reference_lines")}
        for split in ("dev", "train")
        for r in harness.read(f"{HERE}/textbook72/textbook_{split}.jsonl")
    ]
    dev = harness.dev_split(harness.read(f"{harness.FORK}/data/ladder/transfer.jsonl"))
    out = {"textbook72": textbook, "dev_long": [r for r in dev if r["n_lines"] >= LONG]}
    out["dev"] = dev  # the harness's whole dev pool (every theorem has true length >= 7)
    if stage != "pretrain" and os.path.exists(f"{run_dir}/found_targets.jsonl"):
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
    ap.add_argument("--loops", type=int, nargs="*", default=[1, 2, 3, 4, 6, 8, 12])
    ap.add_argument("--limit", type=int, default=0, help="first N problems per pool (smoke test)")
    ap.add_argument("--pools", nargs="*", default=["textbook72", "dev_long", "rl_unsolved"])
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
    looped = hasattr(model, "n_loops")
    n_blocks = len(model.blocks)
    out = {"run_dir": a.run_dir, "stage": a.stage, "seed": seed, "results": []}
    for n in a.loops if looped else [0]:
        if looped:
            model.n_loops = n
        for name, pool in pools(a.run_dir, a.stage, seed).items():
            if name not in a.pools:
                continue
            pool = pool[: a.limit or None]
            # blocks applied per token: entry + core * n + exit, or every untied block
            depth = 2 + (n_blocks - 2) * n if looped else n_blocks
            # one KV cache per block application: past 34 they no longer fit in 48 GB at 1024
            batch = harness.SAMPLE_BATCH * min(34, depth) // depth
            rows, samp = harness.sample_and_judge(
                model, tok, pool, POOL_K[name], harness.TEMP, seed * 1000 + 777,
                batch, harness.MAX_NEW,
            )  # fmt: skip
            out["results"].append(
                {
                    "loops": n,
                    "depth": depth,
                    "pool": name,
                    "k": POOL_K[name],
                    "n": len(pool),
                    "solved": sum(r["solved"] for r in rows),
                    "secs": samp["secs"],
                    "samples_per_sec": samp["samples_per_sec"],
                    "problems": {r["name"]: {"ref": r["n_lines"], "n_ok": r["n_ok"]} for r in rows},
                }
            )
            res = out["results"][-1]
            harness.log(
                f"{a.stage} loops {n} (depth {res['depth']}) {name}: solved {res['solved']}/"
                f"{res['n']} at k={res['k']}, {res['samples_per_sec']:.0f} samples/s, "
                f"{res['secs']:.0f}s"
            )
            with open(f"{a.run_dir}/loop_sweep_{a.stage}.json", "w") as f:
                json.dump(out, f)


if __name__ == "__main__":
    t0 = time.time()
    main()
    harness.log(f"loop sweep done in {time.time() - t0:.0f}s")
