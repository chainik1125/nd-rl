#!/usr/bin/env python3
# imports the fork (via harness), which is not resolvable here
# pyright: reportMissingImports=false
"""On the pod: score one HF checkpoint on the transfer holdout (or dev) half, offline.

  python3 score_holdout.py <hf-run-dir> [--pool holdout|dev] [--stage final] --out result.json

<hf-run-dir> is `runs/<it>` in the HF repo, e.g. 153-claude_lean-width384_s0. Downloads that
run's `source/pretrain.py` and `<stage>.pt`, rebuilds the architecture by calling the run's own
`pretrain()` with a ~zero budget (it constructs the model, then its wall-clock loop exits before
step 1), loads the weights strictly, and scores with the harness's sampler and judge: same k,
temperature, max_new and sampling seed (seed * 1000 + 777) as the harness's dev eval. With
--pool dev it should reproduce the run's recorded dev metric, which is the correctness check.
Needs remote.sh's environment (fork checkout, Lean, LEAN_GATE_WORKERS).
"""

import argparse
import json
import os
import subprocess
import sys
import time

import harness  # frozen harness: fork imports, constants, sampler, judge

HF = "https://huggingface.co/robbiethompson2018/nd-rl-checkpoints/resolve/main/runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--pool", choices=["holdout", "dev"], default="holdout")
    ap.add_argument("--stage", default="final")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    work = f"/workspace/ckpt/{a.run}"
    os.makedirs(work, exist_ok=True)
    # 000-010 have no source/ on HF: src/<run>.py is rebuilt from git (003/007/008 from 001's file)
    local_src = f"src/{a.run}.py"
    if os.path.exists(local_src):
        subprocess.run(["cp", local_src, f"{work}/pretrain.py"], check=True)
    for path in ("source/pretrain.py", f"{a.stage}.pt"):
        dest = f"{work}/{os.path.basename(path)}"
        if not os.path.exists(dest):  # atomic: concurrent jobs may fetch the same checkpoint
            tmp = f"{dest}.{os.getpid()}"
            subprocess.run(["curl", "-sSfL", "-o", tmp, f"{HF}/{a.run}/{path}"], check=True)
            os.replace(tmp, dest)
    ckpt = harness.torch.load(f"{work}/{a.stage}.pt", map_location="cpu", weights_only=True)
    # 000-010 (harness d602830: 600 s, 2 rounds, EVAL_K 32) saved no seed; 010 was seed 1
    legacy = int(a.run[:3]) <= 10
    seed = ckpt.get("seed", 1 if a.run.startswith("010") else 0)
    eval_k = 32 if legacy else harness.EVAL_K

    sys.path.insert(0, work)  # this run's pretrain.py, not the incumbent's
    import pretrain

    train_recs = harness.read(f"{harness.FORK}/data/train.jsonl")

    def rebuild(budget):
        model, tok, _ = pretrain.pretrain(train_recs, budget, harness.DEV, seed, lambda m: None)
        model.load_state_dict(ckpt["state"])  # strict: fails loudly on an architecture mismatch
        return model, tok

    try:
        model, tok = rebuild(1e-6)
    except RuntimeError as e:
        # Variants that grow the model on a schedule (stacking, 140) or need steps to exist (SWA
        # tail, 131) reach their final shape in 90 s; final.pt then overwrites the weights.
        harness.log(f"rebuild with a ~0 s budget failed ({str(e)[:200]}); retrying at 90 s")
        harness.torch.cuda.empty_cache()
        model, tok = rebuild(90)
    model.eval()
    assert tok.mode == ckpt["tok_mode"], tok.mode  # pre-Lean harness has no TOK_MODE

    transfer = harness.read(f"{harness.FORK}/data/ladder/transfer.jsonl")
    pool = harness.holdout_split(transfer) if a.pool == "holdout" else harness.dev_split(transfer)
    rows, samp = harness.sample_and_judge(
        model, tok, pool, eval_k, harness.TEMP, seed * 1000 + 777,
        harness.SAMPLE_BATCH, harness.MAX_NEW,
    )  # fmt: skip
    solved = {r["name"] for r in rows if r["solved"]}
    ls, ge = harness.lstar(solved, pool)
    out = {
        "run": a.run,
        "stage": a.stage,
        "pool": a.pool,
        "seed": seed,
        "n": len(pool),
        "solved": len(solved),
        "solved_ge7": ge[7],
        "lstar": ls,
        "ge": ge,
        "by_bin": harness.by_bin(solved, pool),
        "attempts_per_theorem": eval_k,
        "samples_per_sec": samp["samples_per_sec"],
        "gpu": harness.torch.cuda.get_device_name(0) if harness.DEV == "cuda" else "cpu",
        "secs": time.time() - t0,
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    harness.log(f"{a.run} {a.pool}: ge7 {ge[7]}/{len(pool)} in {out['secs']:.0f}s")


if __name__ == "__main__":
    main()
