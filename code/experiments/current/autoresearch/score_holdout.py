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
--pool textbook scores textbook72/ (58 dev + 14 train textbook problems, never trained on) instead
of the transfer pool, at k = TEXTBOOK_K. --pool holdout250 scores a fixed 250-problem subset of
the holdout at k = PASSK_K and records every problem's success count, for pass@k curves.
Needs remote.sh's environment (fork checkout, Lean, LEAN_GATE_WORKERS).
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import harness  # frozen harness: fork imports, constants, sampler, judge

HF = "https://huggingface.co/robbiethompson2018/nd-rl-checkpoints/resolve/main/runs"
TEXTBOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "textbook72")
DEV_NAMES = {r["name"] for r in harness.read(f"{TEXTBOOK}/textbook_dev.jsonl")}
TEXTBOOK_K = 256  # 72 problems: pass@256 costs about what k=64 costs on the 1,177 holdout
PASSK_N, PASSK_K = 250, 256  # 64k samples, about one k=64 holdout scoring


def textbook_pool():
    # train14 has no reference_lines; dev58's are upper bounds (a shorter proof may exist)
    return [
        {**r, "n_lines": r.get("reference_lines")}
        for split in ("dev", "train")
        for r in harness.read(f"{TEXTBOOK}/textbook_{split}.jsonl")
    ]


def textbook_summary(rows):
    """Per-split solve counts plus, per solved problem, the shortest pruned proof found."""
    solved = [r for r in rows if r["solved"]]
    return {
        "solved_dev": sum(r["name"] in DEV_NAMES for r in solved),
        "solved_train": sum(r["name"] not in DEV_NAMES for r in solved),
        "solved_ref_ge7": sum((r["n_lines"] or 0) >= 7 for r in solved),
        "problems": {
            r["name"]: {"ref": r["n_lines"], "n_ok": r["n_ok"], "min_pruned": min(r["pruned_lens"])}
            for r in solved
        },
    }


def holdout250(transfer):
    """The same 250 holdout problems for every checkpoint, by a hash independent of the split's."""
    rank = lambda r: hashlib.sha1(f"passk:{r['key']}".encode()).hexdigest()  # noqa: E731
    return sorted(harness.holdout_split(transfer), key=rank)[:PASSK_N]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument(
        "--pool", choices=["holdout", "dev", "textbook", "holdout250"], default="holdout"
    )
    ap.add_argument("--stage", default="final")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    # A pod with broken CUDA silently falls back to CPU and samples for hours (2026-09-27).
    assert harness.DEV == "cuda", "no CUDA on this pod"
    t0 = time.time()
    work = f"/workspace/ckpt/{a.run}"
    os.makedirs(work, exist_ok=True)
    for path in ("source/pretrain.py", f"{a.stage}.pt"):
        dest = f"{work}/{os.path.basename(path)}"
        if not os.path.exists(dest):  # atomic: concurrent jobs may fetch the same checkpoint
            tmp = f"{dest}.{os.getpid()}"
            subprocess.run(["curl", "-sSfL", "-o", tmp, f"{HF}/{a.run}/{path}"], check=True)
            os.replace(tmp, dest)
    ckpt = harness.torch.load(f"{work}/{a.stage}.pt", map_location="cpu", weights_only=True)
    seed = ckpt["seed"]

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
    assert tok.mode == harness.TOK_MODE, tok.mode

    k = harness.EVAL_K
    if a.pool == "textbook":
        pool, k = textbook_pool(), TEXTBOOK_K
    else:
        transfer = harness.read(f"{harness.FORK}/data/ladder/transfer.jsonl")
        if a.pool == "holdout250":
            pool, k = holdout250(transfer), PASSK_K
        else:
            split = harness.holdout_split if a.pool == "holdout" else harness.dev_split
            pool = split(transfer)
    rows, samp = harness.sample_and_judge(
        model, tok, pool, k, harness.TEMP, seed * 1000 + 777,
        harness.SAMPLE_BATCH, harness.MAX_NEW,
    )  # fmt: skip
    solved = {r["name"] for r in rows if r["solved"]}
    out = {"run": a.run, "stage": a.stage, "pool": a.pool, "seed": seed, "n": len(pool)}
    out["solved"] = len(solved)
    if a.pool == "textbook":
        out |= textbook_summary(rows)
    else:
        ls, ge = harness.lstar(solved, pool)
        out |= {"solved_ge7": ge[7], "lstar": ls, "ge": ge, "by_bin": harness.by_bin(solved, pool)}
        if a.pool == "holdout250":  # unsolved problems too: pass@k averages over all of them
            out["problems"] = {r["name"]: {"ref": r["n_lines"], "n_ok": r["n_ok"]} for r in rows}
    out |= {
        "attempts_per_theorem": k,
        "samples_per_sec": samp["samples_per_sec"],
        "gpu": harness.torch.cuda.get_device_name(0) if harness.DEV == "cuda" else "cpu",
        "secs": time.time() - t0,
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    harness.log(f"{a.run} {a.pool}: solved {len(solved)}/{len(pool)} in {out['secs']:.0f}s")


if __name__ == "__main__":
    main()
