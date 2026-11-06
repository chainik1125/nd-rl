#!/usr/bin/env python3
# imports the fork (via harness), which is not resolvable here
# pyright: reportMissingImports=false
"""On the pod: continue expert iteration from a finished harness run. Not the frozen protocol.

  python more_rl.py <run-dir> --rounds 8

Loads <run-dir>/final.pt (the model after the harness's 4 rounds) and runs the harness's own
stage_rl for more rounds on the same 1,500 targets with the same settings, then its dev eval.
Writes <run-dir>/more_rl/: final.pt, rl_round_NN.pt, source/, summary_more_rl.json, and a
found_targets.jsonl holding the proofs from every round, the harness's included, so loop_sweep.py
on this directory takes "unsolved" to mean unsolved after all of them.

stage_rl starts with an empty proof set, so its per-round counts restart; `cumulative_solved` in
the summary is the count including the harness's rounds.
"""

import argparse
import json
import os
import random
import shutil
import sys
import time
import types

import harness  # frozen harness: fork imports, constants, RL and eval stages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--rounds", type=int, default=8)
    a = ap.parse_args()
    assert harness.DEV == "cuda", "no CUDA on this pod"
    out = f"{a.run_dir}/more_rl"
    os.makedirs(out, exist_ok=True)
    shutil.copytree(f"{a.run_dir}/source", f"{out}/source", dirs_exist_ok=True)
    # harness.save_checkpoint stamps each checkpoint with the hash of the pretrain.py beside it
    shutil.copy2(f"{a.run_dir}/source/pretrain.py", f"{harness.HERE}/pretrain.py")
    ckpt = harness.torch.load(f"{a.run_dir}/final.pt", map_location="cpu", weights_only=True)
    seed = ckpt["seed"]
    sys.path.insert(0, f"{a.run_dir}/source")  # this run's pretrain.py
    import pretrain

    train_recs = harness.read(f"{harness.FORK}/data/train.jsonl")
    model, tok, _ = pretrain.pretrain(train_recs, 1e-6, harness.DEV, seed, lambda m: None)
    model.load_state_dict(ckpt["state"])
    model.eval()
    # the harness's own pools
    targets = harness.read(f"{harness.FORK}/data/ladder/rl_targets.jsonl")
    targets = random.Random(seed).sample(targets, harness.N_TARGETS)
    dev_pool = harness.dev_split(harness.read(f"{harness.FORK}/data/ladder/transfer.jsonl"))
    heldout = harness.read(f"{harness.FORK}/data/heldout.jsonl")
    heldout = [heldout[i * len(heldout) // 2000] for i in range(2000)]
    args = types.SimpleNamespace(
        out=out, seed=seed, rl_rounds=a.rounds, k=harness.K, eval_k=harness.EVAL_K,
        max_new=harness.MAX_NEW, ft_steps=harness.FT_STEPS, retain=harness.RETAIN,
    )  # fmt: skip
    harness.log(f"more_rl: continuing {a.run_dir} for {a.rounds} rounds")
    t0 = time.time()
    rl = harness.stage_rl(args, tok, model, targets, train_recs, random.Random(seed + 1))
    ev = harness.stage_eval(args, tok, model, dev_pool, heldout)
    harness.save_checkpoint(args, model, tok, "final")
    before = harness.read(f"{a.run_dir}/found_targets.jsonl")
    new = harness.read(f"{out}/found_targets.jsonl")
    solved = {r["name"] for r in before}
    cumulative = [
        len(solved | {r["name"] for r in new if r["round"] <= i}) for i in range(1, a.rounds + 1)
    ]
    with open(f"{out}/found_targets.jsonl", "a") as f:
        for r in before:
            f.write(json.dumps(r) + "\n")
    with open(f"{out}/summary_more_rl.json", "w") as f:
        json.dump(
            {
                "rounds_before": harness.RL_ROUNDS,
                "solved_before": len(solved),
                "cumulative_solved": cumulative,
                "rl": rl,
                "eval": ev,
                "secs": time.time() - t0,
            },
            f,
            indent=1,
        )
    harness.log(f"more_rl done: cumulative solved {cumulative}, dev score {ev['solved_ge7']}")


if __name__ == "__main__":
    main()
