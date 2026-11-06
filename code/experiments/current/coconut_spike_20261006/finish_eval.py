#!/usr/bin/env python3
# imports the fork (via harness), which is not resolvable here
# pyright: reportMissingImports=false
"""On the pod: the dev eval of a run whose harness hit TOTAL_SECS after RL but before its eval.

  python finish_eval.py <run-dir>

Loads <run-dir>/rl_round_04.pt, which holds the weights final.pt would have held, runs the
harness's own stage_eval on them (same pool, k, temperature and sampling seed), and writes
final.pt and eval_after_timeout.json. The run's status stays `timeout`: only the wall-clock rule
was broken, and the score is what the harness would have recorded with a longer limit.
"""

import json
import shutil
import sys
import types

import harness  # frozen harness: fork imports, constants, eval stage


def main():
    run_dir = sys.argv[1]
    assert harness.DEV == "cuda", "no CUDA on this pod"
    last = f"rl_round_{harness.RL_ROUNDS:02d}.pt"
    ckpt = harness.torch.load(f"{run_dir}/{last}", map_location="cpu", weights_only=True)
    seed = ckpt["seed"]
    # harness.save_checkpoint stamps each checkpoint with the hash of the pretrain.py beside it
    shutil.copy2(f"{run_dir}/source/pretrain.py", f"{harness.HERE}/pretrain.py")
    sys.path.insert(0, f"{run_dir}/source")  # this run's pretrain.py
    import pretrain

    train_recs = harness.read(f"{harness.FORK}/data/train.jsonl")
    model, tok, _ = pretrain.pretrain(train_recs, 1e-6, harness.DEV, seed, lambda m: None)
    model.load_state_dict(ckpt["state"])
    model.eval()
    dev_pool = harness.dev_split(harness.read(f"{harness.FORK}/data/ladder/transfer.jsonl"))
    heldout = harness.read(f"{harness.FORK}/data/heldout.jsonl")
    heldout = [heldout[i * len(heldout) // 2000] for i in range(2000)]
    args = types.SimpleNamespace(
        out=run_dir, seed=seed, eval_k=harness.EVAL_K, max_new=harness.MAX_NEW
    )
    ev = harness.stage_eval(args, tok, model, dev_pool, heldout)
    harness.save_checkpoint(args, model, tok, "final")
    with open(f"{run_dir}/eval_after_timeout.json", "w") as f:
        json.dump(ev, f, indent=1)


if __name__ == "__main__":
    main()
