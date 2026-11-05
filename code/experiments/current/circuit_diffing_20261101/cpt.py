"""Weight-drift control (PREREG Stage 1): continue Stage-1 training of S1g3_s0 on its own
pretraining pairs with GRPO's optimizer, step count and tokens per step.

    python -m code.experiments.current.circuit_diffing_20261101.cpt --gpu 0 --lr 1.5e-4

`--base_seed S` starts from S1g3_sS (the record order stays seed 0's; output
`cpt_lr<lr>_sS.pt` in `--out_dir`). `train.finetune` (AdamW betas 0.9 / 0.95, grad clip 1): 312
steps of 400 records (about 260k tokens per step, GRPO's L2 + L3 chain trains 269k per AdamW
step), lr warmup 10 steps then cosine to 30%, weight decay 0. Saves
`/scratch/cowifornia/nd-rl/circuits/ckpts/cpt/cpt_lr<lr>.pt` in the base's format, and prints the
Stage 0 median relative weight change against the base.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .common import ART, RUNS, assert_gpu_free, pt_now

OUT = ART / "ckpts/cpt"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--steps", type=int, default=312)
    ap.add_argument("--recs", type=int, default=400)
    ap.add_argument(
        "--base_seed", type=int, default=0, help="S1g3 seed of the base (cpt_control_20261105)"
    )
    ap.add_argument("--out_dir", default=str(OUT))
    a = ap.parse_args(argv)
    assert_gpu_free(a.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    import torch

    from code.thinking import train as T
    from code.thinking.ckpt import run_env, save_ckpt

    from .common import load_model, state_dict
    from .stage0_weights import matrices

    run_env(ART / "runenv")
    base = RUNS / f"stage1/S1g3_s{a.base_seed}/model.pt"
    model, tok, extra = load_model(base, "cuda")
    for p in model.parameters():
        p.requires_grad_(True)
    data = T.load_data(RUNS / "data/S1g3.pkl", tok)
    recs = data["records"]
    print(f"{pt_now()} CPT lr {a.lr}: {len(recs)} records", flush=True)
    model, last = T.finetune(
        model,
        tok,
        recs,
        steps=a.steps,
        lr=a.lr,
        min_lr=0.3 * a.lr,
        warmup=10,
        wd=0.0,
        recs=a.recs,
        seed=0,  # the record order: the same for every base seed
        dev="cuda",
        log_every=26,
    )
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = "" if a.base_seed == 0 else f"_s{a.base_seed}"
    path = out / f"cpt_lr{a.lr:g}{tag}.pt"
    assert not path.exists(), f"{path} exists"
    ex = dict(
        extra,
        cpt=dict(init=str(base), lr=a.lr, steps=a.steps, recs=a.recs, last=last, data_seed=0),
    )
    save_ckpt(path, model, tok.mode, extra=ex)
    W0 = matrices(state_dict(base))
    W1 = matrices(state_dict(path))
    rel = [float((W1[k] - W0[k]).norm() / W0[k].norm()) for k in W0 if k.startswith("b")]
    info = dict(path=str(path), lr=a.lr, median_rel=float(np.median(rel)), last_loss=last)
    path.with_suffix(".json").write_text(json.dumps(info))
    print(f"{pt_now()} {json.dumps(info)}", flush=True)
    del torch
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
