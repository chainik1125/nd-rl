"""Arm A of base_saturation_20261105 (PREREG): finish Stage 1's anneal of S1g3_s0, nothing else.

    cd <the cd089700 code snapshot> &&
    python -m code.experiments.current.base_saturation_20261105.anneal \
        --base S1g3_s0/model.pt --data S1g3.pkl --heldout heldout.jsonl --out OUT

Stage 1's own trainer (`code.thinking.train.train_best`, the copy of `state_train_best.train` that
can start from a given model) with its optimizer split (Muon on the block, head and MTP matrices,
AdamW lr 1e-3 x scale, wd 0.1, betas 0.9 / 0.95 on the rest), token budget, grad clip and data, on
S1g3's own pairs. Changed, and only these:
  - the schedule: `--lr_scale 0.3` makes both peaks equal to where Stage 1 ended (its cosine floor,
    MIN_LR_FRAC 0.3 of 1e-3 for AdamW and of MUON_LR for Muon); a linear warmup of `--warmup` (50)
    steps up to that floor, because Stage 1 saved no optimizer state (no AdamW moments, no Muon
    momentum); then a cosine to 0 (MIN_LR_FRAC 0) at step `--steps` (1,200, 5% of Stage 1's 24,000);
  - `--mtp 0`: Stage 1 did not save its MTP head, and a fresh one (loss ~ln 118) would set the
    direction of the first Muon updates of the shared trunk and head. With weight 0 the recipe's
    `mtp_loss` returns the main loss only.
Validation loss as Stage 1 logs it (the first 2,000 held-out records, action tokens, empty think
block, effort 0), before training (the base) and along the run. Saves OUT/model.pt in the base's
checkpoint format (the base's format keys kept) and OUT/train.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from code.thinking import stateview
from code.thinking import train as T
from code.thinking.ckpt import load_ckpt, run_env, save_ckpt
from code.thinking.dan import bare

KEEP = ("think_format", "state_format", "no_trace_mode", "has_traces")  # the base's format keys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--floor", type=float, default=0.3, help="Stage 1's MIN_LR_FRAC: the start lr")
    ap.add_argument("--mtp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--curve_every", type=float, default=5.0, help="seconds between curve points")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no_compile", action="store_true")
    x = ap.parse_args(argv)
    assert x.floor == T.STB.MIN_LR_FRAC, (T.STB.MIN_LR_FRAC, x.floor)  # Stage 1's floor
    T.STB.WARMUP = x.warmup
    T.STB.MIN_LR_FRAC = 0.0
    targv = ["--data", x.data, "--heldout", x.heldout, "--out", x.out, "--seed", str(x.seed)]
    targv += ["--best_steps", str(x.steps), "--lr_scale", str(x.floor), "--mtp", str(x.mtp)]
    targv += ["--curve_every", str(x.curve_every), "--max_len", "1024"]
    targv += ["--no_compile"] if x.no_compile else []
    a = T.parser().parse_args(targv)
    out = Path(a.out)
    assert not (out / "model.pt").exists(), f"{out / 'model.pt'} exists"
    run_env(out)
    record = T.D.record
    data = T.load_data(a.data)
    tok = T.tokenizer_for(data)
    fmt = T.think_format(data)
    assert fmt == "think", fmt
    sfmt = stateview.set_state_format(stateview.data_format(data))
    model, btok, bextra = load_ckpt(x.base, "cpu", set_format=False)
    assert btok.mode == tok.mode, (btok.mode, tok.mode)
    assert stateview.ckpt_format(bextra) == sfmt, (bextra.get("state_format"), sfmt)
    cfg = model.cfg
    a.n_layer, a.d, a.n_head, a.d_ff = cfg["n_layer"], cfg["d"], cfg["n_head"], cfg["d_ff"]
    assert cfg["max_len"] == a.max_len == 1024, cfg
    records = data["records"]
    held = T.with_effort(T.heldout_records(a.heldout, tok, data["no_trace_mode"]), tok, 0)
    sched = dict(
        base=str(x.base),
        steps=x.steps,
        warmup=x.warmup,
        start_frac=x.floor,
        end_frac=0.0,
        adamw_start_lr=a.lr * x.floor,
        mtp=x.mtp,
        shape="linear warmup to the floor, then cosine to 0 (train_best; WARMUP, MIN_LR_FRAC set)",
    )
    with bare():
        record.save_config({**vars(a), "anneal": sched}, str(out) + "/", role="anneal")
    dev = x.device or ("cuda" if torch.cuda.is_available() else "cpu")
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda")
    vd = T.make_data(held, fmt, tok, dev)
    t0 = time.time()
    base_val = vd.val_loss(model.to(dev), tok, autocast)
    base_val_action = vd.val_loss(model, tok, autocast, w=0.0)
    print(
        f"base val {base_val:.4f} (action {base_val_action:.4f}) in {time.time() - t0:.0f}s",
        flush=True,
    )
    torch.manual_seed(a.seed)
    model, extra = T.train_best(a, tok, records, held, dev, record, init_model=model, fmt=fmt)
    extra["final_val_action"] = vd.val_loss(model, tok, autocast, w=0.0)
    extra.update(
        args=vars(a),
        anneal=dict(sched, base_val=base_val, base_val_action=base_val_action),
        n_params=model.n_params(),
        pairs=sum(len(r) for r in records),
        records=len(records),
        data=str(a.data),
    )
    full: dict[str, Any] = {k: bextra[k] for k in KEEP if k in bextra}
    full.update(extra)
    ck = out / "model.pt"
    save_ckpt(ck, model, tok.mode, extra=full)
    (out / "train.json").write_text(json.dumps(full, indent=1, default=str))
    print(
        f"saved {ck}: val {base_val:.4f} -> {extra['final_val']:.4f}, action "
        f"{base_val_action:.4f} -> {extra['final_val_action']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
