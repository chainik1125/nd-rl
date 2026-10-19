"""Per-theorem pass rates under one checkpoint: k plain attempts each, Lean-judged.

    P=code.experiments.current.thinking_traces_20261005.pools.passrate
    CUDA_VISIBLE_DEVICES=0 python -m $P --ckpt runs/stage1/T0_s0/model.pt \
        --sets cands.jsonl --out DIR --k 16

The OLMo-3 offline difficulty filter's measurement (arXiv 2512.13961 §4.4.2):
`sample.think_generate` (plain arm, the checkpoint's own think mode; no-think for T0) at
temperature 0.8, then `sample.judge` (Lean on every finished proof's literal text), as
`code.thinking.eval` does.
Theorems go in chunks of --chunk; each chunk's rows are appended to DIR/rows.jsonl as it finishes
(name, prompt, k, n_ok, ok, status, steps, tokens), so a killed run resumes where it stopped
(names already in rows.jsonl are skipped).  Progress goes to stdout.  Chunk c uses seed
seed * 1000 + c.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from code.thinking.ckpt import load_ckpt, run_env
from code.thinking.data import read_jsonl
from code.thinking.sample import judge, think_generate
from code.thinking.tok import has_think


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sets", required=True, help="jsonl with name, prompt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--budget", type=int, default=0, help="think budget (thinking checkpoints)")
    ap.add_argument("--max_action", type=int, default=512)
    ap.add_argument("--max_steps", type=int, default=96)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lean", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out)
    run_env(out)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    recs = read_jsonl(a.sets)
    if a.limit:
        recs = recs[: a.limit]
    rows_path = out / "rows.jsonl"
    done = {r["name"] for r in read_jsonl(rows_path)} if rows_path.exists() else set()
    todo = [r for r in recs if r["name"] not in done]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, extra = load_ckpt(a.ckpt, dev)
    think = has_think(tok) and extra.get("think_format", "think") != "none"
    print(
        f"{len(recs)} theorems, {len(done)} done, {len(todo)} to go; think={think}, k={a.k}, "
        f"device {dev}",
        flush=True,
    )
    t0 = time.time()
    n_done = n_solved = 0
    for c in range(0, len(todo), a.chunk):
        part = todo[c : c + a.chunk]
        prompts = [r["prompt"] for r in part for _ in range(a.k)]
        st: dict[str, Any] = {}
        res = think_generate(
            model,
            tok,
            prompts,
            "plain",
            a.budget if think else 0,
            0,
            think,
            a.temperature,
            False,
            a.max_action,
            a.max_steps,
            0,
            a.batch,
            a.seed * 1000 + (len(done) + c) // a.chunk,
            st,
            keep_trace=False,
        )
        res = judge(tok, prompts, res, a.lean)
        with open(rows_path, "a") as f:
            for j, r in enumerate(part):
                att = res[j * a.k : (j + 1) * a.k]
                n_ok = sum(bool(x["ok"]) for x in att)
                n_solved += n_ok > 0
                row = {
                    "name": r["name"],
                    "prompt": r["prompt"],
                    "k": a.k,
                    "n_ok": n_ok,
                    "ok": [int(bool(x["ok"])) for x in att],
                    "status": dict(collections.Counter(x["status"] for x in att)),
                    "steps": [x["steps"] for x in att],
                    "tokens": sum(x["tokens"] for x in att),
                }
                f.write(json.dumps(row) + "\n")
        n_done += len(part)
        el = time.time() - t0
        print(
            f"chunk {c // a.chunk}: {n_done}/{len(todo)} theorems, solved {n_solved} "
            f"({100.0 * n_solved / n_done:.1f}%), {el:.0f}s, "
            f"eta {el / n_done * (len(todo) - n_done):.0f}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
