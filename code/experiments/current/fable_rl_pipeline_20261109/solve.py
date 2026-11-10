#!/usr/bin/env python3
"""Solve theorems with a fable_rl checkpoint.

    python solve.py --ckpt C --in theorems.jsonl --out rows.jsonl --k K --seed S [--mode search|plain]

Input rows: `name` and `thm` ("p1 , p2 |- c", the format of y). Output: one row per attempt with
`name`, `draw`, `proof` (the complete proof as the `lean_seq` tactic text that Dan's Lean check
takes after the theorem statement; empty if the attempt produced no complete proof), `nd` (the
same proof in ND form), `verdict` (the solver's own: "complete" or "incomplete:<why>"),
`confidence` (lowest verifier probability over the proof's steps), `tokens` (every token the model
generated inside the attempt, refused and restarted samples included), `tries`, `steps`.

The solver never calls Lean or any type checker. `--mode search` (the default for checkpoints with
a verifier head) screens steps with the model's own verifier; `--mode plain` is one rollout with
no resampling. `--judge` additionally runs Lean 4 on the output and adds `lean_ok` (for scoring).
"""

import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from fr.model import load_ckpt
from fr.solvelib import fmt_table, judge, solve, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default=None, choices=["search", "plain"])
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--resample", type=int, default=None)
    ap.add_argument("--tries", type=int, default=None)
    ap.add_argument("--temp_bump", type=float, default=None)
    ap.add_argument("--backtrack", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=128)
    ap.add_argument("--max_action", type=int, default=384)
    ap.add_argument(
        "--judge", action="store_true", help="also run Lean on the output and print pass rates"
    )
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, extra = load_ckpt(a.ckpt, dev)
    solver_cfg = extra.get("solver", {})
    mode = a.mode or solver_cfg.get("mode") or ("search" if extra.get("has_aux") else "plain")
    if mode == "search" and not extra.get("has_aux"):
        sys.exit("this checkpoint has no verifier head: use --mode plain")
    kw = {
        k: solver_cfg.get(k)
        for k in ("temperature", "tau", "resample", "tries", "temp_bump", "backtrack")
        if mode == solver_cfg.get("mode")
    }
    for k in ("temperature", "tau", "resample", "tries", "temp_bump", "backtrack"):
        if getattr(a, k) is not None:
            kw[k] = getattr(a, k)
    if mode == "plain":
        kw.pop("tau", None)
        kw.pop("temp_bump", None)
        kw.pop("backtrack", None)
    rows = [json.loads(l) for l in open(a.inp) if l.strip()]
    t0 = time.time()
    out, cfg = solve(
        model,
        rows,
        k=a.k,
        seed=a.seed,
        mode=mode,
        max_steps=a.max_steps,
        max_action=a.max_action,
        **kw,
    )
    if a.judge:
        judge(out)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(out)
    print(
        json.dumps(
            {
                "attempts": n,
                "complete": sum(bool(r["proof"]) for r in out),
                "mean_tokens_per_attempt": sum(r["tokens"] for r in out) / max(1, n),
                "secs": round(time.time() - t0, 1),
                "config": cfg,
            }
        )
    )
    if a.judge:
        meta = {r["name"]: r for r in rows}
        print(fmt_table(table(out, meta)))


if __name__ == "__main__":
    main()
