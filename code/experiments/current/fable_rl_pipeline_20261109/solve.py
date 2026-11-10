#!/usr/bin/env python3
"""Solve theorems with a fable_rl checkpoint.

    python solve.py --ckpt C --in theorems.jsonl --out rows.jsonl --k K --seed S

Input rows: `name` and `thm` ("p1 , p2 |- c", the format of y). Output: one row per attempt with
- `name`, `draw` (0..K-1);
- `proof`: the ND proof body in Dan's format ("N1 ... ; ... QED", what `lean_judge.judge_many`
  takes), or "LEANPARSE incomplete: <why>" when the attempt produced no complete proof;
- `verdict`: the solver's own yes/no (the environment says the proof is complete and, in full
  mode, the verifier passed every step); it never comes from Lean. A full-mode attempt whose
  screened tries all fail hands in one unscreened rollout (`fallback` true, `verdict` false);
- `tokens_generated`: every token the model generated inside the attempt, with refused samples,
  abandoned boxes and restarted tries included;
- extras: `lean_text` (the same proof as the tactic text), `confidence` (lowest verifier
  probability over the proof's steps), `tries`, `steps`, `scaffold_lines`.

Mode: environment variable `SOLVER_MODE` = `full` (default) or `plain`; `--mode` overrides it.
- full: inside one attempt the model's verifier head screens each step it writes; a doubted step
  or one the environment refuses is resampled, a stuck box is abandoned, a failed try restarts.
  The premise-restating lines (line k restates premise k, the format's fixed preamble, a function
  of the statement alone) are written by the solver and cost no tokens; `scaffold_lines` counts them.
- plain: one model rollout, no resampling, no scaffold; an action the environment refuses ends it.
Neither mode calls Lean or any type checker. `--judge` runs Lean on the finished output afterwards
and prints pass rates (for scoring dev sets; not part of solving).

Everything imported is in this folder (`fr/`, `vendor_dan/`) or in the lab environment.
`SOLVER_PROCS` (default 1) worker processes use the one GPU; one process with large decode
chunks was the fastest setting measured on an A40.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODES = {"full": "search", "search": "search", "plain": "plain"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default=None, choices=sorted(MODES))
    ap.add_argument("--procs", type=int, default=int(os.environ.get("SOLVER_PROCS", "1")))
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--resample", type=int, default=None)
    ap.add_argument("--tries", type=int, default=None)
    ap.add_argument("--temp_bump", type=float, default=None)
    ap.add_argument("--backtrack", type=int, default=None)
    ap.add_argument("--max_calls", type=int, default=None)
    ap.add_argument("--subgoals", type=int, default=None)
    ap.add_argument("--ver_views", type=int, default=None)
    ap.add_argument("--scaffold", type=int, default=None, choices=[0, 1])
    ap.add_argument("--fallback", type=int, default=None, choices=[0, 1])
    ap.add_argument("--max_steps", type=int, default=128)
    ap.add_argument("--max_action", type=int, default=512)
    ap.add_argument("--tok_budget", type=int, default=0, help="padded tokens per decode chunk per worker")
    ap.add_argument("--verifier", default=None, help="verifier to use instead of the one in the checkpoint: FILE, or FILE:THR,FILE:THR for a combination")
    ap.add_argument("--judge", action="store_true", help="afterwards, run Lean on the output and print pass rates")
    a = ap.parse_args()

    import torch

    from fr.model import load_ckpt
    from fr.solvelib import fmt_table, judge, solve, solve_parallel, table
    from fr.verifier import build_verifier

    inp, out_fn, ckpt = os.path.abspath(a.inp), os.path.abspath(a.out), os.path.abspath(a.ckpt)
    cuda = torch.cuda.is_available()
    if cuda and os.environ.get("SOLVER_GPU_GB"):  # testing: cap this process to the memory of a smaller card
        tot = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ["SOLVER_GPU_GB"]) * 2**30 / tot))
    model, extra = load_ckpt(ckpt, "cpu")
    solver_cfg = extra.get("solver") or {}
    mode = MODES[a.mode or os.environ.get("SOLVER_MODE") or "full"]
    if a.verifier:  # a verifier file given separately (dev tuning): pack it with the policy
        from fr.verifier import load_verifier_spec

        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        ck.setdefault("extra", {})["verifier"] = load_verifier_spec(a.verifier)
        ckpt = out_fn + ".ckpt.tmp"
        torch.save(ck, ckpt)
        extra = ck["extra"]
    if mode == "search" and not extra.get("verifier"):
        sys.exit("this checkpoint has no verifier: use SOLVER_MODE=plain")
    # settings frozen into the checkpoint for its mode, then command-line overrides
    kw = dict(solver_cfg.get(mode) or {})
    keys = ("temperature", "tau", "resample", "tries", "temp_bump", "backtrack", "max_calls", "subgoals", "ver_views")
    for key in keys:
        if getattr(a, key) is not None:
            kw[key] = getattr(a, key)
    if a.scaffold is not None:
        kw["scaffold"] = bool(a.scaffold)
    if a.fallback is not None:
        kw["fallback"] = bool(a.fallback)
    if mode == "plain":
        kw = {key: v for key, v in kw.items() if key == "temperature"}
    rows = [json.loads(line) for line in open(inp) if line.strip()]
    t0 = time.time()
    procs = a.procs if cuda else 1
    if procs > 1:
        del model
        tb = a.tok_budget or 660_000 // procs
        out, cfg = solve_parallel(ckpt, rows, k=a.k, seed=a.seed, mode=mode, procs=procs,
                                  max_steps=a.max_steps, max_action=a.max_action, tok_budget=tb, **kw)  # fmt: skip
    else:
        dev = "cuda" if cuda else "cpu"
        model = model.to(dev)
        out, cfg = solve(model, rows, k=a.k, seed=a.seed, mode=mode, max_steps=a.max_steps,
                         max_action=a.max_action, tok_budget=a.tok_budget or 500_000,
                         verifier=build_verifier(extra, dev), **kw)  # fmt: skip
    if a.judge:
        judge(out)
    tmp = out_fn + ".tmp"
    with open(tmp, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, out_fn)
    if a.verifier and os.path.exists(out_fn + ".ckpt.tmp"):
        os.remove(out_fn + ".ckpt.tmp")
    n = len(out)
    print(json.dumps({
        "theorems": len(rows), "attempts": n, "complete": sum(r["verdict"] for r in out), "fallback_proofs": sum(r["fallback"] and not r["proof"].startswith("LEANPARSE") for r in out),
        "tokens_per_attempt": round(sum(r["tokens_generated"] for r in out) / max(1, n), 1),
        "scaffold_lines": sum(r["scaffold_lines"] for r in out),
        "secs": round(time.time() - t0, 1),
        "peak_gpu_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2) if cuda else 0.0, "config": cfg,
    }), flush=True)  # fmt: skip
    if a.judge:
        meta = {r["name"]: r for r in rows}
        print(fmt_table(table(out, meta)), flush=True)


if __name__ == "__main__":
    main()
