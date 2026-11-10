"""Solving a list of theorems with a checkpoint, and scoring the result with Lean.

Two modes of one attempt:
- "plain": one rollout in the step environment at the given temperature. An action the
  environment refuses ends the attempt (the baseline's rule).
- "search": the model's verifier head screens each atomic step (probability >= tau or the step is
  resampled, up to `resample` times); an action the environment refuses is resampled too; a try
  that gets stuck or hits the step cap is restarted, up to `tries` tries. The first complete proof
  is the attempt's one answer. Neither Lean nor any type check is consulted, and attempts do not
  see each other. `tokens` counts everything generated inside the attempt.
"""

from __future__ import annotations

import collections
import time

from .common import TOK, lean_check_nd, thm_to_prompt
from .sampler import Attempt, rollouts
from .verifier import build_verifier

DEFAULTS = {
    "plain": dict(gate="none", resample=0, tries=1, temperature=1.0),
    "search": dict(gate="verifier", resample=4, tries=4, tau=0.5, temperature=1.0, temp_bump=0.15,
                   backtrack=2, max_calls=300, scaffold=True, fallback=True),
}  # fmt: skip


def _rows(atts):
    out = []
    for a in atts:
        complete = a.done
        nd = a.env.nd() if complete else ""
        if complete and nd.startswith("LEANPARSE"):
            complete = False
        out.append({
            "name": a.tag[0],
            "draw": a.tag[1],
            "proof": nd if complete else "LEANPARSE incomplete: " + (a.why or "?"),
            "verdict": bool(complete and not a.fb),
            "tokens_generated": a.n_tok,
            "prompt": a.prompt,
            "lean_text": TOK.text(a.env.text) if complete else "",
            "confidence": round(a.min_ver, 4) if complete else 0.0,
            "tries": a.tries,
            "steps": len(a.acts),
            "scaffold_lines": a.n_scaffold,
            "fallback": bool(a.fb),
        })  # fmt: skip
    return out


def solve(model, rows, k=1, seed=0, mode="search", max_steps=128, max_action=512, log=None,
          tok_budget=700_000, verifier=None, **kw):  # fmt: skip
    """rows: dicts with `name` and `thm` -> (one output row per attempt, config used)."""
    cfg = dict(DEFAULTS[mode])
    cfg.update({key: v for key, v in kw.items() if v is not None})
    atts = []
    for r in rows:
        p = thm_to_prompt(r["thm"])
        for d in range(k):
            atts.append(Attempt(p, tag=(r["name"], d)))
    t0 = time.time()
    rollouts(model, atts, seed=seed, max_steps=max_steps, max_action=max_action, log=log,
             tok_budget=tok_budget, verifier=verifier if mode == "search" else None, **cfg)  # fmt: skip
    return _rows(atts), {"secs": time.time() - t0, "mode": mode, "max_steps": max_steps,
                         "max_action": max_action, **cfg}  # fmt: skip


def _solve_worker(ckpt, rows, k, seed, mode, max_steps, max_action, tok_budget, kw):
    import torch

    from .model import load_ckpt

    torch.set_num_threads(1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, extra = load_ckpt(ckpt, dev)
    return solve(model, rows, k=k, seed=seed, mode=mode, max_steps=max_steps, max_action=max_action,
                 tok_budget=tok_budget, verifier=build_verifier(extra, dev), **kw)  # fmt: skip


def solve_parallel(ckpt, rows, k=1, seed=0, mode="search", procs=3, max_steps=128, max_action=512,
                   tok_budget=220_000, **kw):  # fmt: skip
    """`solve` over `procs` worker processes that share the GPU (the attempt loop is half Python).
    Theorems are dealt to the workers round-robin; each worker's seed is derived from `seed`."""
    import multiprocessing as mp

    shards = [rows[p::procs] for p in range(procs)]
    jobs = [(ckpt, sh, k, seed * 131 + p, mode, max_steps, max_action, tok_budget, kw)
            for p, sh in enumerate(shards) if sh]  # fmt: skip
    t0 = time.time()
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        res = pool.starmap(_solve_worker, jobs)
    out = [r for rs, _ in res for r in rs]
    order = {r["name"]: i for i, r in enumerate(rows)}
    out.sort(key=lambda r: (order[r["name"]], r["draw"]))
    cfg = dict(res[0][1])
    cfg["secs"] = time.time() - t0
    cfg["procs"] = len(jobs)
    return out, cfg


def judge(out_rows):
    """Adds `lean_ok` to each row: Lean 4 core on the ND proof, by the route of Dan's
    `lean_judge.judge_many` (nd2lean.translate with require_all_pr=False)."""
    ok = lean_check_nd([(r["prompt"], r["proof"]) for r in out_rows])
    for r, o in zip(out_rows, ok):
        r["lean_ok"] = bool(o)
    return out_rows


def table(out_rows, meta, k_list=(1, 5)):
    """meta: name -> dict with n_lines and source. -> summary dict with pass@1 (mean over draws
    of the share solved), pass@k (solved within the first k draws), tokens per attempt, by length."""
    by = collections.defaultdict(dict)
    for r in out_rows:
        by[r["name"]][r["draw"]] = r
    groups = collections.defaultdict(list)
    for name, d in by.items():
        m = meta.get(name, {})
        groups["all"].append(d)
        groups[f"len{m.get('n_lines', '?')}"].append(d)
        groups[f"src:{m.get('source', '?')}"].append(d)
    res = {}

    def _order(kv):
        g = kv[0]
        return (0, 0, g) if g == "all" else (1, int(g[3:]), g) if g[3:].isdigit() else (2, 0, g)

    for g, ds in sorted(groups.items(), key=_order):
        n = len(ds)
        k = max(len(d) for d in ds)
        row = {"n": n, "draws": k}
        row["pass@1"] = 100.0 * sum(sum(x["lean_ok"] for x in d.values()) / len(d) for d in ds) / n
        for kk in k_list:
            if kk <= k and kk > 1:
                row[f"pass@{kk}"] = 100.0 * sum(
                    any(d[i]["lean_ok"] for i in range(kk) if i in d) for d in ds) / n  # fmt: skip
        row["complete%"] = (
            100.0 * sum(sum(bool(x["verdict"]) for x in d.values()) / len(d) for d in ds) / n
        )
        row["tok/attempt"] = sum(x["tokens_generated"] for d in ds for x in d.values()) / sum(
            len(d) for d in ds
        )
        res[g] = row
    return res


def fmt_table(res):
    keys = ["n", "pass@1", "pass@5", "complete%", "tok/attempt"]
    lines = ["group | " + " | ".join(keys), "---|" + "---|" * len(keys)]
    for g, row in res.items():
        cells = []
        for key in keys:
            v = row.get(key)
            cells.append("" if v is None else (str(v) if isinstance(v, int) else f"{v:.1f}"))
        lines.append(f"{g} | " + " | ".join(cells))
    return "\n".join(lines)
