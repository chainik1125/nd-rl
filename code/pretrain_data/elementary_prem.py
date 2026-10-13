"""Very elementary problems at 2, 3, 4 and 5 premises (same shapes, more premises), scored on one of Dan's checkpoints
through his environment sampler with the pinned verifier as judge, k samples each. Prints per-problem solved counts
and a breakdown of what the failed samples look like (parse failure / verifier rejection reason), and a few raw
bodies, so the premise-count wall can be seen rather than inferred. None of these problems is sealed.

    cd code && ND_REGISTRY=0 uv run python pretrain_data/elementary_prem.py --ckpt la_T1_best12_s0_r8
Writes data/pretrain_data/screens/elementary_prem_<ckpt>.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "textbook_problem")]
import pool_common as PC  # noqa: E402
import screen_ckpt_dan as SD  # noqa: E402
from strict_verify import safe_verify  # noqa: E402

OUT = os.path.join(PC.DATA, "screens")

# (label, premises, goal); atoms are bare tokens, compound formulas fully parenthesised
PROBLEMS = [
    # pick one premise
    ("pick_2", ["P", "Q"], "P"),
    ("pick_3", ["P", "Q", "R"], "P"),
    ("pick_4", ["P", "Q", "R", "S"], "P"),
    ("pick_4_last", ["P", "Q", "R", "S"], "S"),
    ("pick_5", ["P", "Q", "R", "S", "( P & Q )"], "P"),
    # conjoin two premises
    ("and_2", ["P", "Q"], "( P & Q )"),
    ("and_3", ["P", "Q", "R"], "( P & Q )"),
    ("and_4", ["P", "Q", "R", "S"], "( P & Q )"),
    ("and_4_ends", ["P", "Q", "R", "S"], "( P & S )"),
    ("and_4_all", ["P", "Q", "R", "S"], "( ( P & Q ) & ( R & S ) )"),
    # one modus ponens, the rest unused
    ("mp_2", ["P", "( P > Q )"], "Q"),
    ("mp_3", ["P", "( P > Q )", "R"], "Q"),
    ("mp_4", ["P", "( P > Q )", "R", "S"], "Q"),
    ("mp_4_mid", ["R", "P", "S", "( P > Q )"], "Q"),
    ("mp_5", ["P", "( P > Q )", "R", "S", "( R > S )"], "Q"),
    # chains that use every premise
    ("chain_2", ["P", "( P > Q )"], "Q"),
    ("chain_3", ["P", "( P > Q )", "( Q > R )"], "R"),
    ("chain_4", ["P", "( P > Q )", "( Q > R )", "( R > S )"], "S"),
    ("chain_4_rev", ["( R > S )", "( Q > R )", "( P > Q )", "P"], "S"),
    # or-introduction from one premise
    ("ori_2", ["P", "Q"], "( P v R )"),
    ("ori_4", ["P", "Q", "R", "S"], "( P v ( Q & R ) )"),
]


def prompt_of(prem, goal):
    return f"THM {' , '.join(prem)} SEQ {goal} PRF"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="la_T1_best12_s0_r8", help="Dan checkpoint name, or 'robbie' for scale-d384-p1200_s0 (lean_seq)")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = [dict(label=l, prompt=prompt_of(p, g), n_prem=len(p)) for l, p, g in PROBLEMS]
    prompts = [r["prompt"] for r in rows]
    per = [dict(r, solved=0, parse_fail=0, reasons=collections.Counter(), samples=[]) for r in rows]
    t0 = time.time()
    if a.ckpt == "robbie":
        import screen_ckpt as SC
        import eval_ckpt as EC
        model, tok, meta, fork_sample, lean_tok = EC.load_robbie(SC.RUN, "pretrain", "cpu")
        a.ckpt = SC.RUN
        judged = EC.sample_set(model, tok, fork_sample, lean_tok, [dict(name=r["label"], prompt=r["prompt"]) for r in rows], a.k, 1.0, 400, 256, a.seed,
                               lambda *x: None)
        if isinstance(judged, tuple):  # (judged, secs)
            judged = judged[0]
        print(f"sampled {sum(len(j) for j in judged)} in {time.time() - t0:.0f}s", flush=True)
        for p, js in zip(per, judged):
            for x in js:
                ok, reason = bool(x["ok"]), ("ok" if x["ok"] else str(x["reason"])[:60])
                p["solved"] += ok
                p["parse_fail"] += str(x["reason"]).startswith("LEANPARSE")
                p["reasons"][reason] += 1
                if len(p["samples"]) < 4:
                    p["samples"].append(dict(ok=ok, body=str(x.get("body") or "")[:300]))
        bodies = None
    else:
        SD.load_dan()
        model, tok, extra = SD.load_ckpt(os.path.join(SD.CKPTS, a.ckpt + ".pt"))
        flat = [(i, j) for i in range(len(rows)) for j in range(a.k)]
        bodies = SD.env_generate(model, tok, [prompts[i] for i, _ in flat], greedy=False, temperature=1.0, max_action=256, max_steps=48,
                                 seed=a.seed, batch=256)
        print(f"sampled {len(bodies)} in {time.time() - t0:.0f}s", flush=True)
    for (i, j), body in (zip(flat, bodies) if bodies is not None else []):
        p = per[i]
        if not body or body.startswith("LEANPARSE"):
            p["parse_fail"] += 1
            p["reasons"][(body or "parse_fail")[:40]] += 1
            if len(p["samples"]) < 4:
                p["samples"].append(dict(ok=False, body=(body or "")[:300]))
            continue
        v = safe_verify(prompts[i], body)
        ok = bool(v[0])
        reason = "ok" if ok else str(v[1])[:60]
        p["solved"] += ok
        p["reasons"][reason] += 1
        if len(p["samples"]) < 4:
            p["samples"].append(dict(ok=ok, body=body[:300]))
    print(f"{'label':<12} {'prem':>4} {'solved/k':>9}  failure reasons")
    for p in per:
        reasons = ", ".join(f"{r} {n}" for r, n in p["reasons"].most_common(4) if r != "ok")
        print(f"{p['label']:<12} {p['n_prem']:>4} {p['solved']:>5}/{a.k:<3}  {reasons}")
    by = collections.defaultdict(lambda: [0, 0])
    for p in per:
        by[p["n_prem"]][0] += p["solved"]; by[p["n_prem"]][1] += a.k
    print("by premise count (samples verified / samples):", {k: f"{v[0]}/{v[1]}" for k, v in sorted(by.items())})
    for p in per:
        p["reasons"] = dict(p["reasons"])
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(ckpt=a.ckpt, k=a.k, seed=a.seed, problems=per), open(os.path.join(OUT, f"elementary_prem_{a.ckpt}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
