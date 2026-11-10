"""Stage 2: reinforcement learning on the y statements (no proofs are read from y).

Each round:
  1. Attempts. Every training theorem gets K attempts sampled from the current policy in Dan's
     step environment (more for theorems with no proof yet). At training time the proof checker
     is in the loop step by step: a step with a certain Lean type error is refused and resampled
     ("search at training time"). Each refusal is kept as a negative for the verifier head.
  2. Reward. A finished attempt is sent to Lean 4 (the unpruned text the model produced and its
     dependency-pruned form). Only Lean-accepted proofs are kept.
  3. Hindsight. A failed attempt on a still-unsolved theorem is read as a proof of the top-level
     line it derived with the largest dependency closure (a lemma the model made itself); kept
     only if Lean accepts it for that lemma statement.
  4. Update. Supervised steps on a mixture: x (replay), the shortest proof known per theorem,
     other fresh proofs, hindsight lemmas, and verifier/value examples.
     Proofs enter twice: pruned (states hold only the lines the proof uses) and as produced
     (states keep every line the attempt derived; only the steps the proof uses are policy
     targets), so that the policy is trained on the long, cluttered states it meets in long proofs.

    python -m fr.rl --init /workspace/runs/pre_s0/pre.pt --out /workspace/runs/rl_s0 --seed 0
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import random
import time

import torch

from .common import (
    TOK,
    Trace,
    enc,
    lean_check,
    n_lines,
    nd_pairs,
    pruned_nd,
    step_kind,
    thm_to_prompt,
)
from .data import Mixer, Store, load_x
from .model import load_ckpt, save_ckpt
from .sampler import Attempt, rollouts
from .solvelib import fmt_table, judge, solve, table
from .train import fit, make_opt


def trace_examples(tr: Trace, useful, val=-1):
    """Examples from a replayed attempt: steps in `useful` are policy targets (None: none are);
    atomic steps are verifier positives; `val` labels every state."""
    out = []
    for i, s in enumerate(tr.steps):
        k = s["kind"]
        pol = 1 if (useful is not None and i in useful) else 0
        out.append((enc(s["state"]), enc(s["act"]), pol, 1 if k in ("atom", "pr") else -1, val))
    return out


def pruned_examples(prompt, nd):
    pairs, text = nd_pairs(prompt, nd)
    ex = [(enc(s), enc(a), 1, 1 if step_kind(a) in ("atom", "pr") else -1, -1) for s, a in pairs]
    return ex, text


def process_done(att):
    """A finished attempt -> candidate dict (no Lean yet) or None."""
    tr = Trace(att.prompt, att.acts, att.base)
    if not tr.done:
        return None
    nd = tr.env.nd()
    if nd.startswith("LEANPARSE"):
        return None
    pn = pruned_nd(nd)
    if pn is None:
        return None
    try:
        exA, textA = pruned_examples(att.prompt, pn)
    except Exception:
        return None
    useful = tr.useful_for_proof()
    return {"prompt": att.prompt, "nd": pn, "len": n_lines(pn), "textA": textA, "textB": tr.text(),
            "exA": exA, "exB": trace_examples(tr, useful) if len(useful) < len(tr.steps) else [],
            "steps": len(tr.steps)}  # fmt: skip


def process_her(att, min_haves=4):
    """A failed attempt -> its hindsight lemma candidate (largest closure) or None."""
    tr = Trace(att.prompt, att.acts, att.base)
    if not tr.ok or not tr.top_haves:
        return None
    prem = set(tr.env.prem)
    best = None
    for k, (end, uid, f) in enumerate(tr.top_haves):
        if f in prem or f == tr.env.concl:
            continue
        c = tr.closure(uid)
        nh = sum(1 for i in c if tr.steps[i]["kind"] != "exact")
        if nh >= min_haves and (best is None or nh > best[0]):
            best = (nh, k)
    if best is None:
        return None
    p2, acts2, useful, end = tr.lemma(best[1])
    t2 = Trace(p2, acts2, att.base)
    if not t2.done:
        return None
    nd = t2.env.nd()
    if nd.startswith("LEANPARSE"):
        return None
    pn = pruned_nd(nd)
    try:
        exA, textA = pruned_examples(p2, pn)
    except Exception:
        return None
    u2 = set(useful) | {len(t2.steps) - 1}
    return {"prompt": p2, "nd": pn, "len": n_lines(pn), "textA": textA, "textB": t2.text(),
            "exA": exA, "exB": trace_examples(t2, u2) if len(u2) < len(t2.steps) else [],
            "haves": best[0]}  # fmt: skip


def lean_filter(cands):
    """Keep the candidates whose pruned text Lean accepts; drop the as-produced examples of those
    whose unpruned text it rejects. -> (kept, n_rejected)."""
    items = []
    for c in cands:
        items.append((c["prompt"], c["textA"]))
        items.append((c["prompt"], c["textB"]))
    ok = lean_check(items)
    kept, rej = [], 0
    for i, c in enumerate(cands):
        if not ok[2 * i]:
            rej += 1
            continue
        if not ok[2 * i + 1]:
            rej += 1
            c["exB"] = []
        kept.append(c)
    return kept, rej


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--x", default="/workspace/data/train_k12.jsonl.gz")
    ap.add_argument("--train", default="/workspace/data/y_train.jsonl")
    ap.add_argument("--dev", default="/workspace/data/y_dev.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--hours", type=float, default=6.0, help="no new round starts after this")
    ap.add_argument("--k_unsolved", type=int, default=24)
    ap.add_argument("--k_solved", type=int, default=8)
    ap.add_argument("--resample", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--temp_bump", type=float, default=0.3, help="temperature rise per refused sample of a step")
    ap.add_argument("--backtrack", type=int, default=2, help="boxes a stuck attempt may abandon")
    ap.add_argument("--max_steps", type=int, default=96)
    ap.add_argument("--max_action", type=int, default=384)
    ap.add_argument("--steps", type=int, default=600, help="optimiser steps per round")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--w_unl", type=float, default=0.25, help="weight of the unlikelihood of refused steps")
    ap.add_argument("--tok_budget", type=int, default=120_000)
    ap.add_argument("--mix", default="x=0.35,best=0.30,recent=0.10,her=0.10,aux=0.15")
    ap.add_argument(
        "--keep_per_thm", type=int, default=3, help="fresh proofs kept per theorem per round"
    )
    ap.add_argument("--her_per_thm", type=int, default=6)
    ap.add_argument("--no_her", action="store_true")
    ap.add_argument("--no_asproduced", action="store_true", help="ablation: pruned proofs only")
    ap.add_argument(
        "--aux_attempts",
        type=int,
        default=8000,
        help="attempts replayed per round for value/verifier positives",
    )
    ap.add_argument("--neg_cap", type=int, default=250_000)
    ap.add_argument("--dev_every", type=int, default=3)
    ap.add_argument("--dev_k", type=int, default=5)
    ap.add_argument(
        "--limit", type=int, default=0, help="smoke test: use only this many training theorems"
    )
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "rl.log"), "a")

    def log(m):
        line = f"{time.strftime('%FT%TZ', time.gmtime())} {m}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    dev_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    mixw = dict(kv.split("=") for kv in a.mix.split(","))
    mixw = {k: float(v) for k, v in mixw.items()}

    train = [json.loads(l) for l in open(a.train)]
    if a.limit:
        train = train[: a.limit]
    devrows = [json.loads(l) for l in open(a.dev)]
    if a.limit:
        devrows = devrows[: max(20, a.limit // 10)]
    for r in train:
        r["prompt"] = thm_to_prompt(r["thm"])
    devmeta = {r["name"]: r for r in devrows}
    xtr, _, xmeta = load_x(a.x, os.path.join(os.path.dirname(a.x), "x_cache.pkl"))
    log(f"args {json.dumps(vars(a))}")
    log(f"train theorems {len(train)}, dev {len(devrows)}, x pairs {len(xtr)}")

    state_fn = os.path.join(a.out, "state.pkl")
    if a.resume and os.path.exists(state_fn):
        with open(state_fn, "rb") as f:
            S = pickle.load(f)
        model, _ = load_ckpt(os.path.join(a.out, "last.pt"), dev_)
        log(f"resumed at round {S['round']}")
    else:
        S = {"round": 0, "best": {}, "recent": [], "her": [], "aux": [], "elapsed": 0.0,
             "tokens_sampled": 0, "history": []}  # fmt: skip
        model, _ = load_ckpt(a.init, dev_)
    opt = make_opt(model, a.lr)
    if a.resume and os.path.exists(os.path.join(a.out, "opt.pt")):
        opt.load_state_dict(torch.load(os.path.join(a.out, "opt.pt"), map_location=dev_))
    gen = torch.Generator(device=dev_)
    gen.manual_seed(a.seed + 1000 * S["round"])
    t_run = time.time()

    def do_dev(rnd):
        res = {}
        for mode in ("plain", "search"):
            out, cfg = solve(model, devrows, k=a.dev_k, seed=10_000 + rnd, mode=mode)
            judge(out)
            tb = table(out, devmeta)
            res[mode] = tb
            log(f"dev round {rnd} mode {mode} ({cfg['secs']:.0f}s)\n{fmt_table(tb)}")
        return res

    while S["round"] < a.rounds and (S["elapsed"] + time.time() - t_run) / 3600 < a.hours:
        rnd = S["round"]
        t0 = time.time()
        best = S["best"]
        # ---- 1. attempts
        atts = []
        for ti, r in enumerate(train):
            k = a.k_solved if r["name"] in best else a.k_unsolved
            atts += [Attempt(r["prompt"], tag=ti) for _ in range(k)]
        neg = []
        rollouts(model, atts, gate="checker", temperature=a.temperature, resample=a.resample,
                 tries=1, max_steps=a.max_steps, max_action=a.max_action,
                 seed=a.seed * 100_003 + rnd, collect=neg,
            temp_bump=a.temp_bump, backtrack=a.backtrack)  # fmt: skip
        t_roll = time.time() - t0
        ntok = sum(x.n_tok for x in atts)
        S["tokens_sampled"] += ntok
        # ---- 2. finished attempts -> candidates, Lean
        t1 = time.time()
        per = collections.defaultdict(dict)  # theorem -> pruned nd -> candidate
        n_done = 0
        solved_now = set()
        for x in atts:
            if not x.done:
                continue
            n_done += 1
            c = process_done(x)
            if c is None:
                continue
            solved_now.add(x.tag)
            d = per[x.tag]
            if c["nd"] not in d:
                d[c["nd"]] = c
        cands = []
        for ti, d in per.items():
            cs = sorted(d.values(), key=lambda c: (c["len"], c["steps"]))[: a.keep_per_thm]
            for c in cs:
                c["ti"] = ti
            cands += cs
        kept, rej = lean_filter(cands)
        new_best = improved = 0
        fresh = []
        for c in kept:
            name = train[c["ti"]]["name"]
            cur = best.get(name)
            if cur is None or c["len"] < cur["len"]:
                new_best += cur is None
                improved += cur is not None
                best[name] = {"nd": c["nd"], "len": c["len"], "exA": c["exA"], "exB": c["exB"],
                              "round": rnd}  # fmt: skip
            else:
                fresh.append(c)
        # ---- 3. hindsight on failed attempts of unsolved theorems
        her = []
        her_rej = 0
        if not a.no_her:
            hper = collections.defaultdict(dict)
            for x in atts:
                if x.done or train[x.tag]["name"] in best or len(x.acts) < 5:
                    continue
                c = process_her(x)
                if c is None:
                    continue
                d = hper[x.tag]
                if c["prompt"] not in d or c["len"] < d[c["prompt"]]["len"]:
                    d[c["prompt"]] = c
            hc = []
            for ti, d in hper.items():
                hc += sorted(d.values(), key=lambda c: -c["haves"])[: a.her_per_thm]
            her, her_rej = lean_filter(hc)
        # ---- aux examples: value labels and verifier positives from a sample of attempts
        aux = []
        for x in rng.sample(atts, min(a.aux_attempts, len(atts))):
            tr = Trace(x.prompt, x.acts, x.base)
            if tr.ok:
                aux += trace_examples(tr, None, val=1 if x.done else 0)
        if len(neg) > a.neg_cap:
            neg = rng.sample(neg, a.neg_cap)
        aux += [(s, act, 2 if a.w_unl > 0 else 0, 0, -1) for s, act in neg]
        t_proc = time.time() - t1
        # ---- pools
        S["recent"] = (S["recent"] + [fresh])[-2:]
        S["her"] = (S["her"] + [her])[-3:]
        S["aux"] = (S["aux"] + [aux])[-2:]

        def proof_store(cs):
            ex = []
            for c in cs:
                ex += c["exA"]
                if not a.no_asproduced:
                    ex += c["exB"]
            return Store(ex)

        st_best = proof_store(best.values())
        st_recent = proof_store([c for grp in S["recent"] for c in grp])
        st_her = proof_store([c for grp in S["her"] for c in grp])
        st_aux = Store([e for grp in S["aux"] for e in grp])
        # ---- 4. update
        t2 = time.time()
        mixer = Mixer([(xtr, mixw["x"]), (st_best, mixw["best"]), (st_recent, mixw["recent"]),
                       (st_her, mixw["her"]), (st_aux, mixw["aux"])],
                      tok_budget=a.tok_budget, seed=a.seed * 7919 + rnd)  # fmt: skip
        warm = 100 if rnd == 0 else 0
        last = fit(model, opt, mixer, a.steps, lambda s: a.lr * min(1.0, (s + 1) / warm) if warm else a.lr,
                   log=log, log_every=200, w_unl=a.w_unl, gen=gen, tag=f"[r{rnd}] ")  # fmt: skip
        t_fit = time.time() - t2
        # ---- log
        by_len = collections.Counter()
        tot_len = collections.Counter()
        for r in train:
            tot_len[r["n_lines"]] += 1
            by_len[r["n_lines"]] += r["name"] in best
        rec = {
            "round": rnd, "attempts": len(atts), "done": n_done, "tokens": ntok,
            "solved_this_round": len(solved_now), "solved_total": len(best), "new": new_best,
            "shorter": improved, "lean_rej": rej, "fresh": len(fresh), "her": len(her),
            "her_rej": her_rej, "neg": len(neg), "aux": len(aux),
            "pairs": {"best": len(st_best), "recent": len(st_recent), "her": len(st_her), "aux": len(st_aux)},
            "solved_by_len": {str(k): f"{by_len[k]}/{tot_len[k]}" for k in sorted(tot_len)},
            "mean_best_len": sum(b["len"] for b in best.values()) / max(1, len(best)),
            "secs": {"roll": round(t_roll), "proc": round(t_proc), "fit": round(t_fit)},
            "loss": {k: (last[k] / max(1, last["n"]) if k in ("pol", "ver", "val", "unl") else last[k]) for k in last},
        }  # fmt: skip
        S["round"] = rnd + 1
        if (rnd + 1) % a.dev_every == 0 or rnd == 0:
            rec["dev"] = do_dev(rnd + 1)
            save_ckpt(
                os.path.join(a.out, f"r{rnd + 1}.pt"),
                model,
                {"stage": "rl", "round": rnd + 1, "args": vars(a)},
            )
        S["history"].append(rec)
        log("round " + json.dumps({k: v for k, v in rec.items() if k != "dev"}))
        with open(os.path.join(a.out, "rounds.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        S["elapsed"] += time.time() - t_run
        t_run = time.time()
        save_ckpt(
            os.path.join(a.out, "last.pt"),
            model,
            {"stage": "rl", "round": rnd + 1, "args": vars(a)},
        )
        torch.save(opt.state_dict(), os.path.join(a.out, "opt.pt"))
        tmp = state_fn + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(S, f, protocol=4)
        os.replace(tmp, state_fn)
    if S["round"] % a.dev_every != 0:
        rec = {"round": S["round"], "final_dev": do_dev(S["round"])}
        with open(os.path.join(a.out, "rounds.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        save_ckpt(
            os.path.join(a.out, f"r{S['round']}.pt"),
            model,
            {"stage": "rl", "round": S["round"], "args": vars(a)},
        )
    # proofs found (statement, pruned proof, length): the training-side record of what RL solved
    with open(os.path.join(a.out, "found.jsonl"), "w") as f:
        for r in train:
            b = S["best"].get(r["name"])
            if b:
                f.write(json.dumps({"name": r["name"], "n_lines": r["n_lines"], "found_len": b["len"],
                                    "round": b["round"], "proof": b["nd"]}) + "\n")  # fmt: skip
    log("RL_DONE")


if __name__ == "__main__":
    main()
