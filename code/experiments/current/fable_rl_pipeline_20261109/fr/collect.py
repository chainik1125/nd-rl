"""One round's attempts and their processing, in-process or over several worker processes.

The attempt loop is half Python (environment, tokens, type check) and half GPU, so one process
keeps an A40 about half busy. `Collector(procs > 1)` splits the theorems over worker processes
that share the GPU; each loads the round's checkpoint, samples its shard's attempts and turns them
into candidates (finished proofs, hindsight lemmas, verifier/value examples). Lean runs in the
parent on the merged candidates.
"""

from __future__ import annotations

import multiprocessing as mp
import random

import torch

from .common import TOK, Trace, enc, n_lines, nd_pairs, pruned_nd, step_kind
from .sampler import Attempt, rollouts


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


def process_done(tr: Trace):
    """A finished attempt's trace -> candidate dict (no Lean yet) or None."""
    if not tr.done:
        return None
    nd = tr.env.nd()
    if nd.startswith("LEANPARSE"):
        return None
    pn = pruned_nd(nd)
    if pn is None:
        return None
    try:
        exA, textA = pruned_examples(tr.prompt, pn)
    except Exception:
        return None
    useful = tr.useful_for_proof()
    return {"prompt": tr.prompt, "nd": pn, "len": n_lines(pn), "textA": textA, "textB": tr.text(),
            "exA": exA, "exB": trace_examples(tr, useful) if len(useful) < len(tr.steps) else [],
            "steps": len(tr.steps)}  # fmt: skip


def process_her(tr: Trace, min_haves=4):
    """A failed attempt's trace -> its hindsight lemma candidate (largest closure) or None."""
    if not tr.ok or not tr.top_haves:
        return None
    prem = set(tr.env.prem)
    best = None
    for k, (_end, uid, f) in enumerate(tr.top_haves):
        if f in prem or f == tr.env.concl:
            continue
        c = tr.closure(uid)
        nh = sum(1 for i in c if tr.steps[i]["kind"] in ("atom", "box", "ore"))
        if nh >= min_haves and (best is None or nh > best[0]):
            best = (nh, k)
    if best is None:
        return None
    p2, acts2, useful, _end = tr.lemma(best[1])
    t2 = Trace(p2, acts2, tr.base)
    if not t2.done:
        return None
    nd = t2.env.nd()
    if nd.startswith("LEANPARSE"):
        return None
    pn = pruned_nd(nd)
    if pn is None:
        return None
    try:
        exA, textA = pruned_examples(p2, pn)
    except Exception:
        return None
    u2 = set(useful) | {len(t2.steps) - 1}
    return {"prompt": p2, "nd": pn, "len": n_lines(pn), "textA": textA, "textB": t2.text(),
            "exA": exA, "exB": trace_examples(t2, u2) if len(u2) < len(t2.steps) else [],
            "haves": best[0]}  # fmt: skip


def shard_work(model, specs, cfg):
    """specs: list of (theorem index, prompt, attempts, hindsight allowed). cfg: rollout settings
    plus `seed`, `aux_p` (share of attempts replayed for value/verifier examples), `keep`
    (finished candidates kept per theorem), `her_keep`, `her`.
    -> dict(done={ti: [candidates]}, her={ti: [candidates]}, aux=[examples], neg=[(state, action)],
            n_att, n_done, ntok, fail=Counter-like dict)."""
    atts = []
    for ti, prompt, k, _ in specs:
        atts += [Attempt(prompt, tag=ti) for _ in range(k)]
    neg = []
    samp = {}
    rollouts(model, atts, gate=cfg["gate"], temperature=cfg["temperature"], resample=cfg["resample"],
             tries=1, max_steps=cfg["max_steps"], max_action=cfg["max_action"], seed=cfg["seed"],
             collect=neg, temp_bump=cfg["temp_bump"], backtrack=cfg["backtrack"], stats=samp,
             max_calls=cfg["max_calls"], tok_budget=cfg.get("tok_budget", 700_000))  # fmt: skip
    rng = random.Random(cfg["seed"] + 17)
    her_ok = {ti for ti, _, _, h in specs if h}
    done, her, aux = {}, {}, []
    fail = {}
    n_done = 0
    solved = {a.tag for a in atts if a.done}
    for a in atts:
        want_aux = rng.random() < cfg["aux_p"]
        want_her = cfg["her"] and not a.done and a.tag in her_ok and a.tag not in solved and len(a.acts) >= 5
        if not (a.done or want_aux or want_her):
            fail[a.why] = fail.get(a.why, 0) + 1
            continue
        tr = Trace(a.prompt, a.acts, a.base)
        if a.done:
            n_done += 1
            c = process_done(tr)
            if c is not None:
                d = done.setdefault(a.tag, {})
                if c["nd"] not in d:
                    d[c["nd"]] = c
        else:
            fail[a.why] = fail.get(a.why, 0) + 1
            if want_her:
                c = process_her(tr)
                if c is not None:
                    d = her.setdefault(a.tag, {})
                    if c["prompt"] not in d or c["len"] < d[c["prompt"]]["len"]:
                        d[c["prompt"]] = c
        if want_aux and tr.ok:
            aux += trace_examples(tr, None, val=1 if a.done else 0)
    keep = cfg["keep"]
    if cfg["gate"] == "none":
        keep = 12  # nothing was checked while sampling: the parent sends up to 12 proofs per theorem to Lean
    done = {ti: sorted(d.values(), key=lambda c: (c["len"], c["steps"]))[:keep] for ti, d in done.items()}
    her = {ti: sorted(d.values(), key=lambda c: -c["haves"])[: cfg["her_keep"]] for ti, d in her.items()}
    return {"done": done, "her": her, "aux": aux, "neg": neg, "n_att": len(atts), "n_done": n_done,
            "ntok": sum(a.n_tok for a in atts), "fail": fail, "samples": samp}  # fmt: skip


_W = {}


def _work(ckpt, specs, cfg):
    from .model import load_ckpt

    torch.set_num_threads(1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_ckpt(ckpt, dev)
    out = shard_work(model, specs, cfg)
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return out


class Collector:
    def __init__(self, procs: int = 1):
        self.procs = procs
        self.pool = mp.get_context("spawn").Pool(procs) if procs > 1 else None

    def run(self, model, ckpt_path, specs, cfg):
        """`ckpt_path`: where the current model was just saved (the workers load it)."""
        if self.pool is None:
            return shard_work(model, specs, cfg)
        order = sorted(range(len(specs)), key=lambda i: -specs[i][2])
        shards = [[specs[i] for i in order[p :: self.procs]] for p in range(self.procs)]
        jobs = [(ckpt_path, sh, {**cfg, "seed": cfg["seed"] * 31 + p}) for p, sh in enumerate(shards) if sh]
        res = self.pool.starmap(_work, jobs)
        out = {"done": {}, "her": {}, "aux": [], "neg": [], "n_att": 0, "n_done": 0, "ntok": 0, "fail": {},
               "samples": {}}  # fmt: skip
        for r in res:
            out["done"].update(r["done"])
            out["her"].update(r["her"])
            out["aux"] += r["aux"]
            out["neg"] += r["neg"]
            for k in ("n_att", "n_done", "ntok"):
                out[k] += r[k]
            for k, v in r["fail"].items():
                out["fail"][k] = out["fail"].get(k, 0) + v
            for k, v in r["samples"].items():
                out["samples"][k] = out["samples"].get(k, 0) + v
        return out

    def close(self):
        if self.pool is not None:
            self.pool.terminate()


__all__ = ["Collector", "shard_work", "process_done", "process_her", "trace_examples", "TOK"]
