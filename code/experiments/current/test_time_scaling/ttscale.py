#!/usr/bin/env python3
# imports the fork (via harness) and a checkpoint's pretrain.py, neither resolvable here
# pyright: reportMissingImports=false
"""Run one test-time method on one checkpoint and pool; write per-theorem results as JSON.
Methods and definitions: test_time_scaling_initial_plan.md.

  python3 ttscale.py --run combo-c005_s0 --pool holdout --method plain --k 64 --out x.json
  python3 ttscale.py ... --method guided --k 64 --cap 32   # rejections logged, so any cap <= 32
  python3 ttscale.py ... --method guided_kv --k 64 --cap 10  # guided.py itself: wall-clock only
  python3 ttscale.py ... --method beam --B 4 --E 4
  python3 ttscale.py ... --method smc --N 16
  python3 ttscale.py ... --method bfs --E 8 --alpha 0.5 --budget 16000

--run is a run dir in the HF repo (runs/<run>): its source/pretrain.py supplies the model classes
(186-claude's ALiBiGPT + PeriLNBlock for c005) and final.pt the weights. --ckpt-dir reads the
same two files from disk instead. Needs remote.sh's environment (ND_FORK, Lean).
"""

import argparse
import bisect
import collections
import hashlib
import heapq
import json
import os
import random
import subprocess
import sys
import time

import harness  # frozen harness: fork on sys.path, constants, pools (so it imports first)
import torch
from engine import Gate, Node, check, sample_segments
from fast_sample import _decode
from guided import STATS as GUIDED_STATS
from guided import generate as guided_generate
from nd_verify import verify_text

HF = "https://huggingface.co/robbiethompson2018/nd-rl-checkpoints/resolve/main/runs"
SEG_MAX = 400  # guided.py's lean_seq segment cap (segments span nested box headers)


def load(run, ckpt_dir):
    if ckpt_dir is None:
        ckpt_dir = f"/workspace/ckpt/{run}"
        os.makedirs(ckpt_dir, exist_ok=True)
        for path in ("source/pretrain.py", "final.pt"):
            dest = f"{ckpt_dir}/{os.path.basename(path)}"
            if not os.path.exists(dest):  # atomic: concurrent jobs may fetch the same file
                tmp = f"{dest}.{os.getpid()}"
                subprocess.run(["curl", "-sSfL", "-o", tmp, f"{HF}/{run}/{path}"], check=True)
                os.replace(tmp, dest)
    ckpt = torch.load(f"{ckpt_dir}/final.pt", map_location="cpu", weights_only=True)
    sys.path.insert(0, ckpt_dir)
    import pretrain  # the run's own model classes (186-claude lineage)
    from tokenizer import make_tokenizer  # fork

    tok = make_tokenizer(pretrain.TOK_MODE)
    model = pretrain.ALiBiGPT(tok.vocab_size, pretrain.N_LAYER, pretrain.D, pretrain.N_HEAD)
    model.blocks = torch.nn.ModuleList(
        pretrain.PeriLNBlock(pretrain.D, pretrain.N_HEAD) for _ in range(pretrain.N_LAYER)
    )
    model.load_state_dict(ckpt["state"])  # strict: fails loudly on an architecture mismatch
    assert tok.mode == harness.TOK_MODE, tok.mode
    return model.to(harness.DEV).eval(), tok, ckpt["seed"]


def pool(name, limit, names):
    if name == "textbook":  # Dmitry's 72 (dev58 + train14): score only, never tune on them
        book = os.path.join(os.path.dirname(harness.__file__), "textbook72")
        return [
            {**r, "n_lines": r.get("reference_lines")}  # train14 has no reference proof
            for split in ("dev", "train")
            for r in harness.read(f"{book}/textbook_{split}.jsonl")
        ]
    transfer = harness.read(f"{harness.FORK}/data/ladder/transfer.jsonl")
    recs = (harness.holdout_split if name == "holdout" else harness.dev_split)(transfer)
    if names:  # a comma-separated list of theorem names, e.g. the ones every method missed
        keep = set(names.split(","))
        recs = [r for r in recs if r["name"] in keep]
        assert len(recs) == len(keep), "a name is not in this pool"
    if limit:  # a fixed subset, by a hash independent of the split's
        recs = sorted(recs, key=lambda r: hashlib.sha1(f"tts:{r['key']}".encode()).hexdigest())
        recs = recs[:limit]
    return recs


# ---------------------------------------------------------------- method 0: plain sampling


def run_plain(model, tok, recs, a, gen, stats, gate):
    """The harness eval's sampler (fast_sample._decode: the fork's distribution, finished rows
    compacted out). Per sample: [ok, sampled tokens]."""
    prompts = [r["prompt"] for r in recs for _ in range(a.k)]
    ids = [tok.encode_prompt(p) for p in prompts]
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    gens = [None] * len(ids)
    t0 = time.time()
    for s in range(0, len(order), a.batch):
        chunk = order[s : s + a.batch]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=harness.DEV == "cuda"):
            outs = _decode(model, tok, [ids[i] for i in chunk], harness.TEMP, harness.MAX_NEW, gen)
        for i, o in zip(chunk, outs, strict=True):
            gens[i] = o[: o.index(tok.eos) + 1] if tok.eos in o else o
            stats["prefill_tokens"] += len(ids[i])
            stats["sampled_tokens"] += len(gens[i])
    stats["gpu_secs"] += time.time() - t0
    ok = gate(prompts, gens)
    return [
        {"samples": [[int(ok[j]), len(gens[j])] for j in range(i * a.k, (i + 1) * a.k)]}
        for i in range(len(recs))
    ]


# ---------------------------------------------------------------- method 1: guided resampling


def run_guided(model, tok, recs, a, gen, stats, gate):
    """Each sample writes segments; an invalid segment is resampled from the same prefix, until
    the sample's cap+1-th rejection abandons it. Per sample: [ok, sampled tokens, rejections as
    [cause, sampled tokens so far, repeat of an earlier rejected segment at this prefix],
    accepted segments, end ("done", "cap" or "max_new")].
    Outcomes under any smaller cap c follow (in distribution): a sample with more than c
    rejections would have been abandoned at its (c+1)-th, after that rejection's tokens.
    With --swor a redraw never repeats a segment already rejected at the same prefix: it samples
    the model's distribution with those segments removed (engine._swor_shift). Segments that hit
    the length cap without ending are not forbidden (they never ended, so there is no segment)."""
    rows = [
        {"t": i, "node": Node.root(r["prompt"]), "tok": 0, "rej": [], "bad": set(), "end": None}
        for i, r in enumerate(recs)
        for _ in range(a.k)
    ]
    for w in rows:
        w["forbid"] = collections.defaultdict(list)  # prefix length -> [(ids, tempered logprobs)]
    pids = [tok.encode_prompt(r["prompt"]) for r in recs]
    live = rows
    while live:
        items = [
            (
                pids[w["t"]] + w["node"].gen,
                min(SEG_MAX, harness.MAX_NEW - len(w["node"].gen)),
                w["forbid"][len(w["node"].gen)] if a.swor else (),
            )
            for w in live
        ]
        segs = sample_segments(model, tok, items, harness.TEMP, gen, a.batch, stats)
        t0 = time.time()
        for w, (seg, end, lp, lq) in zip(live, segs, strict=True):
            w["tok"] += len(seg)
            verdict, child = check(w["node"], seg, end, lp, tok.itos)
            if verdict == "ok":
                w["node"] = child
            elif verdict == "done":
                w["node"], w["end"] = child, "done"
            else:  # repeat: this exact segment was already rejected at this prefix
                bad = (len(w["node"].gen), tuple(seg))
                w["rej"].append([verdict, w["tok"], int(bad in w["bad"])])
                w["bad"].add(bad)
                if end != "len":
                    w["forbid"][len(w["node"].gen)].append((seg, lq))
                if len(w["rej"]) > a.cap:
                    w["end"] = "cap"
            if w["end"] is None and len(w["node"].gen) >= harness.MAX_NEW:
                w["end"] = "max_new"  # a valid prefix with no tokens left
        stats["check_secs"] += time.time() - t0
        live = [w for w in live if w["end"] is None]
    fin = [w for w in rows if w["end"] == "done"]
    ok = gate([recs[w["t"]]["prompt"] for w in fin], [w["node"].gen for w in fin])
    for w, o in zip(fin, ok, strict=True):
        w["ok"] = o
    out = [{"samples": []} for _ in recs]
    for w in rows:
        sample = [int(w.get("ok", False)), w["tok"], w["rej"], w["node"].depth, w["end"]]
        out[w["t"]]["samples"].append(sample)
    return out


def run_guided_kv(model, tok, recs, a, gen, stats, gate):
    """Method 1 as the combined pipeline's guided.generate runs it in RL: rejected segments are
    masked out of a persistent KV cache instead of re-encoding the prefix, so this is the
    wall-clock of an efficient implementation. A row ends at its cap+1-th rejection (retries and
    rollbacks both capped at `cap`) or after max_new + 256 decode steps. Per sample: [ok, None]
    (guided.py does not count tokens per row)."""
    cfg = {"guide_retries": a.cap, "guide_rollbacks": a.cap, "guide_slack": 256}
    cfg["guide_seg_max"] = SEG_MAX
    prompts = [r["prompt"] for r in recs for _ in range(a.k)]
    t0 = time.time()
    res = guided_generate(
        model, tok, prompts, harness.TEMP, harness.MAX_NEW, a.batch, a.sample_seed, cfg
    )
    stats["gpu_secs"] += time.time() - t0  # decode, line checks and the Lean gate together
    stats.update({f"guided_{k}": v for k, v in GUIDED_STATS.items()})
    ok = [int(verify_text(p + " " + r)[0]) for p, r in zip(prompts, res, strict=True)]
    return [{"samples": [[o, None] for o in ok[i * a.k : (i + 1) * a.k]]} for i in range(len(recs))]


# ---------------------------------------------------------------- method 2/3: search


def _expand(model, tok, recs, pids, jobs, gen, a, stats):
    """jobs: (theorem index, node) pairs, one segment each. Returns per job (verdict, child,
    sampled tokens), with finished children gated: verdict "solved" or "leanrej"."""
    items = [(pids[t] + n.gen, min(SEG_MAX, harness.MAX_NEW - len(n.gen))) for t, n in jobs]
    segs = sample_segments(model, tok, items, harness.TEMP, gen, a.batch, stats)
    t0 = time.time()
    res = [
        (*check(n, seg, end, lp, tok.itos), len(seg))
        for (_, n), (seg, end, lp, _) in zip(jobs, segs, strict=True)
    ]
    stats["check_secs"] += time.time() - t0
    return res


def _finish(res, jobs, recs, gate):
    """Gate the finished children in `res` in one batch; rewrite their verdicts in place."""
    fin = [i for i, r in enumerate(res) if r[0] == "done"]
    ok = gate([recs[jobs[i][0]]["prompt"] for i in fin], [res[i][1].gen for i in fin])
    for i, o in zip(fin, ok, strict=True):
        res[i] = ("solved" if o else "leanrej", res[i][1], res[i][2])


def _set_values(a, pids, pairs, stats):
    """With --value, compute V for every (theorem, node) in one batch (node.v)."""
    if a.V and pairs:
        vs = a.V([pids[t] for t, _ in pairs], [n.gen for _, n in pairs], stats)
        for (_, n), v in zip(pairs, vs, strict=True):
            n.v = v


def _theorem_state():
    """A theorem stops at the end of the round that solves it, so `tok` at the end is the cost of
    the solve, counting the round's parallel samples that did not solve it."""
    return {"solved": False, "tok": 0, "rounds": 0, "causes": collections.Counter()}


def run_beam(model, tok, recs, a, gen, stats, gate):
    """Line-level beam search: every beam node gets E sampled segments; the valid, unseen
    children are ranked by the logprob of the whole prefix (with --value: by V) and the top B
    form the next beam."""
    pids = [tok.encode_prompt(r["prompt"]) for r in recs]
    st = [_theorem_state() for _ in recs]
    beams = [[Node.root(r["prompt"])] for r in recs]
    seen = [set() for _ in recs]
    active = list(range(len(recs)))
    while active:
        jobs = [(t, n) for t in active for n in beams[t] for _ in range(a.E)]
        res = _expand(model, tok, recs, pids, jobs, gen, a, stats)
        _finish(res, jobs, recs, gate)
        kids = collections.defaultdict(list)
        for (t, _), (v, child, ntok) in zip(jobs, res, strict=True):
            s = st[t]
            s["tok"] += ntok
            s["causes"][v] += 1
            if v == "solved":
                s["solved"] = True
            elif v == "ok" and child.key not in seen[t] and len(child.gen) < harness.MAX_NEW:
                seen[t].add(child.key)
                kids[t].append(child)
        _set_values(a, pids, [(t, n) for t in active for n in kids[t]], stats)
        for t in active:
            st[t]["rounds"] += 1
            beams[t] = sorted(kids[t], key=lambda n: -(n.v if a.V else n.logp))[: a.B]
        active = [t for t in active if not st[t]["solved"] and beams[t]]
    return st


def run_smc(model, tok, recs, a, gen, stats, gate):
    """Bootstrap particle filter with a validity likelihood: each of N particles proposes one
    segment from the policy; invalid ones get weight 0; the survivors are resampled back to N
    (systematic resampling; all weights are 1). With --value it is a twisted SMC: a valid child's
    weight is V(child) / V(parent) and resampling is proportional to weight. Tracks how many
    distinct first segments (by ND lines) the particles descend from."""
    rng = random.Random(a.sample_seed)
    pids = [tok.encode_prompt(r["prompt"]) for r in recs]
    st = [_theorem_state() for _ in recs]
    parts = [[(Node.root(r["prompt"]), None)] * a.N for r in recs]  # (node, ancestor id)
    _set_values(a, pids, [(t, p[0][0]) for t, p in enumerate(parts)], stats)
    active = list(range(len(recs)))
    while active:
        jobs = [(t, n) for t in active for n, _ in parts[t]]
        anc = [x for t in active for _, x in parts[t]]
        res = _expand(model, tok, recs, pids, jobs, gen, a, stats)
        _finish(res, jobs, recs, gate)
        surv = collections.defaultdict(list)
        for j, ((t, _), (v, child, ntok)) in enumerate(zip(jobs, res, strict=True)):
            s = st[t]
            s["tok"] += ntok
            s["causes"][v] += 1
            if v == "solved":
                s["solved"] = True
            elif v == "ok" and len(child.gen) < harness.MAX_NEW:
                surv[t].append((child, child.key if anc[j] is None else anc[j], jobs[j][1]))
        _set_values(a, pids, [(t, c) for t in active for c, _, _ in surv[t]], stats)
        for t in active:
            s = st[t]
            s["rounds"] += 1
            if surv[t]:
                w = [c.v / max(p.v, 1e-6) if a.V else 1.0 for c, _, p in surv[t]]
                cum, acc = [], 0.0
                for x in w:
                    acc += x
                    cum.append(acc)
                u = rng.random()  # systematic resampling: N evenly spaced points on the CDF
                picks = [bisect.bisect_right(cum, (u + i) / a.N * acc) for i in range(a.N)]
                parts[t] = [surv[t][min(p, len(w) - 1)][:2] for p in picks]
                s["ancestors"] = len({x for _, x in parts[t]})
        active = [t for t in active if not st[t]["solved"] and surv[t]]
    return st


def run_bfs(model, tok, recs, a, gen, stats, gate):
    """Line-level best-first search: a frontier of every valid unexpanded prefix, keyed by
    logprob / segments^alpha (with --value: by V). Each round pops each theorem's best node and
    samples E segments from it; valid unseen children join the frontier. An empty frontier
    restarts from the root. Stops at a proof or at `budget` sampled tokens; the solve-rate-vs-
    budget curve for any smaller budget follows from each theorem's `tok` (anytime)."""
    pids = [tok.encode_prompt(r["prompt"]) for r in recs]
    st = [_theorem_state() for _ in recs]
    roots = [Node.root(r["prompt"]) for r in recs]
    front = [[] for _ in recs]
    seen = [set() for _ in recs]
    tie = iter(range(1 << 62))
    active = list(range(len(recs)))
    while active:
        jobs = []
        for t in active:
            node = heapq.heappop(front[t])[2] if front[t] else roots[t]
            jobs += [(t, node)] * a.E
        res = _expand(model, tok, recs, pids, jobs, gen, a, stats)
        _finish(res, jobs, recs, gate)
        new = []
        for (t, _), (v, child, ntok) in zip(jobs, res, strict=True):
            s = st[t]
            s["tok"] += ntok
            s["causes"][v] += 1
            if v == "solved":
                s["solved"] = True
            elif v == "ok" and child.key not in seen[t] and len(child.gen) < harness.MAX_NEW:
                seen[t].add(child.key)
                new.append((t, child))
        _set_values(a, pids, new, stats)
        for t, child in new:
            score = child.v if a.V else child.logp / child.depth**a.alpha
            heapq.heappush(front[t], (-score, next(tie), child))
        for t in active:
            st[t]["rounds"] += 1
        active = [t for t in active if not st[t]["solved"] and st[t]["tok"] < a.budget]
    return st


METHODS = {
    "plain": run_plain,
    "guided": run_guided,
    "guided_kv": run_guided_kv,
    "beam": run_beam,
    "smc": run_smc,
    "bfs": run_bfs,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt-dir")
    ap.add_argument("--pool", choices=["holdout", "dev", "textbook"], default="holdout")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--names", help="comma-separated theorem names: score only these")
    ap.add_argument("--method", choices=list(METHODS), required=True)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--cap", type=int, default=32)
    ap.add_argument("--swor", action="store_true", help="guided: never redraw a rejected segment")
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--E", type=int, default=4)
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--budget", type=int, default=16000)
    ap.add_argument("--value", help="beam/smc/bfs: a value head from value.py (.pt)")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--sample-seed", type=int)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    model, tok, seed = load(a.run, a.ckpt_dir)
    a.sample_seed = seed * 1000 + 777 if a.sample_seed is None else a.sample_seed
    recs = pool(a.pool, a.limit, a.names)
    a.V = None
    if a.value:
        from value import Value  # value.py imports this module, so not at the top

        a.V = Value(model, a.value)
    gen = torch.Generator(device=harness.DEV)
    gen.manual_seed(a.sample_seed)
    stats = collections.Counter()
    gate = Gate(tok, stats)
    t1 = time.time()
    per = METHODS[a.method](model, tok, recs, a, gen, stats, gate)
    stats["method_secs"] = time.time() - t1
    for r, p in zip(recs, per, strict=True):
        p |= {"name": r["name"], "L": r["n_lines"]}
        if "role" in r:  # textbook: textbook_dev (dev58) or rl_train (train14)
            p["role"] = r["role"]
    out = {
        "args": {k: v for k, v in vars(a).items() if k != "V"},
        "n": len(recs),
        "stats": dict(stats),
        "gpu": torch.cuda.get_device_name(0) if harness.DEV == "cuda" else "cpu",
        "secs": time.time() - t0,
        "theorems": per,
    }
    with open(a.out, "w") as f:
        json.dump(out, f)
    solved = sum(any(s[0] for s in p["samples"]) if "samples" in p else p["solved"] for p in per)
    harness.log(
        f"{a.method} {a.run} {a.pool}: solved {solved}/{len(recs)}, "
        f"{stats['sampled_tokens'] / len(recs):.0f} sampled tok/thm, {stats['method_secs']:.0f}s"
    )


if __name__ == "__main__":
    main()
