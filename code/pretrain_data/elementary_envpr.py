"""Does Dan's checkpoint prove four-premise problems once the premise block is written FOR it? The environment applies
the premise lines h1..hn itself (they are boilerplate the verifier requires), then the model acts from that state as
usual (temperature 1.0, 48 steps, 256 tokens per action); every finished proof is judged by the pinned verifier.
Sets: the elementary ladder (elementary_prem.PROBLEMS), the premise-stratified distract10 sample
(distract_by_premises.sample_rows: 120 at 4, 60 at 5, 120 at <= 3 premises) and, aggregates only, the distracted 72.

    cd code && ND_REGISTRY=0 uv run python pretrain_data/elementary_envpr.py --ckpt la_T1_best12_s0_r8
Writes data/pretrain_data/screens/elementary_envpr_<ckpt>.json (no sealed content) and appends the sealed-set
aggregates to experiments/pretrain-data/textbook72_distract2/run/metrics_envpr.jsonl.
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
import elementary_prem as EP  # noqa: E402
import distract_by_premises as DP  # noqa: E402
from strict_verify import safe_verify  # noqa: E402

OUT = os.path.join(PC.DATA, "screens")
T72 = os.path.join(PC.ROOT, "docs", "textbook-problem", "textbook72")
EXP = os.path.join(PC.ROOT, "experiments", "pretrain-data", "textbook72_distract2", "run")


def generate_with_premises(model, tok, prompts, k, seed, env_pr=True, batch=256, max_steps=48, max_action=256):
    """per prompt: list of k bodies (ND text or 'LEANPARSE …'); mirrors state_sample.env_generate, except that the
    environment applies the premise lines first when env_pr is set. Canonical names as in Dan's lean_staten arm."""
    import random
    import torch
    import state_sample as SS
    from lean_tok import ftoks
    from state_sample import NAME_BASE_MAX
    flat = [(i, j) for i in range(len(prompts)) for j in range(k)]
    res = [None] * len(flat)
    fail = collections.Counter()
    nxt, live, wave = 0, [], 0

    def make_env(n):
        base = random.Random(seed * 1000003 + n).randint(0, NAME_BASE_MAX)
        e = SS.Env(prompts[flat[n][0]], canon=True, base=base, assign=True)
        if env_pr:
            for j, p in enumerate(e.prem):
                ok, why = e.apply(["have", f"n{e.next_name()}", ":"] + ftoks(p) + [":=", f"h{j + 1}", ";"])
                assert ok, why
        return e

    def finish(n, e):
        res[n] = e.nd()

    while nxt < len(flat) or live:
        while len(live) < batch and nxt < len(flat):
            live.append((nxt, make_env(nxt))); nxt += 1
        pids = [SS.prompt_ids(tok, e) for _, e in live]
        with torch.no_grad():
            outs = SS.generate_ids_fast(model, tok, pids, goals=None, greedy=False, temperature=1.0, max_new=max_action,
                                        seed=seed * 100003 + wave, early="eos", compact=True, stats={})
        wave += 1
        keep = []
        for (n, e), o in zip(live, outs):
            atoks, ended = tok.decode_action(o)
            if not ended:
                e.failed = "action truncated"; fail["truncated"] += 1; finish(n, e); continue
            ok, why = e.apply(atoks)
            if not ok:
                fail[why] += 1; finish(n, e); continue
            if e.done:
                finish(n, e); continue
            if e.steps >= max_steps:
                e.failed = "max steps"; fail["max steps"] += 1; finish(n, e); continue
            keep.append((n, e))
        live = keep
    out = [[None] * k for _ in prompts]
    for (i, j), b in zip(flat, res):
        out[i][j] = b
    return out, fail


def score(model, tok, rows, k, seed, env_pr):
    prompts = [r["prompt"] for r in rows]
    bodies, fail = generate_with_premises(model, tok, prompts, k, seed, env_pr=env_pr)
    hits = []
    for p, bs in zip(prompts, bodies):
        hits.append(sum(1 for b in bs if b and not b.startswith("LEANPARSE") and safe_verify(p, b)[0]))
    return hits, dict(fail), bodies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="la_T1_best12_s0_r8")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_sealed", action="store_true")
    a = ap.parse_args()
    SD.load_dan()
    model, tok, extra = SD.load_ckpt(os.path.join(SD.CKPTS, a.ckpt + ".pt"))
    out = dict(ckpt=a.ckpt, k=a.k, seed=a.seed)
    # 1. elementary ladder, env-written premises vs the model writing them
    rows = [dict(label=l, prompt=EP.prompt_of(p, g), n_prem=len(p)) for l, p, g in EP.PROBLEMS]
    for env_pr in (True, False):
        t0 = time.time()
        hits, fail, bodies = score(model, tok, rows, a.k, a.seed, env_pr)
        tag = "env_premises" if env_pr else "model_premises"
        out[f"ladder_{tag}"] = dict(rows=[dict(r, hits=h, sample=next((b for b in bs if b and not b.startswith("LEANPARSE")), bs[0])[:300])
                                          for r, h, bs in zip(rows, hits, bodies)], fail=fail)
        by = collections.defaultdict(lambda: [0, 0])
        for r, h in zip(rows, hits):
            by[r["n_prem"]][0] += h; by[r["n_prem"]][1] += a.k
        print(f"ladder {tag}: by premise count {dict((n, f'{v[0]}/{v[1]}') for n, v in sorted(by.items()))}  "
              f"per problem {[(r['label'], h) for r, h in zip(rows, hits)]}  ({time.time() - t0:.0f}s)", flush=True)
    # 2. generator distractor rows, stratified by premise count
    rows = DP.sample_rows()
    t0 = time.time()
    hits, fail, _ = score(model, tok, rows, a.k, a.seed, True)
    by = collections.defaultdict(lambda: dict(n=0, solved=0, samples=0))
    for r, h in zip(rows, hits):
        b = by[r["n_prem"]]; b["n"] += 1; b["solved"] += h > 0; b["samples"] += h
    out["distract10_env_premises"] = dict(by_prem={str(n): dict(v, pass32=round(v["solved"] / v["n"], 3), pass1=round(v["samples"] / (a.k * v["n"]), 3))
                                                   for n, v in sorted(by.items())}, fail=fail,
                                          rows=[dict(n_prem=r["n_prem"], n_lines=r["n_lines"], hits=h) for r, h in zip(rows, hits)])
    print("distract10 sample, env premises:", json.dumps(out["distract10_env_premises"]["by_prem"]), f"({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, f"elementary_envpr_{a.ckpt}.json"), "w"), indent=1)
    # 3. the distracted 72 and their originals, aggregates only
    if a.no_sealed:
        return
    from data import load_textbook72
    orig = {r["name"]: r["prompt"] for r in load_textbook72()}
    dist = [json.loads(l) for l in open(os.path.join(T72, "textbook72_distract2.jsonl"))]
    summ = dict(ckpt=a.ckpt, k=a.k, seed=a.seed, n=len(dist), protocol="environment writes the premise block")
    res = {}
    for label, prompts in (("original", [orig[r["name"]] for r in dist]), ("distract2", [r["prompt"] for r in dist])):
        t0 = time.time()
        hits, fail, _ = score(model, tok, [dict(prompt=p) for p in prompts], a.k, a.seed, True)
        res[label] = hits
        summ[label] = dict(avg_solve_rate=round(sum(hits) / (a.k * len(hits)), 4), any_of_k=sum(1 for h in hits if h > 0), secs=round(time.time() - t0))
        print(label, summ[label], flush=True)
    cross = [(ho, hd) for r, ho, hd in zip(dist, res["original"], res["distract2"]) if r["n_prem"] >= 4]
    stay = [(ho, hd) for r, ho, hd in zip(dist, res["original"], res["distract2"]) if r["n_prem"] <= 3]
    summ["crossing_to_4plus"] = dict(n=len(cross), orig_solved=sum(ho > 0 for ho, _ in cross), dist_solved=sum(hd > 0 for _, hd in cross))
    summ["staying_le3"] = dict(n=len(stay), orig_solved=sum(ho > 0 for ho, _ in stay), dist_solved=sum(hd > 0 for _, hd in stay))
    bins = collections.defaultdict(lambda: dict(n=0, orig_solved=0, dist_solved=0))
    for r, ho, hd in zip(dist, res["original"], res["distract2"]):
        b = bins[min(3, r["n_prem_orig"])]; b["n"] += 1; b["orig_solved"] += ho > 0; b["dist_solved"] += hd > 0
    summ["by_orig_premises"] = {str(k): v for k, v in sorted(bins.items())}
    with open(os.path.join(EXP, "metrics_envpr.jsonl"), "a") as f:
        f.write(json.dumps(summ) + "\n")
    with open(os.path.join(T72, f"envpr_hits_{a.ckpt}_k{a.k}.json"), "w") as f:
        json.dump(dict(names=[r["name"] for r in dist], **res), f)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
