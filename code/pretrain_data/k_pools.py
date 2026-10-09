"""Pools from ideas.md § K (2026-10-09): diversify K12 with what the screens found hard but relevant.

Every pool is K12 with a 10 % slice (15,500 rows outside the 1,024-row val head) replaced, as mix.py does, so the screens'
`slice` rows are the new material. Sources:

  K1  k12_jdist10     join problems (data/pretrain_data/joins.jsonl.gz) with 1–2 unused premises, shortcut-checked by
                      minlen as scan_pools.py's distractors are (the minimal proof of the distracted problem is no shorter).
  K2  k12_jsubst10    join problems with atoms replaced by depth-1 formulas at rate 0.7 (scan_pools.substitute).
  K4  k12_prem4_10    generator problems with 4–5 premises (the pinned gen.py's Gen class with max_prem 5, called with
      k12_prem4_1e2   n_prem 4 or 5 in both its forward and goal modes and both its short and long settings; the CLI never
      k12_prem4_1e3   asks for more than 3). Nested 10⁻² and 10⁻³ slices for the surfacing dial.
  K5  k12_long16_10   generator long mode at 13–16 lines, box depth ≤ 3, with at least one of the hard proofs' rule joins.
  K6  k12_goalimp10   population rows whose goal is a conditional with a conditional inside (hypothetical-syllogism shapes).
  K7  k12_nestore10   population proofs with ∨E citing ∨E where the outer case blocks have different rule sequences.
  K8  k12_reductio10  population proofs with a ¬I box of ≥ 5 lines that ends in the goal (reductio, not a one-line clash).
  K10 k12_minfloor8   K12 rows whose minimal proof is ≥ 8 lines (k12_minimal's labels), nothing else: a hard-only pool.
  K12 k12_reuse10     population proofs where a derived line is cited by two or more later lines.
  K9  k12_hard10      (sub-command `hard`, loads Robbie's base) population problems the base solves ≤ 3 times in 8.

"Population" = the size ladder's generator files under data/pretrain_data/gen/ (2.2M rows), streamed once. Every new
record: one per curry-aware renaming class and not a class of K12, re-verified by the pinned verifier, box depth ≤ 2
unless the pool says otherwise, audited against every evaluation set and the sealed 72. A source short of 15,500 is
filled with atom permutations of its own records (as the join and sub-problem slices were), tagged `perm_of`.
K3 (join chains) and K11 (case-flip pairs) need new generation logic and are not built here.

    cd code && nohup uv run python pretrain_data/k_pools.py build > ../data/pretrain_data/k_pools.log 2>&1 &
    cd code && nohup uv run python pretrain_data/k_pools.py hard  > ../data/pretrain_data/k_pools_hard.log 2>&1 &

Writes data/pretrain_data/pools/k/<name>.jsonl.gz, stats.json and the section "## K pools" of docs/pretrain-data/pools.md.
"""
from __future__ import annotations

import argparse
import collections
import glob
import itertools
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "textbook_problem"), os.path.join(HERE, "..", "textbook_problem", "fork_tools")]
import pool_common as PC  # noqa: E402  (pins the verifier first)
import scan_pools as SP  # noqa: E402  (_work_distract, substitute, rand_formula)
from patterns import parse  # noqa: E402
import gen as G  # noqa: E402  (the pinned generator, used through its Gen class; not edited)

OUT_DIR = os.path.join(PC.DATA, "pools", "k")
GEN_DIR = os.path.join(PC.DATA, "gen")
DOC = os.path.join(PC.ROOT, "docs", "pretrain-data", "pools.md")
HEAD = "## K pools (ideas.md § K): diversifying from what was hard"
VAL_N = 1024
SLICE = 15_500
JOIN_PAIRS = set().union(*PC.JOINS.values())


# ------------------------------------------------------------------ helpers
def bigrams_of(lines):
    by = {ln["idx"]: ln for ln in lines}
    out = set()
    for ln in lines:
        for r in ln["refs"]:
            if r in by:
                out.add((ln["rule"], by[r]["rule"]))
    return out


def max_depth(lines):
    return max(ln["depth"] for ln in lines)


def expand(rows, need, rng, tag):
    """fill to `need` with atom permutations of the rows themselves (verified), originals first"""
    rows = list(rows)
    if len(rows) >= need:
        rng.shuffle(rows)
        return rows[:need], 0
    seen = {r["prompt"] for r in rows}
    perms = []
    order = list(rows)
    rng.shuffle(order)
    for r in order:
        for p in PC.permutations_of(r, tag)[1:]:
            if p["prompt"] not in seen:
                seen.add(p["prompt"])
                perms.append(p)
        if len(rows) + len(perms) >= need:
            break
    rng.shuffle(perms)
    out = rows + perms[: need - len(rows)]
    rng.shuffle(out)
    return out, min(len(perms), need - len(rows))


def replace_slice(base, new_rows, rng, tag):
    idx = rng.sample(range(VAL_N, len(base)), len(new_rows))
    out = list(base)
    for i, r in zip(idx, new_rows):
        out[i] = dict(r, slice=tag)
    return out


def finish(name, base, new_rows, rng, idx, stats, extra=None):
    kept, arep = idx.audit(new_rows)
    rows, n_perm = expand(kept, SLICE, rng, name)
    pool = replace_slice(base, rows, rng, name)
    PC.write(pool, os.path.join(OUT_DIR, f"{name}.jsonl.gz"))
    s = dict(label=name, n_source=len(new_rows), n_after_audit=len(kept), audit=arep, n_slice=len(rows), n_perm=n_perm,
             by_len=dict(sorted(collections.Counter(r["n_lines"] for r in rows).items())),
             by_prem=dict(sorted(collections.Counter(r["n_prem"] for r in rows).items())),
             by_depth=dict(sorted(collections.Counter(max_depth(parse(r["proof"])) for r in rows).items())))
    if extra:
        s.update(extra)
    stats.append(s)
    print(name, json.dumps({k: v for k, v in s.items() if k not in ("by_len", "by_prem", "by_depth")}), flush=True)
    return rows


def gen_rows(rng, want, accept, long, nprem_choices, min_len, max_len, seed_tag, max_secs, k12_keys):
    """draw from the pinned generator until `want` accepted, deduplicated, verified records (or max_secs)"""
    g = G.Gen(rng, max_prem=5)
    g.ore_steps = 3 if long else 1
    g.strict = long
    g.bot_p = 0.0 if long else 0.02
    seen = set(k12_keys)
    out, rep, t0 = [], collections.Counter(), time.time()
    while len(out) < want and time.time() - t0 < max_secs:
        rep["tries"] += 1
        try:
            n_prem = rng.choice(nprem_choices)
            if rng.random() < 0.5:
                r = g.forward(rng.randint(6, 22) if long else rng.randint(1, 10), n_prem)
            else:
                r = g.goal(n_prem, rng.choice([2, 3, 3]) if long else rng.choice([1, 2, 2, 3]),
                           rng.choice([3, 4, 5]) if long else rng.choice([1, 2, 3]))
        except (G.Fail, RecursionError) as e:
            rep["fail:" + str(e).split(":")[0]] += 1
            continue
        if not (min_len <= r["n_lines"] <= max_len):
            rep["len_out"] += 1
            continue
        ks = PC.BR.keys(r["prompt"])
        if ks & seen:
            rep["dup"] += 1
            continue
        rec, why = PC.make_record(r["prompt"], r["proof"], seed_tag, contra_prem=r.get("contra_prem", False))
        if rec is None:
            rep["verify_fail"] += 1
            continue
        if not accept(rec):
            rep["rejected"] += 1
            continue
        seen |= ks
        out.append(rec)
        if len(out) % 2000 == 0:
            print(f"  {seed_tag}: {len(out)} in {time.time() - t0:.0f}s {dict(rep)}", flush=True)
    rep["secs"] = round(time.time() - t0)
    return out, dict(rep)


# ------------------------------------------------------------------ population filters (K6, K7, K8, K12)
def is_imp(f):
    return f[0] == "imp"


def goal_imp_nested(rec, lines):
    _, goal = PC.L.parse_prompt(rec["prompt"])
    return is_imp(goal) and (is_imp(goal[1]) or is_imp(goal[2]))


def nested_ore_different_cases(rec, lines):
    by = {ln["idx"]: ln for ln in lines}
    pos = {ln["idx"]: i for i, ln in enumerate(lines)}
    for ln in lines:
        if ln["rule"] != "ORE" or len(ln["refs"]) < 5:
            continue
        inner = [by[r] for r in ln["refs"] if r in by and by[r]["rule"] == "ORE"]
        if not inner:
            continue
        a1, c1, a2, c2 = (pos.get(r) for r in ln["refs"][1:5])
        if None in (a1, c1, a2, c2):
            continue
        case1 = tuple(x["rule"] for x in lines[a1: c1 + 1])
        case2 = tuple(x["rule"] for x in lines[a2: c2 + 1])
        if case1 != case2:
            return True
    return False


def long_reductio(rec, lines):
    pos = {ln["idx"]: i for i, ln in enumerate(lines)}
    last = lines[-1]
    for ln in lines:
        if ln["rule"] in ("NEGI",) and len(ln["refs"]) >= 2:
            a, c = pos.get(ln["refs"][0]), pos.get(ln["refs"][1])
            if a is not None and c is not None and c - a + 1 >= 5 and (ln is last or last["rule"] == "DN" and ln["idx"] in last["refs"]):
                return True
    return False


def reuse(rec, lines):
    cited = collections.Counter()
    by = {ln["idx"]: ln for ln in lines}
    for ln in lines:
        for r in set(ln["refs"]):
            if r in by and by[r]["rule"] not in ("PR", "AS"):
                cited[r] += 1
    return any(v >= 2 for v in cited.values())


POP_FILTERS = {
    "k12_goalimp10": (goal_imp_nested, 2),
    "k12_nestore10": (nested_ore_different_cases, 2),
    "k12_reductio10": (long_reductio, 2),
    "k12_reuse10": (reuse, 2),
}


def population_pass(k12_keys, need_each, limit=0):
    """one stream over the generator files; per pool up to 3×need candidates, deduplicated by class, verified"""
    cand = {k: [] for k in POP_FILTERS}
    seen = {k: set(k12_keys) for k in POP_FILTERS}
    rep = collections.Counter()
    n = 0
    for path in sorted(glob.glob(os.path.join(GEN_DIR, "*.jsonl"))):
        with open(path) as f:
            for line in f:
                n += 1
                if limit and n > limit:
                    break
                if all(len(cand[k]) >= 3 * need_each for k in cand):
                    break
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "prompt" not in r:
                    continue
                lines = parse(r["proof"])
                if lines is None:
                    continue
                d = max_depth(lines)
                for k, (fn, dmax) in POP_FILTERS.items():
                    if len(cand[k]) >= 3 * need_each or d > dmax:
                        continue
                    try:
                        hit = fn(r, lines)
                    except Exception:  # noqa: BLE001
                        rep[f"{k}:filter_error"] += 1
                        continue
                    if not hit:
                        continue
                    ks = PC.BR.keys(r["prompt"])
                    if ks & seen[k]:
                        rep[f"{k}:dup"] += 1
                        continue
                    rec, why = PC.make_record(r["prompt"], r["proof"], k, contra_prem=r.get("contra_prem", False))
                    if rec is None:
                        rep[f"{k}:verify_fail"] += 1
                        continue
                    seen[k] |= ks
                    cand[k].append(rec)
        print(os.path.basename(path), "read", n, {k: len(v) for k, v in cand.items()}, flush=True)
    return cand, dict(rep)


# ------------------------------------------------------------------ doc
def write_doc(stats):
    W = [f"\n{HEAD}\n",
         "Built by `k_pools.py` (2026-10-09) from ideas.md § K. Every pool is K12 with a 10 % slice (15,500 rows outside the val head) "
         "replaced by the source named in the script's docstring; `n source` = records the source produced, `after audit` = after removing "
         "twins of the sealed 72 and of every evaluation problem, `perm` = atom permutations added to reach 15,500 when the source was "
         "short. New records are one per curry-aware class, not a class of K12, re-verified by the pinned verifier. `k12_minfloor8` is "
         "not a slice: it is the val head plus the K12 rows whose minimal proof is 8 lines or more, nothing else.\n",
         "| pool | n source | after audit | removed eval twins | removed sealed twins | slice | perm | lines | premises | depth | notes |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in stats:
        W.append(f"| {s['label']} | {s['n_source']:,} | {s['n_after_audit']:,} | {s['audit'].get('removed_eval_twin', 0)} | "
                 f"{s['audit'].get('removed_sealed_twin', 0)} | {s['n_slice']:,} | {s['n_perm']:,} | {s['by_len']} | {s['by_prem']} | "
                 f"{s['by_depth']} | {s.get('notes', '')} |")
    W.append("")
    text = open(DOC).read() if os.path.exists(DOC) else ""
    if HEAD in text:
        i = text.index(HEAD)
        j = text.find("\n## ", i + len(HEAD))
        text = text[:i].rstrip("\n") + "\n" + (text[j:] if j > 0 else "")
    with open(DOC, "w") as fh:
        fh.write(text.rstrip("\n") + "\n" + "\n".join(W) + "\n")


def load_stats():
    p = os.path.join(OUT_DIR, "stats.json")
    return json.load(open(p)) if os.path.exists(p) else []


def save_stats(stats):
    old = {s["label"]: s for s in load_stats()}
    for s in stats:
        old[s["label"]] = s
    out = list(old.values())
    json.dump(out, open(os.path.join(OUT_DIR, "stats.json"), "w"), indent=1, ensure_ascii=False)
    write_doc(out)


# ------------------------------------------------------------------ build
def cmd_build(a):
    rng = random.Random(a.seed)
    os.makedirs(OUT_DIR, exist_ok=True)
    k12 = PC.load_k12()
    k12_keys = set()
    for r in k12:
        k12_keys |= PC.BR.keys(r["prompt"])
    idx = PC.EvalIndex()
    stats = []
    only = set(a.only or [])

    def want(name):
        return not only or name in only

    # K1, K2: from the join problems
    if want("k12_jdist10") or want("k12_jsubst10"):
        joins = PC.read(os.path.join(PC.DATA, "joins.jsonl.gz"))
        print("joins", len(joins), flush=True)
    if want("k12_jdist10"):
        t0 = time.time()
        out, rep = [], collections.Counter()
        with ProcessPoolExecutor(a.workers) as ex:
            it = ex.map(SP._work_distract, ((r, r["proof"], a.secs, a.seed * 1_000_003 + i) for i, r in enumerate(joins)), chunksize=16)
            for r, res in zip(joins, it):
                rep[res["status"].split(":")[0]] += 1
                if "rec_orig" in res:
                    out.append(dict(res["rec_orig"], pool="k12_jdist10", family=r.get("family"), joins=r.get("joins")))
        print("jdist", dict(rep), f"{time.time() - t0:.0f}s", flush=True)
        finish("k12_jdist10", k12, out, rng, idx, stats,
               dict(notes=f"distractor status {dict(rep)}; by k {dict(collections.Counter(r['n_distract'] for r in out))}"))
    if want("k12_jsubst10"):
        out, rep = [], collections.Counter()
        for r in joins:
            rec, why = SP.substitute(r, rng)
            rep[why.split(":")[0]] += 1
            if rec:
                out.append(dict(rec, pool="k12_jsubst10", family=r.get("family"), joins=r.get("joins")))
        finish("k12_jsubst10", k12, out, rng, idx, stats, dict(notes=f"substitute {dict(rep)}"))

    # K4: 4–5 premises, both settings
    if want("k12_prem4_10"):
        short, rs = gen_rows(rng, SLICE * 2 // 3, lambda rec: rec["n_prem"] >= 4, False, [4, 5], 2, 6, "prem4_short", a.gen_secs, k12_keys)
        longr, rl = gen_rows(rng, SLICE - len(short), lambda rec: rec["n_prem"] >= 4 and max_depth(parse(rec["proof"])) <= 2, True, [4, 5], 7, 12,
                             "prem4_long", a.gen_secs, k12_keys)
        allr = [dict(r, pool="k12_prem4_10") for r in short + longr]
        rows = finish("k12_prem4_10", k12, allr, rng, idx, stats, dict(notes=f"short {rs}; long {rl}"))
        for frac, name in ((100, "k12_prem4_1e2"), (1000, "k12_prem4_1e3")):
            sub = rows[: SLICE // frac]
            pool = replace_slice(k12, [dict(r, pool=name) for r in sub], rng, name)
            PC.write(pool, os.path.join(OUT_DIR, f"{name}.jsonl.gz"))
            stats.append(dict(label=name, n_source=len(sub), n_after_audit=len(sub), audit={}, n_slice=len(sub), n_perm=0,
                              by_len=dict(sorted(collections.Counter(r["n_lines"] for r in sub).items())),
                              by_prem=dict(sorted(collections.Counter(r["n_prem"] for r in sub).items())),
                              by_depth=dict(sorted(collections.Counter(max_depth(parse(r["proof"])) for r in sub).items())),
                              notes="nested subset of k12_prem4_10"))
            print(name, len(sub), flush=True)

    # K5: 13–16 lines, depth ≤ 3, a hard join present
    if want("k12_long16_10"):
        def ok16(rec):
            lines = parse(rec["proof"])
            return max_depth(lines) <= 3 and bool(bigrams_of(lines) & JOIN_PAIRS)
        longr, rl = gen_rows(rng, SLICE, ok16, True, [0, 1, 1, 2, 2, 3], 13, 16, "long16", a.gen_secs * 2, k12_keys)
        finish("k12_long16_10", k12, [dict(r, pool="k12_long16_10") for r in longr], rng, idx, stats, dict(notes=f"gen {rl}"))

    # K6, K7, K8, K12: one pass over the population
    pop_names = [n for n in POP_FILTERS if want(n)]
    if pop_names:
        cand, rep = population_pass(k12_keys, SLICE, a.limit)
        print("population pass", rep, flush=True)
        for name in pop_names:
            rows = cand[name]
            rng.shuffle(rows)
            finish(name, k12, rows, rng, idx, stats, dict(notes=f"candidates {len(rows)}"))

    # K10: hard-only K12 by minimal length
    if want("k12_minfloor8"):
        mn = PC.read(os.path.join(PC.DATA, "pools", "k12_minimal.jsonl.gz"))
        rows = [dict(r, pool="k12_minfloor8", slice="k12_minfloor8") for r in mn[VAL_N:]
                if r.get("minlen_status") in ("shorter", "same") and r["n_lines"] >= 8]
        pool = k12[:VAL_N] + rows
        PC.write(pool, os.path.join(OUT_DIR, "k12_minfloor8.jsonl.gz"))
        stats.append(dict(label="k12_minfloor8", n_source=len(rows), n_after_audit=len(rows), audit={}, n_slice=len(rows), n_perm=0,
                          by_len=dict(sorted(collections.Counter(r["n_lines"] for r in rows).items())),
                          by_prem=dict(sorted(collections.Counter(r["n_prem"] for r in rows).items())),
                          by_depth=dict(sorted(collections.Counter(max_depth(parse(r["proof"])) for r in rows).items())),
                          notes=f"whole pool = val head + {len(rows)} rows (K12's own, minimal form); status {dict(collections.Counter(r['minlen_status'] for r in rows))}"))
        print("k12_minfloor8", len(rows), flush=True)
    save_stats(stats)
    print("wrote", OUT_DIR, "and the K-pools section of pools.md")


# ------------------------------------------------------------------ hard (K9)
def cmd_hard(a):
    import screen_ckpt as SC
    import eval_ckpt as EC
    rng = random.Random(a.seed)
    os.makedirs(OUT_DIR, exist_ok=True)
    k12 = PC.load_k12()
    k12_keys = set()
    for r in k12:
        k12_keys |= PC.BR.keys(r["prompt"])
    # candidates: a flat-by-length sample of the population, depth ≤ 2, not K12 classes
    per_len = a.n_cand // 11
    cand = collections.defaultdict(list)
    seen = set(k12_keys)
    paths = sorted(glob.glob(os.path.join(GEN_DIR, "*.jsonl")))
    for path in paths:
        with open(path) as f:
            for line in f:
                if all(len(cand[L]) >= per_len for L in range(2, 13)):
                    break
                if rng.random() > a.sample_rate:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "prompt" not in r:
                    continue
                L = r.get("n_lines") or r["proof"].count(";") + 1
                if not (2 <= L <= 12) or len(cand[L]) >= per_len:
                    continue
                lines = parse(r["proof"])
                if lines is None or max_depth(lines) > 2:
                    continue
                ks = PC.BR.keys(r["prompt"])
                if ks & seen:
                    continue
                rec, why = PC.make_record(r["prompt"], r["proof"], "k12_hard10", contra_prem=r.get("contra_prem", False))
                if rec is None:
                    continue
                seen |= ks
                cand[L].append(rec)
    rows = [r for L in range(2, 13) for r in cand[L]]
    print("candidates", len(rows), {L: len(cand[L]) for L in range(2, 13)}, flush=True)
    model, tok, meta, fork_sample, lean_tok = EC.load_robbie(SC.RUN, "pretrain", "cpu")
    print("loaded", SC.RUN, meta["n_params"], "params", flush=True)
    hard, rep, t0 = [], collections.Counter(), time.time()
    B = 512
    for s in range(0, len(rows), B):
        chunk = rows[s: s + B]
        hits = SC.pass_at_k_hits(model, tok, fork_sample, chunk, a.k, a.seed + s) if hasattr(SC, "pass_at_k_hits") else None
        if hits is None:
            hits = pass_hits(model, tok, fork_sample, chunk, a.k, a.seed + s)
        for r, h in zip(chunk, hits):
            rep[f"hits={h}"] += 1
            if h is not None and h <= a.max_hits:
                hard.append(dict(r, base_hits=h, base_k=a.k))
        print(f"  {s + len(chunk)} / {len(rows)} screened, {len(hard)} hard, {time.time() - t0:.0f}s", flush=True)
    print("hits distribution", dict(sorted(rep.items())), flush=True)
    idx = PC.EvalIndex()
    stats = []
    finish("k12_hard10", k12, hard, rng, idx, stats,
           dict(notes=f"{len(rows)} candidates × {a.k} samples on {SC.RUN}; kept hits ≤ {a.max_hits}; hits {dict(sorted(rep.items()))}"))
    save_stats(stats)
    print("wrote k12_hard10")


def pass_hits(model, tok, fork_sample, rows, k, seed, max_new=400, bs=256):
    """per-row number of verified samples out of k (None if the prompt cannot be encoded); mirrors screen_ckpt.pass_at_k"""
    import torch
    from strict_verify import safe_verify
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    prompts = []
    for r in rows:
        try:
            prompts.append(tok.encode_prompt(r["prompt"]))
        except Exception:  # noqa: BLE001
            prompts.append(None)
    flat = [(i, j) for i, p in enumerate(prompts) if p is not None for j in range(k)]
    order = sorted(range(len(flat)), key=lambda x: len(prompts[flat[x][0]]))
    outs = [None] * len(flat)
    for s in range(0, len(order), bs):
        chunk = order[s: s + bs]
        res = fork_sample.generate_ids(model, tok, [prompts[flat[x][0]] for x in chunk], greedy=False, temperature=1.0,
                                       max_new=max_new, gen=gen)
        for x, o in zip(chunk, res):
            outs[x] = o
    hits = collections.Counter()
    for (i, j), o in zip(flat, outs):
        body = tok.decode(o)
        if not body.startswith("LEANPARSE") and safe_verify(rows[i]["prompt"], body)[0]:
            hits[i] += 1
    return [None if prompts[i] is None else hits[i] for i in range(len(rows))]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--only", nargs="*", help="pool names to build (default: all except k12_hard10)")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--workers", type=int, default=6)
    b.add_argument("--secs", type=float, default=2.0, help="minlen time limit per distractor check")
    b.add_argument("--gen_secs", type=float, default=900, help="time budget per generator draw")
    b.add_argument("--limit", type=int, default=0, help="population rows to read (smoke test)")
    h = sub.add_parser("hard")
    h.add_argument("--seed", type=int, default=0)
    h.add_argument("--n_cand", type=int, default=33_000)
    h.add_argument("--sample_rate", type=float, default=0.05)
    h.add_argument("--k", type=int, default=8)
    h.add_argument("--max_hits", type=int, default=3)
    d = sub.add_parser("doc")
    a = ap.parse_args()
    if a.cmd == "build":
        cmd_build(a)
    elif a.cmd == "hard":
        cmd_hard(a)
    else:
        write_doc(load_stats())
        print("rewrote the K-pools section")


if __name__ == "__main__":
    main()
