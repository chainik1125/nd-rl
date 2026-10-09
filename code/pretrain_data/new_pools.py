"""Seven more Stage-1 pools from ideas.md entries that had no builder yet (2026-10-07), all at K12's size (155,000).

Sub-commands (sources first, `assemble` last; `compose` and `multi` call `minlen` per row and run detached):

  closure   every interior line of a minimal proof as its own theorem (ideas A2 / H1: AlphaGeometry traceback, HER).
            Sources: `k12_minimal` and the depth-3 side pool; premises pruned to the ones the sub-proof cites, enclosing
            assumptions promoted to premises; one record per renaming class that is not already in K12.
  compose   two minimal K12 proofs spliced where the first proves a premise of the second (ideas A4 / H15: lemma then use).
            Kept only when `minlen` finds nothing shorter (bound = length − 1, 2 s): the composition is the minimal proof.
  recover   a minimal proof with ONE dead-but-valid line inserted after the premises (ideas B3 / H3: mistake-and-recovery):
            ∧E / ∨I / ∧I / DN / R from premises, never cited; the rest of the proof is the clean completion.
  multi     a SECOND, different proof of a K12 theorem (ideas B2 / H11: several proofs per theorem): `minlen` under a
            restriction the original does not satisfy (no DN, no derived ∨E, no lemmas, depth cap), at most 2 lines longer.
  dual      Γ ⊢ A rewritten as Γ, ¬A ⊢ ⊥ and as Γ ⊢ ¬¬A, each with the original proof wrapped (ideas A5).
  shape     importance resampling of the generator population to the SEED split's shape (ideas C3 / H10: DSIR). Cell =
            (goal connective, premises, length bin, box depth); target = smoothed cell histogram of the 224 seed-split
            reference proofs (textbook *shape*, never a textbook problem); population = `k12_500k` ∪ depth-3 side pool.
  balance   rule-balanced resampling of the same population (ideas C5 / H13, SimpleLogic shortcuts): each record is
            weighted by 1 / share of its rarest rule, so every rule's share moves toward uniform.
  assemble  builds the pools below and appends "## New pools" to docs/pretrain-data/pools.md.

Pools (records 0–1,023 are K12's first 1,024 rows in every pool, the pinned trainer's val head):

  k12_closure10   K12 with a 10 % slice of closure sub-theorems (flat over 3–12 lines).
  k12_compose10   K12 with a 10 % slice of compositions.
  k12_recover20   K12 with 20 % of its rows replaced by the recovery version of their minimal proof.
  k12_multi10     K12 with a 10 % slice of second proofs (the originals stay in the pool).
  k12_dual1       K12 with a 1 % slice of duals (half ⊥-goal, half ¬¬-goal).
  k12_shape       the shape-resampled generator pool (whole pool, not a slice).
  k12_balance     the rule-balanced generator pool (whole pool).

    cd code && uv run python pretrain_data/new_pools.py closure
    cd code && nohup uv run python pretrain_data/new_pools.py compose --n 20000 --workers 4 > ../data/pretrain_data/new/compose.log 2>&1 &
    cd code && uv run python pretrain_data/new_pools.py recover --n 33000
    cd code && nohup uv run python pretrain_data/new_pools.py multi --n 20000 --workers 4 > ../data/pretrain_data/new/multi.log 2>&1 &
    cd code && uv run python pretrain_data/new_pools.py dual --n 1000
    cd code && uv run python pretrain_data/new_pools.py shape
    cd code && uv run python pretrain_data/new_pools.py balance
    cd code && uv run python pretrain_data/new_pools.py assemble

Every record is checked by the pinned verifier (`strict_verify.safe_verify`) and every new theorem is audited by renaming
class against the 412 evaluation problems and the sealed 72 (counted only) before it enters a pool. Nothing here trains.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pool_common as PC  # noqa: E402
import logic as LG  # noqa: E402
from minlen import minlen  # noqa: E402
from patterns import fstr, lines_to_text, parse, prune_lines  # noqa: E402
from release_1002 import BR  # noqa: E402
from subproblems import subproblem  # noqa: E402
from screen_data import validation_set  # noqa: E402
import scan_pools as SP  # noqa: E402

NEW = os.path.join(PC.DATA, "new")
OUT_DIR = os.path.join(PC.DATA, "pools", "new")
DOC = os.path.join(PC.ROOT, "docs", "pretrain-data", "pools.md")
VAL_N = SP.VAL_N
N_POOL = SP.N_POOL
DEPTH3 = os.path.join(PC.DATA, "pools", "size", "gen_depth3.jsonl.gz")
K500 = os.path.join(PC.DATA, "pools", "size", "k12_500k.jsonl.gz")
RULES = ["ANDI", "ANDE1", "ANDE2", "IMPE", "IMPI", "ORI1", "ORI2", "ORE", "NEGE", "NEGI", "BOTE", "DN"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def goal_text(prompt):
    toks = prompt.split()
    return " ".join(toks[toks.index("SEQ") + 1: -1])


def prem_texts(prompt):
    prem, _ = LG.parse_prompt(prompt)
    return [fstr(p) for p in prem]


def iter_rows(path):
    import gzip
    with gzip.open(path, "rt") as f:
        for line in f:
            yield json.loads(line)


def all_prem_cited(lines):
    """every PR line is in the dependency closure of the last line (prune_lines keeps PR lines unconditionally)"""
    byidx = {ln["idx"]: ln for ln in lines}
    keep, stack = set(), [lines[-1]["idx"]]
    while stack:
        i = stack.pop()
        if i in keep or i not in byidx:
            continue
        keep.add(i)
        stack.extend(byidx[i]["refs"])
    return all(ln["idx"] in keep for ln in lines if ln["rule"] == "PR")


def minimal_rows(require_all_cited=True):
    """k12_minimal rows (with their K12 index) that are minimal and, by default, cite every premise"""
    mn = SP.load_minimal()
    out = []
    for i, r in enumerate(mn):
        if i < VAL_N or r.get("minlen_status") not in ("shorter", "same"):
            continue
        lines = parse(r["proof"])
        if not lines or (require_all_cited and not all_prem_cited(lines)):
            continue
        out.append((i, r, lines))
    return out


def k12_keys():
    ks = set()
    for r in PC.load_k12():
        ks |= BR.keys(r["prompt"])
    return ks


def finish(rows, name, allowed=("v0", "lean_v2")):
    """dedup by class, audit, write NEW/name.jsonl.gz with a stats json; -> kept rows"""
    rows = PC.dedup(rows)
    idx = PC.EvalIndex()
    kept, rep = idx.audit(rows, allowed_sets=allowed)
    os.makedirs(NEW, exist_ok=True)
    PC.write(kept, os.path.join(NEW, f"{name}.jsonl.gz"))
    st = PC.stats(kept, name, PC.hard_pieces())
    st["audit"] = rep
    json.dump(st, open(os.path.join(NEW, f"{name}.stats.json"), "w"), indent=1, ensure_ascii=False)
    log(name, "kept", len(kept), "audit", rep)
    return kept


# ---------------------------------------------------------------- closure

def cmd_closure(a):
    rng = random.Random(a.seed)
    known = k12_keys()
    srcs = [("k12_minimal", [r for _, r, _ in minimal_rows(require_all_cited=False)])]
    deep = PC.read(DEPTH3)
    rng.shuffle(deep)
    srcs.append(("gen_depth3", deep[: a.deep]))
    out, seen, rep = [], set(), collections.Counter()
    for src, rows in srcs:
        for r in rows:
            lines = parse(r["proof"])
            if not lines:
                continue
            for i, ln in enumerate(lines[:-1]):
                if ln["rule"] in ("PR", "AS"):
                    continue
                sp = subproblem(lines, i, prune_prem=True)
                if sp is None:
                    rep["trivial"] += 1
                    continue
                prompt, proof = sp
                ks = BR.keys(prompt)
                if ks & known or ks & seen:
                    rep["known_or_dup"] += 1
                    continue
                rec, why = PC.make_record(prompt, proof, "closure", source=src, parent=r["prompt"])
                if rec is None:
                    rep[f"reject:{why}"] += 1
                    continue
                if rec["n_lines"] < a.min_lines:
                    rep["short"] += 1
                    continue
                seen |= ks
                out.append(rec)
            if len(out) >= a.n:
                break
        log(src, "closure records", len(out), dict(rep))
    # the depth-3 side pool is raw generator output, so its sub-proofs can carry lazy-premise case splits: re-prove
    # every sub-theorem from that source minimally (bound = one line fewer, as minimal_k12.py does) and keep the shorter proof
    todo = [k for k, r in enumerate(out) if r["source"] == "gen_depth3" and r["n_lines"] >= 4]
    n_short = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for k, u in zip(todo, ex.map(_work_min, [(out[k]["prompt"], out[k]["n_lines"], 2.0) for k in todo], chunksize=16)):
            if u:
                rec, _ = PC.make_record(out[k]["prompt"], u, "closure", source=out[k]["source"], parent=out[k]["parent"], padded_lines=out[k]["n_lines"])
                if rec:
                    out[k] = rec
                    n_short += 1
    log("gen_depth3 sub-proofs re-proved minimal", n_short, "of", len(todo))
    finish(out, "closure")


def _work_min(args):
    prompt, n, secs = args
    u = minlen(prompt, bound=n - 1, time_limit=secs)
    return u["proof"] if u.get("min_lines_ub") else None


# ---------------------------------------------------------------- compose

def splice(A, Al, B, Bl, c_text):
    """A proves c; B has premise c. -> (prompt, proof_text) of the composition, or None."""
    a_prem = prem_texts(A["prompt"])
    b_prem = prem_texts(B["prompt"])
    b_goal = goal_text(B["prompt"])
    if b_goal in a_prem or b_goal == c_text or goal_text(A["prompt"]) != c_text:
        return None
    prem = list(a_prem)
    for p in b_prem:
        if p != c_text and p not in prem:
            prem.append(p)
    if len(prem) > 6 or b_goal in prem:
        return None
    new, idx = [], 0
    pr_idx = {}
    for p in prem:
        idx += 1
        pr_idx[p] = idx
        new.append(dict(idx=idx, depth=0, formula=None, rule="PR", refs=[], text=p))
    # A's derived lines
    amap = {}
    for ln in Al:
        if ln["rule"] == "PR":
            amap[ln["idx"]] = pr_idx[fstr(ln["formula"])]
            continue
        idx += 1
        amap[ln["idx"]] = idx
        new.append(dict(idx=idx, depth=ln["depth"], formula=ln["formula"], rule=ln["rule"], refs=[amap[x] for x in ln["refs"]]))
    c_idx = idx  # A's last line proves c
    bmap = {}
    c_src = next(ln["idx"] for ln in Bl if ln["rule"] == "PR" and fstr(ln["formula"]) == c_text)
    users = {ln["rule"] for ln in Bl if c_src in ln["refs"]}
    if not users or users <= {"ANDI", "ORI1", "ORI2"}:
        return None  # the lemma must be *used* (→E, ∨E, ∧E, ¬E, DN), not merely re-wrapped
    for ln in Bl:
        if ln["rule"] == "PR":
            t = fstr(ln["formula"])
            bmap[ln["idx"]] = c_idx if t == c_text else pr_idx[t]
            continue
        idx += 1
        bmap[ln["idx"]] = idx
        new.append(dict(idx=idx, depth=ln["depth"], formula=ln["formula"], rule=ln["rule"], refs=[bmap[x] for x in ln["refs"]]))
    for ln in new:
        if ln["formula"] is None:
            ln["formula"], _ = LG.parse_formula(ln["text"].split(), 0)
    prompt = "THM " + " , ".join(prem) + " SEQ " + b_goal + " PRF"
    return prompt, lines_to_text(new)


def _work_compose(args):
    rec, secs = args
    n = rec["n_lines"]
    u = minlen(rec["prompt"], bound=n - 1, time_limit=secs)
    rec = dict(rec, minimal=u["proof"] is None and not u.get("timeout"), minlen_timeout=bool(u.get("timeout")))
    return rec


def cmd_compose(a):
    rng = random.Random(a.seed)
    rows = minimal_rows()
    by_prem = collections.defaultdict(list)
    for i, r, lines in rows:
        for p in prem_texts(r["prompt"]):
            if p[0] == "(":  # compound premises only: atoms as lemmas are trivial
                by_prem[p].append(i)
    byi = {i: (r, lines) for i, r, lines in rows}
    atoms = ["P", "Q", "R", "S"]
    import itertools
    perms = [dict(zip(atoms, p)) for p in itertools.permutations(atoms)]
    known = k12_keys()
    order = list(byi)
    rng.shuffle(order)
    cands, rep = [], collections.Counter()
    for i in order:
        A, Al = byi[i]
        if A["n_lines"] > a.max_part:
            continue
        rng.shuffle(perms)
        for m in perms[:6]:
            Ap = dict(A, prompt=PC.rename_text(A["prompt"], m), proof=PC.rename_text(A["proof"], m))
            c = goal_text(Ap["prompt"])
            hits = by_prem.get(c)
            if not hits:
                continue
            j = rng.choice(hits)
            B, Bl = byi[j]
            if j == i or B["n_lines"] > a.max_part:
                continue
            sp = splice(Ap, parse(Ap["proof"]), B, Bl, c)
            if sp is None:
                rep["no_splice"] += 1
                continue
            prompt, proof = sp
            if BR.keys(prompt) & known:
                rep["known"] += 1
                continue
            rec, why = PC.make_record(prompt, proof, "compose", lemma=c, parent_a=A["prompt"], parent_b=B["prompt"])
            if rec is None:
                rep[f"reject:{why}"] += 1
                continue
            cands.append(rec)
            break
        if len(cands) >= a.n:
            break
    log("compositions verified", len(cands), dict(rep))
    cands = PC.dedup(cands)
    out = []
    with ProcessPoolExecutor(a.workers) as ex:
        for k, rec in enumerate(ex.map(_work_compose, [(r, a.secs) for r in cands], chunksize=8)):
            out.append(rec)
            if k % 1000 == 0:
                log("minlen", k, "minimal so far", sum(r["minimal"] for r in out))
    n_min = sum(r["minimal"] for r in out)
    log("minimal", n_min, "shortenable", sum(not r["minimal"] and not r["minlen_timeout"] for r in out), "timeout", sum(r["minlen_timeout"] for r in out))
    finish([r for r in out if r["minimal"]], "compose")


# ---------------------------------------------------------------- recover

def dead_line_options(lines, rng):
    """candidate (formula, rule, refs) dead lines from the PR lines; the formula is new to the proof"""
    prs = [ln for ln in lines if ln["rule"] == "PR"]
    have = {fstr(ln["formula"]) for ln in lines}
    opts = []
    for ln in prs:
        f = ln["formula"]
        if f[0] == "and":
            opts.append((f[1], "ANDE1", [ln["idx"]]))
            opts.append((f[2], "ANDE2", [ln["idx"]]))
        if f[0] == "not" and f[1][0] == "not":
            opts.append((f[1][1], "DN", [ln["idx"]]))
        if f[0] == "bot":
            continue
        other = rng.choice([("atom", x) for x in "PQRS"] + [p["formula"] for p in prs])
        opts.append((("or", f, other), "ORI1", [ln["idx"]]))
        opts.append((("or", other, f), "ORI2", [ln["idx"]]))
        opts.append((f, "R", [ln["idx"]]))
    if len(prs) >= 2:
        p1, p2 = rng.sample(prs, 2)
        opts.append((("and", p1["formula"], p2["formula"]), "ANDI", [p1["idx"], p2["idx"]]))
    rng.shuffle(opts)
    opts = [o for o in opts if fstr(o[0]) not in have]
    # ∨I is always available, so it would dominate; prefer the other dead moves when the premises allow them
    non_ori = [o for o in opts if o[1] not in ("ORI1", "ORI2")]
    if non_ori and rng.random() < 0.7:
        return non_ori + [o for o in opts if o[1] in ("ORI1", "ORI2")]
    return opts


def cmd_recover(a):
    rng = random.Random(a.seed)
    rows = minimal_rows()
    rng.shuffle(rows)
    out, rep = [], collections.Counter()
    for i, r, lines in rows:
        if r["n_prem"] == 0:
            continue
        n_pr = sum(1 for ln in lines if ln["rule"] == "PR")
        opts = dead_line_options(lines, rng)
        if not opts:
            rep["no_option"] += 1
            continue
        for f, rule, refs in opts[:3]:
            new = []
            for ln in lines[:n_pr]:
                new.append(dict(ln))
            new.append(dict(idx=10_000, depth=0, formula=f, rule=rule, refs=refs))
            for ln in lines[n_pr:]:
                new.append(dict(ln))
            rec, why = PC.make_record(r["prompt"], lines_to_text(new), "recover", k12_index=i, dead_rule=rule, minimal_lines=r["n_lines"])
            if rec:
                out.append(rec)
                rep[rule] += 1
                break
            rep[f"reject:{why}"] += 1
        if len(out) >= a.n:
            break
    log("recover records", len(out), dict(rep))
    # same theorems as K12: no audit needed (K12 is already audited); still verified
    os.makedirs(NEW, exist_ok=True)
    PC.write(out, os.path.join(NEW, "recover.jsonl.gz"))
    st = PC.stats(out, "recover", PC.hard_pieces())
    st["by_dead_rule"] = {k: v for k, v in rep.items() if not k.startswith("reject") and k != "no_option"}
    json.dump(st, open(os.path.join(NEW, "recover.stats.json"), "w"), indent=1, ensure_ascii=False)


# ---------------------------------------------------------------- multi

def _work_multi(args):
    i, rec, secs = args
    lines = parse(rec["proof"])
    rules = {ln["rule"] for ln in lines}
    depth = max(ln["depth"] for ln in lines)
    n = rec["n_lines"]
    variants = []
    if "DN" in rules:
        variants.append(dict(forbid=("DN",)))
    if "ORE" in rules:
        variants.append(dict(no_derived_ore=True))
        variants.append(dict(forbid=("ORE_IN_IMPI", "ORE_IN_ORE")))
    if depth >= 1:
        variants.append(dict(max_depth=depth - 1))
    variants.append(dict(lemmas=False))
    variants.append(dict(forbid=("DN", "ORE_DERIVED")))
    orig_rules = tuple(sorted(ln["rule"] for ln in prune_lines(lines) if ln["rule"] != "PR"))
    orig_seq = tuple((ln["rule"], ln["depth"]) for ln in prune_lines(lines) if ln["rule"] != "PR")
    for v in variants:
        try:
            u = minlen(rec["prompt"], bound=n + 2, time_limit=secs, **v)
        except AssertionError:
            continue
        p = u.get("proof")
        if not p or u.get("min_lines_ub") is None:
            continue
        l2 = parse(p)
        if l2 is None:
            continue
        pl2 = [ln for ln in prune_lines(l2) if ln["rule"] != "PR"]
        r2 = tuple(sorted(ln["rule"] for ln in pl2))
        s2 = tuple((ln["rule"], ln["depth"]) for ln in pl2)
        if s2 == orig_seq:
            continue  # the same (rule, depth) sequence: the same proof up to citation order
        return dict(prompt=rec["prompt"], proof=p, k12_index=i, variant=json.dumps(v), orig_rules=list(orig_rules), orig_lines=n,
                    new_rules=r2 != orig_rules)
    return None


def cmd_multi(a):
    rng = random.Random(a.seed)
    rows = [(i, r) for i, r, lines in minimal_rows() if r["n_lines"] >= 4]
    rng.shuffle(rows)
    rows = rows[: a.n]
    out, tried = [], 0
    with ProcessPoolExecutor(a.workers) as ex:
        for k, res in enumerate(ex.map(_work_multi, [(i, r, a.secs) for i, r in rows], chunksize=8)):
            tried += 1
            if res:
                rec, why = PC.make_record(res["prompt"], res["proof"], "multi", **{x: res[x] for x in ("k12_index", "variant", "orig_rules", "orig_lines", "new_rules")})
                if rec:
                    out.append(rec)
            if k % 1000 == 0:
                log("multi", k, "found", len(out))
    log("multi tried", tried, "second proofs", len(out), "by variant", dict(collections.Counter(r["variant"] for r in out)))
    os.makedirs(NEW, exist_ok=True)
    PC.write(out, os.path.join(NEW, "multi.jsonl.gz"))
    st = PC.stats(out, "multi", PC.hard_pieces())
    st["tried"] = tried
    st["by_variant"] = dict(collections.Counter(r["variant"] for r in out))
    st["longer_by"] = dict(collections.Counter(r["n_lines"] - r["orig_lines"] for r in out))
    st["new_rule_multiset"] = sum(r["new_rules"] for r in out)
    json.dump(st, open(os.path.join(NEW, "multi.stats.json"), "w"), indent=1, ensure_ascii=False)


# ---------------------------------------------------------------- dual

def dualise(r, lines, kind):
    prem = prem_texts(r["prompt"])
    g = goal_text(r["prompt"])
    if g == "F":
        return None
    ng = f"( ~ {g} )"
    if ng in prem:
        return None
    n_pr = sum(1 for ln in lines if ln["rule"] == "PR")
    gf, _ = LG.parse_formula(g.split(), 0)
    if kind == "bot":
        new = [dict(ln) for ln in lines[:n_pr]]
        new.append(dict(idx=10_000, depth=0, formula=("not", gf), rule="PR", refs=[]))
        for ln in lines[n_pr:]:
            new.append(dict(ln))
        last = lines[-1]["idx"]
        new.append(dict(idx=10_001, depth=0, formula=("bot",), rule="NEGE", refs=[last, 10_000]))
        prompt = "THM " + " , ".join(prem + [ng]) + " SEQ F PRF"
    else:
        new = [dict(ln) for ln in lines[:n_pr]]
        new.append(dict(idx=10_000, depth=1, formula=("not", gf), rule="AS", refs=[]))
        for ln in lines[n_pr:]:
            new.append(dict(ln, depth=ln["depth"] + 1))
        last = lines[-1]["idx"]
        new.append(dict(idx=10_001, depth=1, formula=("bot",), rule="NEGE", refs=[last, 10_000]))
        new.append(dict(idx=10_002, depth=0, formula=("not", ("not", gf)), rule="NEGI", refs=[10_000, 10_001]))
        prompt = "THM " + " , ".join(prem) + " SEQ ( ~ " + ng + " ) PRF" if prem else "THM SEQ ( ~ " + ng + " ) PRF"
    return prompt, lines_to_text(new)


def cmd_dual(a):
    rng = random.Random(a.seed)
    rows = minimal_rows()
    rng.shuffle(rows)
    known = k12_keys()
    out, rep = [], collections.Counter()
    for k, (i, r, lines) in enumerate(rows):
        if r.get("contra_prem"):
            continue  # a ⊥ or ¬¬ goal from contradictory premises teaches nothing about the goal
        kind = "bot" if k % 2 == 0 else "dn"
        d = dualise(r, lines, kind)
        if d is None:
            rep["skip"] += 1
            continue
        prompt, proof = d
        if BR.keys(prompt) & known:
            rep["known"] += 1
            continue
        rec, why = PC.make_record(prompt, proof, "dual", kind=kind, parent=r["prompt"])
        if rec is None:
            rep[f"reject:{why}"] += 1
            continue
        out.append(rec)
        rep[kind] += 1
        if len(out) >= a.n:
            break
    log("duals", dict(rep))
    finish(out, "dual")


# ---------------------------------------------------------------- shape / balance (resampling the generator population)

def shape_cell(prompt, proof_lines, n_prem):
    _, goal = LG.parse_prompt(prompt)
    g = goal[0]
    n = len(proof_lines)
    lb = "2-4" if n <= 4 else "5-6" if n <= 6 else "7-8" if n <= 8 else "9-10" if n <= 10 else "11-12" if n <= 12 else "13+"
    d = min(max(ln["depth"] for ln in proof_lines), 3)
    return (g, min(n_prem, 4), lb, d)


def population_features():
    """one pass over k12_500k ∪ gen_depth3: (file, line_no, cell, rarest-rule weight basis) per record"""
    feats = []
    rule_count = collections.Counter()
    for fn in (K500, DEPTH3):
        t0 = time.time()
        for k, r in enumerate(iter_rows(fn)):
            lines = parse(r["proof"])
            if not lines:
                continue
            rules = frozenset(ln["rule"] for ln in prune_lines(lines) if ln["rule"] not in ("PR", "AS", "R"))
            rule_count.update(rules)
            feats.append((fn, k, shape_cell(r["prompt"], lines, r["n_prem"]), rules))
        log(os.path.basename(fn), "features", len(feats), f"{time.time() - t0:.0f}s")
    return feats, rule_count


def pull(selected):
    """read the selected (file, line_no) records back, in selection order"""
    want = collections.defaultdict(set)
    for fn, k in selected:
        want[fn].add(k)
    got = {}
    for fn in want:
        for k, r in enumerate(iter_rows(fn)):
            if k in want[fn]:
                got[(fn, k)] = r
    return [got[s] for s in selected]


def target_shape():
    val = validation_set()
    c = collections.Counter()
    for v in val:
        lines = parse(v["proof"])
        prem, _ = LG.parse_prompt(v["prompt"])
        c[shape_cell(v["prompt"], lines, len(prem))] += 1
    return c, len(val)


def cmd_shape(a):
    rng = random.Random(a.seed)
    feats, _ = population_features()
    pop = collections.defaultdict(list)
    for fn, k, cell, _r in feats:
        pop[cell].append((fn, k))
    tgt, n_val = target_shape()
    # smoothed target: validation histogram plus alpha * population histogram (keeps every reachable cell at a floor)
    n_pop = len(feats)
    alpha = a.alpha * n_val
    t = {}
    for cell in set(tgt) | set(pop):
        t[cell] = (tgt.get(cell, 0) + alpha * len(pop.get(cell, [])) / n_pop) / (n_val + alpha)
    need = N_POOL - VAL_N
    # unreachable target mass (cells the generator never produces) is reported and redistributed proportionally
    unreachable = {c: t[c] for c in t if not pop.get(c)}
    reach = {c: t[c] for c in t if pop.get(c)}
    # water-filling: a cell with fewer candidates than its quota gives all of them; the rest share the remainder
    chosen, active, remaining = {}, set(reach), need
    while True:
        z = sum(reach[c] for c in active)
        over = [c for c in active if len(pop[c]) < reach[c] / z * remaining]
        if not over:
            break
        for c in over:
            chosen[c] = len(pop[c])
            remaining -= len(pop[c])
            active.remove(c)
    z = sum(reach[c] for c in active)
    for c in active:
        chosen[c] = int(round(reach[c] / z * remaining))
    sel = []
    for c, n in chosen.items():
        sel.extend(rng.sample(pop[c], min(n, len(pop[c]))))
    rng.shuffle(sel)
    sel = sel[:need]
    if len(sel) < need:  # rounding: top up from the largest cells
        have = set(sel)
        extra = [x for c in sorted(pop, key=lambda c: -len(pop[c])) for x in pop[c] if x not in have]
        sel.extend(extra[: need - len(sel)])
    rows = pull(sel)
    rows = [dict(r, pool="shape") for r in rows]
    got = collections.Counter(shape_cell(r["prompt"], parse(r["proof"]), r["n_prem"]) for r in rows)
    rep = dict(n_val=n_val, population=n_pop, alpha=alpha,
               unreachable_target_mass=round(sum(unreachable.values()), 3),
               unreachable_cells=[[list(map(str, c)), round(v, 4)] for c, v in sorted(unreachable.items(), key=lambda x: -x[1])[:12]],
               cells=[[list(map(str, c)), tgt.get(c, 0), round(t[c], 4), len(pop.get(c, [])), got.get(c, 0)] for c in sorted(set(tgt) | set(got), key=lambda c: -t.get(c, 0))])
    os.makedirs(NEW, exist_ok=True)
    PC.write(rows, os.path.join(NEW, "shape.jsonl.gz"))
    json.dump(rep, open(os.path.join(NEW, "shape.stats.json"), "w"), indent=1, ensure_ascii=False)
    log("shape", len(rows), "unreachable target mass", rep["unreachable_target_mass"])


def cmd_balance(a):
    rng = random.Random(a.seed)
    feats, rule_count = population_features()
    n_pop = len(feats)
    share = {r: rule_count[r] / n_pop for r in RULES}
    keys = []
    for fn, k, _cell, rules in feats:
        if not rules:
            w = 1.0
        else:
            w = (1.0 / min(share[r] for r in rules if r in share)) ** a.power
        keys.append((rng.random() ** (1.0 / w), (fn, k)))  # Efraimidis–Spirakis: weighted sampling without replacement
    keys.sort(reverse=True)
    need = N_POOL - VAL_N
    sel = [x for _, x in keys[:need]]
    rows = [dict(r, pool="balance") for r in pull(sel)]
    after = collections.Counter()
    for r in rows:
        lines = parse(r["proof"])
        after.update({ln["rule"] for ln in prune_lines(lines) if ln["rule"] in share})
    rep = dict(population=n_pop, power=a.power, share_before={r: round(share[r], 4) for r in RULES},
               share_after={r: round(after[r] / len(rows), 4) for r in RULES})
    os.makedirs(NEW, exist_ok=True)
    PC.write(rows, os.path.join(NEW, "balance.jsonl.gz"))
    json.dump(rep, open(os.path.join(NEW, "balance.stats.json"), "w"), indent=1, ensure_ascii=False)
    log("balance", len(rows), rep)


# ---------------------------------------------------------------- assemble

def perm_maps(rng):
    """the 23 non-identity atom renamings, shuffled"""
    import itertools
    atoms = ["P", "Q", "R", "S"]
    maps = [dict(zip(atoms, p)) for p in itertools.permutations(atoms) if list(p) != atoms]
    rng.shuffle(maps)
    return maps


def flat_slice(src, need, rng, lo=3, hi=12):
    """a slice flat over line counts lo..hi where the source allows it"""
    by = collections.defaultdict(list)
    for r in src:
        by[min(max(r["n_lines"], lo), hi)].append(r)
    for L in by:
        rng.shuffle(by[L])
    out, left = [], need
    order = sorted(by, key=lambda L: len(by[L]))  # smallest bins first, so their shortfall spreads over the larger ones
    for k, L in enumerate(order):
        q = left // (len(order) - k)
        take = by[L][:q]
        out.extend(take)
        left -= len(take)
    rng.shuffle(out)
    return out[:need]


def cmd_assemble(a):
    rng = random.Random(a.seed)
    k12 = PC.load_k12()
    N = len(k12)
    free = list(range(VAL_N, N))
    pools, notes = {}, {}

    def positions(n):
        return rng.sample(free, n)

    def src(name):
        p = os.path.join(NEW, f"{name}.jsonl.gz")
        return PC.read(p) if os.path.exists(p) else None

    cl = src("closure")
    if cl:
        n = int(round(0.10 * N))
        rows = flat_slice(cl, n, rng)
        if len(rows) < n:
            rows, note = SP.nested_slice(cl, n, rng, "closure")
        else:
            note = dict(source=len(cl), perms=0, repeats=0)
        pools["k12_closure10"] = SP.replace_positions(k12, positions(n), rows, "closure")
        notes["k12_closure10"] = dict(note, by_len=dict(collections.Counter(r["n_lines"] for r in rows)))
    co = src("compose")
    if co:
        n = int(round(0.10 * N))
        rows, note = SP.nested_slice(co, n, rng, "compose")
        pools["k12_compose10"] = SP.replace_positions(k12, positions(n), rows, "compose")
        notes["k12_compose10"] = note
    rc = src("recover")
    if rc:
        n = min(len(rc), int(round(0.20 * N)))
        rng.shuffle(rc)
        rows = rc[:n]
        pools["k12_recover20"] = SP.replace_positions(k12, [r["k12_index"] for r in rows], rows, "recover")
        notes["k12_recover20"] = dict(source=len(rc), used=n, by_dead_rule=dict(collections.Counter(r["dead_rule"] for r in rows)))
    mu = src("multi")
    if mu:
        # the second proofs first (their originals are already in K12), then atom-permuted PAIRS (original + second proof
        # under the same renaming) so that every permuted theorem in the slice also carries two proofs
        n = int(round(0.10 * N))
        rng.shuffle(mu)
        rows = list(mu)
        pairs = 0
        for r in mu:
            if len(rows) >= n:
                break
            orig = k12[r["k12_index"]]
            for m in perm_maps(rng):
                if len(rows) >= n:
                    break
                p_o = PC.rename_text(orig["prompt"], m)
                if p_o == orig["prompt"]:
                    continue
                r_o, _ = PC.make_record(p_o, PC.rename_text(orig["proof"], m), "multi", k12_index=r["k12_index"], perm_of=orig["prompt"], pair="original")
                r_s, _ = PC.make_record(p_o, PC.rename_text(r["proof"], m), "multi", k12_index=r["k12_index"], perm_of=orig["prompt"], pair="second", variant=r["variant"])
                if r_o and r_s:
                    rows.extend([r_o, r_s])
                    pairs += 1
        rows = rows[:n]
        note = dict(source=len(mu), permuted_pairs=pairs, repeats=0)
        protected = {r["k12_index"] for r in mu}
        pos = rng.sample([i for i in free if i not in protected], len(rows))
        pools["k12_multi10"] = SP.replace_positions(k12, pos, rows, "multi")
        notes["k12_multi10"] = dict(note, by_variant=dict(collections.Counter(r.get("variant", "") for r in mu)),
                                    new_rule_multiset=sum(r["new_rules"] for r in mu))
    du = src("dual")
    if du:
        n = int(round(0.01 * N))
        rows, note = SP.nested_slice(du, n, rng, "dual")
        pools["k12_dual1"] = SP.replace_positions(k12, positions(n), rows, "dual")
        notes["k12_dual1"] = dict(note, by_kind=dict(collections.Counter(r["kind"] for r in rows)))
    for name in ("shape", "balance"):
        rows = src(name)
        if rows:
            body = rows[: N - VAL_N]
            pools[f"k12_{name}"] = k12[:VAL_N] + [dict(r, slice=name) for r in body]
            notes[f"k12_{name}"] = json.load(open(os.path.join(NEW, f"{name}.stats.json")))
            if name == "shape":
                notes[f"k12_{name}"] = {k: v for k, v in notes[f"k12_{name}"].items() if k != "cells"}
    os.makedirs(OUT_DIR, exist_ok=True)
    pieces = PC.hard_pieces()
    st = []
    base = PC.stats(k12, "k12", pieces)
    base.update(iff_elim=SP.iff_elim_rate(k12), unused_prem=SP.unused_prem_share(k12),
                mean_tokens=round(sum(SP.n_tokens(r) for r in k12) / N, 1), max_tokens=max(SP.n_tokens(r) for r in k12))
    st.append(base)
    for name, rows in pools.items():
        log("stats", name)
        assert len(rows) == N and all(rows[i] is k12[i] or rows[i]["prompt"] == k12[i]["prompt"] for i in range(VAL_N)), name
        s = PC.stats(rows, name, pieces)
        s.update(iff_elim=SP.iff_elim_rate(rows), unused_prem=SP.unused_prem_share(rows),
                 mean_tokens=round(sum(SP.n_tokens(r) for r in rows) / N, 1), max_tokens=max(SP.n_tokens(r) for r in rows),
                 note=notes.get(name, {}))
        st.append(s)
        if not a.no_write:
            PC.write(rows, os.path.join(OUT_DIR, f"{name}.jsonl.gz"))
    json.dump(dict(pools=st, notes=notes), open(os.path.join(OUT_DIR, "stats.json"), "w"), indent=1, ensure_ascii=False)
    write_doc(st)
    log("wrote", OUT_DIR)


HEAD = "## New pools (2026-10-07): closure, composition, recovery, second proofs, duals, shape- and rule-resampling"


def write_doc(st):
    W = [f"\n{HEAD}\n",
         "Built by `new_pools.py`. Every pool has 155,000 records with K12's first 1,024 rows at its head. "
         "`k12_shape` and `k12_balance` are whole-pool resamples of the generator population (`k12_500k` ∪ the depth-3 side "
         "pool, ≈ 987k verified, audited records); the rest are K12 with a slice replaced. Columns as in \"Scan pools\".\n",
         "| pool | n | mean lines | prem ≥4 | depth ≥3 | ↔ rows | ↔-elim /1k | unused prem | mean tokens | max tokens | ∧E share | ∨E share | DN share | hard pieces | unknown pieces |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in st:
        W.append(f"| {s['label']} | {s['n']:,} | {s['mean_lines']} | {s['prem_ge4']:.4f} | {s['depth_ge3']:.4f} | {s['bicond']:.4f} | {s['iff_elim']} | "
                 f"{s['unused_prem']:.4f} | {s['mean_tokens']} | {s['max_tokens']} | {s['ande_share']:.3f} | {s['ore_share']:.3f} | {s['dn_share']:.3f} | "
                 f"{s.get('hard_pieces_covered', '')} | {s.get('unknown_pieces_covered', '')} |")
    W.append("\nJoin rates per 1,000 citations:\n")
    cols = list(PC.JOINS)
    W.append("| pool | " + " | ".join(cols) + " |")
    W.append("| --- | " + " | ".join("---" for _ in cols) + " |")
    for s in st:
        W.append(f"| {s['label']} | " + " | ".join(str(s["joins"][c]) for c in cols) + " |")
    W.append("\nComposition notes:\n")
    for s in st:
        if s.get("note"):
            W.append(f"- `{s['label']}`: {json.dumps(s['note'], ensure_ascii=False)}")
    W.append("\nLength, premise and depth distributions:\n")
    for s in st:
        W.append(f"- `{s['label']}`: lines {s['by_len']}; premises {s['by_prem']}; depth {s['by_depth']}")
    text = open(DOC).read() if os.path.exists(DOC) else ""
    i = text.find(HEAD)
    if i >= 0:
        j = text.find("\n## ", i + len(HEAD))
        text = text[:i].rstrip("\n") + "\n" + (text[j:] if j > 0 else "")
    with open(DOC, "w") as f:
        f.write(text.rstrip("\n") + "\n" + "\n".join(W) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("closure")
    p.add_argument("--n", type=int, default=60000)
    p.add_argument("--deep", type=int, default=60000, help="depth-3 side-pool rows to mine")
    p.add_argument("--min_lines", type=int, default=3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("compose")
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--max_part", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--secs", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("recover")
    p.add_argument("--n", type=int, default=33000)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("multi")
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--secs", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("dual")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("shape")
    p.add_argument("--alpha", type=float, default=0.2, help="smoothing: population prior weight as a fraction of n_val")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("balance")
    p.add_argument("--power", type=float, default=2.0, help="weight = (1 / share of the rarest rule) ** power")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("assemble")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_write", action="store_true")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
