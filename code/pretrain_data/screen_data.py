"""Data-only screens for Stage-1 pools (ideas.md § J, items J1–J3, J17): no model trained, no checkpoint loaded.

Validation set = textbook-shaped problems of the SEED split only (sets v0 and lean_v2, and the derived problems whose base
is in those sets), with their reference proofs from the release. The sealed 72 and the test split (v1, batch3, hand14 and
derived problems based on them) are never read here.

Screens, one row per pool:
  J3  shallow next-step predictor: a smoothed count model P(next rule | goal connective, last two rules, box depth) fitted on
      the pool's proofs, read as perplexity and top-1 accuracy over the validation proofs' steps (a lookup-table stand-in for
      held-out loss).
  J2  nearest rule-sequence distance: for each validation proof, the Levenshtein distance between its (rule, depth) sequence
      and the nearest pool proof's; reported as the share of validation proofs at distance 0, ≤ 1, ≤ 2, ≤ 3, and the mean
      (capped at 6).
  J1  need-weighted cell coverage: a cell is (citing rule, cited rule, box depth of the citing line ≤ 3, premises ≤ 4);
      each cell is weighted by the number of validation problems *unsolved by every evaluated model* whose reference proof
      uses it; a pool fills a cell if ≥ 1 record in 15,500 (≥ 10 at K12's size) uses it. Score = weighted share filled.
  J17 the k12_wrong* pools are the negative control: data screens that ignore validity should put them next to K12.

    cd code && uv run python pretrain_data/screen_data.py            # all pools, appends "## Data screens" to pools.md
    cd code && uv run python pretrain_data/screen_data.py --pools k12 k12_minimal

Writes data/pretrain_data/screens/data_screens.json and the validation-set names (not the 72) next to it.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pool_common as PC  # noqa: E402
import eval_analysis as EA  # noqa: E402
import logic as LG  # noqa: E402
from patterns import parse, prune_lines  # noqa: E402

OUT = os.path.join(PC.DATA, "screens")
DOC = os.path.join(PC.ROOT, "docs", "pretrain-data", "pools.md")
SEED_SETS = ("v0", "lean_v2")
FILL_SHARE = 1 / 15_500  # ≥ 10 records at K12's size
MAX_D = 6
HEAD = "## Data screens (ideas.md § J1–J3): does the pool contain the shapes the textbook problems need?"


# ---------------------------------------------------------------- validation set

def validation_set():
    """seed-split problems with reference proofs: list of dicts(name, set, prompt, proof, unsolved)"""
    refs = EA.load_refs()
    rows = {r["name"]: r for r in csv.DictReader(open(PC.CSV))}
    out = []
    for name, r in refs.items():
        row = rows.get(name)
        if row is None or not r.get("reference_proof"):
            continue
        s = row["set"]
        if s == "derived":
            base = rows.get(r.get("base_name", ""), {}).get("set")
            if base not in SEED_SETS:
                continue
        elif s not in SEED_SETS:
            continue
        unsolved = all(float(row[c] or 0) == 0 for c in ("pt_d384", "pt_d768", "rl_s0", "rl_s1"))
        lines = parse(r["reference_proof"])
        if not lines:
            continue
        out.append(dict(name=name, set=s, prompt=r["prompt"], proof=r["reference_proof"], unsolved=unsolved, n_lines=len(lines)))
    return out


# ---------------------------------------------------------------- features

def steps(prompt, lines):
    """(context, rule) pairs for the shallow predictor; context = (goal connective, rule[-2], rule[-1], depth)"""
    prem, goal = LG.parse_prompt(prompt)
    g = goal[0]
    hist = ["<s>", "<s>"]
    out = []
    for ln in lines:
        if ln["rule"] == "PR":
            continue
        out.append(((g, hist[-2], hist[-1], min(ln["depth"], 3)), ln["rule"]))
        hist.append(ln["rule"])
    out.append(((g, hist[-2], hist[-1], 0), "<end>"))
    return out


def seq(lines):
    """the (rule, depth) sequence of the pruned proof, PR lines dropped, as a tuple of small ints"""
    return tuple(RULE_ID.get((ln["rule"], min(ln["depth"], 3)), 0) for ln in lines if ln["rule"] != "PR")


RULES = ["AS", "R", "ANDI", "ANDE1", "ANDE2", "IMPE", "IMPI", "ORI1", "ORI2", "ORE", "NEGE", "NEGI", "BOTE", "DN"]
RULE_ID = {(r, d): 1 + i * 4 + d for i, r in enumerate(RULES) for d in range(4)}


def cells(lines, n_prem):
    """(with-premise-count cells, without) — the second ignores how many premises the problem has"""
    by = {ln["idx"]: ln for ln in lines}
    out, out_np = set(), set()
    for ln in lines:
        for ref in ln["refs"]:
            if ref in by and by[ref]["rule"] != "PR":
                out.add((ln["rule"], by[ref]["rule"], min(ln["depth"], 3), min(n_prem, 4)))
                out_np.add((ln["rule"], by[ref]["rule"], min(ln["depth"], 3)))
    return out, out_np


# ---------------------------------------------------------------- screens

class CountModel:
    """P(rule | context) with back-off: full context → (last rule, depth) → depth → unigram; add-0.5 smoothing"""

    def __init__(self):
        self.full = collections.defaultdict(collections.Counter)
        self.mid = collections.defaultdict(collections.Counter)
        self.low = collections.defaultdict(collections.Counter)
        self.uni = collections.Counter()
        self.vocab = set(RULES) | {"<end>"}

    def fit(self, pairs):
        for ctx, r in pairs:
            self.full[ctx][r] += 1
            self.mid[(ctx[2], ctx[3])][r] += 1
            self.low[ctx[3]][r] += 1
            self.uni[r] += 1
            self.vocab.add(r)

    def prob(self, ctx, r):
        V = len(self.vocab)
        p = (self.uni[r] + 0.5) / (sum(self.uni.values()) + 0.5 * V)
        for table, key in ((self.low, ctx[3]), (self.mid, (ctx[2], ctx[3])), (self.full, ctx)):
            c = table.get(key)
            if c:
                n = sum(c.values())
                lam = n / (n + 2.0)
                p = lam * (c[r] / n) + (1 - lam) * p
        return p

    def top1(self, ctx):
        return max(self.vocab, key=lambda r: self.prob(ctx, r))


def levenshtein_capped(a, b, cap):
    """edit distance between tuples a, b, or cap+1 if it exceeds cap (banded)"""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        lo, hi = max(1, i - cap), min(len(b), i + cap)
        best = cap + 1
        for j in range(1, len(b) + 1):
            if j < lo or j > hi:
                cur[j] = cap + 1
                continue
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != b[j - 1]))
            best = min(best, cur[j])
        if best > cap:
            return cap + 1
        prev = cur
    return min(prev[-1], cap + 1)


def nearest(vs, pool_seqs, cap=MAX_D):
    """min edit distance from vs to any sequence in pool_seqs (a set), pruned by length and a bag-of-symbols bound"""
    if vs in pool_seqs:
        return 0
    bag_v = collections.Counter(vs)
    best = cap + 1
    for ps in pool_seqs:
        if abs(len(ps) - len(vs)) >= best:
            continue
        # bag distance lower bound: max(|a\b|, |b\a|) ≤ edit distance
        bag_p = BAGS[ps]
        lb = max(sum((bag_v - bag_p).values()), sum((bag_p - bag_v).values()))
        if lb >= best:
            continue
        d = levenshtein_capped(vs, ps, best - 1)
        if d < best:
            best = d
            if best == 1:
                break
    return best


BAGS = {}


def screen_pool(rows, val, need_w, label):
    t0 = time.time()
    model = CountModel()
    pool_seqs = set()
    cell_count = collections.Counter()
    cell_count_np = collections.Counter()
    n = 0
    for r in rows:
        lines = parse(r["proof"])
        if not lines:
            continue
        n += 1
        pr = prune_lines(lines)
        model.fit(steps(r["prompt"], pr))
        pool_seqs.add(seq(pr))
        cs, cs_np = cells(pr, r["n_prem"])
        for c in cs:
            cell_count[c] += 1
        for c in cs_np:
            cell_count_np[c] += 1
    for ps in pool_seqs:
        if ps not in BAGS:
            BAGS[ps] = collections.Counter(ps)
    # J3
    nlls, hits = [], 0
    for v in val:
        for ctx, r in steps(v["prompt"], prune_lines(v["lines"])):
            pr_ = model.prob(ctx, r)
            nlls.append(-math.log(pr_))
            hits += model.top1(ctx) == r
    m = len(nlls)
    ppl = math.exp(sum(nlls) / m)
    nll_med = sorted(nlls)[m // 2]
    surprise = sum(1 for x in nlls if x > -math.log(0.01)) / m
    # J2
    ds = [nearest(v["seq"], pool_seqs) for v in val]
    by_len = collections.defaultdict(list)
    for v, d in zip(val, ds):
        by_len["≤6" if v["n_lines"] <= 6 else "7–12" if v["n_lines"] <= 12 else "13+"].append(d)
    # J1
    thr = max(1, math.ceil(FILL_SHARE * n))
    filled = {c for c, k in cell_count.items() if k >= thr}
    filled_np = {c for c, k in cell_count_np.items() if k >= thr}
    need_w, need_w_np = need_w
    w_total = sum(need_w.values())
    w_filled = sum(w for c, w in need_w.items() if c in filled)
    w_total_np = sum(need_w_np.values())
    w_filled_np = sum(w for c, w in need_w_np.items() if c in filled_np)
    out = dict(label=label, n=n, distinct_seqs=len(pool_seqs), ppl=round(ppl, 3), acc=round(hits / m, 4),
               nll_med=round(nll_med, 3), surprise=round(surprise, 4),
               d0=round(sum(d == 0 for d in ds) / len(ds), 3), d1=round(sum(d <= 1 for d in ds) / len(ds), 3),
               d2=round(sum(d <= 2 for d in ds) / len(ds), 3), d3=round(sum(d <= 3 for d in ds) / len(ds), 3),
               d_mean=round(sum(min(d, MAX_D) for d in ds) / len(ds), 2),
               d_mean_by_len={k: round(sum(min(d, MAX_D) for d in v) / len(v), 2) for k, v in sorted(by_len.items())},
               need_cells=len(need_w), need_filled=sum(1 for c in need_w if c in filled), need_weighted=round(w_filled / w_total, 3),
               need_cells_np=len(need_w_np), need_filled_np=sum(1 for c in need_w_np if c in filled_np), need_weighted_np=round(w_filled_np / w_total_np, 3),
               fill_threshold=thr, secs=round(time.time() - t0))
    print(json.dumps(out), flush=True)
    return out


def write_doc(results, val, n_unsolved):
    W = [f"\n{HEAD}\n",
         f"Built by `screen_data.py`. No model is trained and no checkpoint is read. Validation set: {len(val)} seed-split problems "
         f"(v0, lean_v2 and the derived problems based on them) with reference proofs, {n_unsolved} of them unsolved by every evaluated "
         f"model; the sealed 72 and the test split are not used. "
         "`ppl` / `acc` / `med` / `surprise` = perplexity, top-1 accuracy, median per-step loss (nats) and share of steps given p < 0.01 by a smoothed "
         "count model P(next rule | goal connective, last two rules, box depth) fitted on the pool and read on the validation proofs' steps. `d≤k` = share of validation proofs whose "
         "(rule, depth) sequence is within k edits of some pool proof's; `d̄` = mean distance capped at 6, overall and by reference length. "
         "`need` = type cells (citing rule, cited rule, depth, premises) used by the unsolved validation problems' reference proofs, "
         "weighted by how many unsolved problems use them; `filled` = cells the pool shows in ≥ 1 record per 15,500; the `no-prem` variant drops the "
         "premise count from the cell, so adding premises alone does not fill cells. "
         "The `k12_wrong*` pools are the negative control: these screens ignore validity, so they should sit next to K12.\n",
         "| pool | n | distinct seqs | ppl | acc | med | surprise | d=0 | d≤1 | d≤2 | d≤3 | d̄ | d̄ ≤6 / 7–12 / 13+ | need filled | need-weighted | no-prem filled | no-prem weighted |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in results:
        bl = s["d_mean_by_len"]
        W.append(f"| {s['label']} | {s['n']:,} | {s['distinct_seqs']:,} | {s['ppl']} | {s['acc']:.3f} | {s['nll_med']} | {s['surprise']:.3f} | {s['d0']:.2f} | {s['d1']:.2f} | {s['d2']:.2f} | "
                 f"{s['d3']:.2f} | {s['d_mean']} | {bl.get('≤6', '')} / {bl.get('7–12', '')} / {bl.get('13+', '')} | "
                 f"{s['need_filled']}/{s['need_cells']} | {s['need_weighted']:.3f} | {s['need_filled_np']}/{s['need_cells_np']} | {s['need_weighted_np']:.3f} |")
    W.append("")
    text = open(DOC).read()
    if HEAD in text:
        i = text.index(HEAD)
        j = text.find("\n## ", i + len(HEAD))
        text = text[:i].rstrip("\n") + "\n" + (text[j:] if j > 0 else "")
    with open(DOC, "w") as f:
        f.write(text.rstrip("\n") + "\n" + "\n".join(W) + "\n")


def pool_files():
    files = {"k12": None, "cap6 (take-home Stage-1 set)": os.path.join(PC.ROOT, "code", "ws3", "fork_pin", "data", "train.jsonl.gz")}
    for sub in ("", "size", "scan", "new", "atoms", "k"):
        for p in sorted(glob.glob(os.path.join(PC.DATA, "pools", sub, "*.jsonl.gz"))):
            name = os.path.basename(p)[: -len(".jsonl.gz")]
            if name != "gen_depth3":
                files[name] = p
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="*", help="pool names (default: all)")
    ap.add_argument("--no_doc", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    val = validation_set()
    for v in val:
        v["lines"] = parse(v["proof"])
        v["seq"] = seq(prune_lines(v["lines"]))
    need_w, need_w_np = collections.Counter(), collections.Counter()
    for v in val:
        if v["unsolved"]:
            prem, _ = LG.parse_prompt(v["prompt"])
            cs, cs_np = cells(prune_lines(v["lines"]), len(prem))
            for c in cs:
                need_w[c] += 1
            for c in cs_np:
                need_w_np[c] += 1
    n_unsolved = sum(v["unsolved"] for v in val)
    json.dump(dict(names=[v["name"] for v in val], sets=dict(collections.Counter(v["set"] for v in val)), unsolved=n_unsolved,
                   need_cells=[[list(c), w] for c, w in sorted(need_w.items(), key=lambda x: -x[1])]),
              open(os.path.join(OUT, "validation_set.json"), "w"), indent=1, ensure_ascii=False)
    print(f"validation set: {len(val)} problems, {n_unsolved} unsolved, {len(need_w)} needed cells", flush=True)
    files = pool_files()
    names = a.pools or list(files)
    results = []
    prev = {}
    rp = os.path.join(OUT, "data_screens.json")
    if os.path.exists(rp):
        prev = {s["label"]: s for s in json.load(open(rp))["pools"]}
    for name in names:
        p = files[name]
        rows = PC.load_k12() if p is None else PC.read(p)
        if name.startswith("cap6"):
            for r in rows:
                r.setdefault("n_prem", len(LG.parse_prompt(r["prompt"])[0]))
        results.append(screen_pool(rows, val, (need_w, need_w_np), name))
        prev[name] = results[-1]
        ordered = [prev[k] for k in files if k in prev]
        json.dump(dict(validation=len(val), unsolved=n_unsolved, pools=ordered), open(rp, "w"), indent=1, ensure_ascii=False)
    if not a.no_doc:
        write_doc([prev[k] for k in files if k in prev], val, n_unsolved)
        print("wrote", DOC)


if __name__ == "__main__":
    main()
