"""Pool-to-pool similarity: how much does each Stage-1 pool share with every other, and with K12 in particular?

The stats table (mix.py) and the data screens (screen_data.py) describe each pool on its own, or against the seed split.
This script compares pools with each other, on four leak-safe views (classes and skeletons, never string matches):

  records   share of a pool's (prompt, proof) records that occur verbatim in K12 (how much of the pool is K12 untouched)
  classes   share of a pool's theorem renaming classes (curry-normal `canon_key`) that occur in K12, and the Jaccard
            overlap of class sets between pools (new theorems vs the same theorems re-proved or re-weighted)
  skeleton  Jensen–Shannon distance (base 2, in [0, 1]) between pools' distributions of (rule, depth) trigrams of the
            pruned proofs, and between their rule-bigram distributions (same theorems, different proofs). Whole
            (rule, depth) sequences are too fine: two disjoint halves of K12 differ at JS 0.36 on them, trigrams at 0.05.
  shape     Jensen–Shannon distance between pools' (goal connective, premises ≤ 4, length bin, depth ≤ 3) histograms,
            the cells k12_shape targets (same proofs, different mixture)

Combined distance = mean of the three JS distances (skeleton, bigram, shape); class overlap is reported separately
because every K12-based pool shares ≥ 70 % of its classes with K12 and the Jaccard would only say that. The noise floor
of each view is the JS between two disjoint halves of K12 itself (77,500 records each), and between two disjoint 15,000-record
samples for reading the size pools.
Every pool is clustered by average linkage on the combined distance. Writes data/pretrain_data/screens/pool_similarity.json
and the section "## Pool-to-pool similarity" of docs/pretrain-data/pools.md.

    cd code && uv run python pretrain_data/pool_similarity.py
    cd code && uv run python pretrain_data/pool_similarity.py --pools k12 k12_minimal k12_recover20
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pool_common as PC  # noqa: E402
import logic as LG  # noqa: E402
from data import canon_key, thm_of_prompt  # noqa: E402
from patterns import parse, prune_lines  # noqa: E402
from release_1002 import BR  # noqa: E402
from screen_data import pool_files  # noqa: E402

OUT = os.path.join(PC.DATA, "screens")
PC_VAL_N = 1024  # K12's val head, shared by every pool
DOC = os.path.join(PC.ROOT, "docs", "pretrain-data", "pools.md")
HEAD = "## Pool-to-pool similarity"
# the pools built before new_pools.py (the "earlier" set a new pool is compared with)
SMALL = 15_000  # the smallest pool's size; the second noise floor is read at this size
NEW = ("k12_closure10", "k12_compose10", "k12_recover20", "k12_multi10", "k12_dual1", "k12_shape", "k12_balance")


def class_of(prompt):
    """one renaming-class key per theorem, curry-aware (the `ck` member of BR.keys on the curry-normal prompt)"""
    return canon_key(thm_of_prompt(BR.curry_normal(prompt)))


def shape_cell(prompt, lines, n_prem):
    _, goal = LG.parse_prompt(prompt)
    n = len(lines)
    lb = "2-4" if n <= 4 else "5-6" if n <= 6 else "7-8" if n <= 8 else "9-10" if n <= 10 else "11-12" if n <= 12 else "13+"
    return (goal[0], min(n_prem, 4), lb, min(max(ln["depth"] for ln in lines), 3))


def features(rows, label):
    """per-pool summaries: record hashes, class set, and the three distributions"""
    t0 = time.time()
    recs, classes = set(), set()
    seqs, bigrams, shapes = collections.Counter(), collections.Counter(), collections.Counter()
    n = 0
    for r in rows:
        lines = parse(r["proof"])
        if not lines:
            continue
        n += 1
        recs.add(hash((r["prompt"], r["proof"])))
        classes.add(class_of(r["prompt"]))
        pr = prune_lines(lines)
        rd = [(ln["rule"], min(ln["depth"], 3)) for ln in pr if ln["rule"] != "PR"]
        rules = [x[0] for x in rd]
        for a, b in zip(["<s>"] + rules, rules + ["<end>"]):
            bigrams[(a, b)] += 1
        for a, b, c in zip([("<s>", 0)] + rd, rd + [("<end>", 0)], (rd + [("<end>", 0), ("<end>", 0)])[1:]):
            seqs[(a, b, c)] += 1
        n_prem = r.get("n_prem") or len(LG.parse_prompt(r["prompt"])[0])
        shapes[shape_cell(r["prompt"], lines, n_prem)] += 1
    print(f"{label}: n={n:,} classes={len(classes):,} trigrams={len(seqs):,} cells={len(shapes)} ({time.time() - t0:.0f}s)", flush=True)
    return dict(label=label, n=n, recs=recs, classes=classes, seqs=seqs, bigrams=bigrams, shapes=shapes)


def noise_floor(k12_rows, small=SMALL):
    """JS between two disjoint halves of K12, and between two disjoint `small`-record samples of it"""
    rows = k12_rows[PC_VAL_N:]
    out = {}
    for tag, (ra, rb) in (("half", (rows[0::2], rows[1::2])), ("small", (rows[0::2][:small], rows[1::2][:small]))):
        a, b = features(ra, f"k12 {tag} A"), features(rb, f"k12 {tag} B")
        js = {v: float(js_matrix([a[v], b[v]])[0, 1]) for v in ("seqs", "bigrams", "shapes")}
        out[tag] = dict(js_seq=js["seqs"], js_bigram=js["bigrams"], js_shape=js["shapes"], combined=sum(js.values()) / 3)
    return out


def js_matrix(counters):
    """pairwise Jensen–Shannon distance (base 2) between distributions given as Counters"""
    vocab = {}
    for c in counters:
        for k in c:
            vocab.setdefault(k, len(vocab))
    P = np.zeros((len(counters), len(vocab)), dtype=np.float64)
    for i, c in enumerate(counters):
        for k, v in c.items():
            P[i, vocab[k]] = v
    P /= P.sum(axis=1, keepdims=True)
    D = np.zeros((len(counters), len(counters)))

    def kl(p, m):
        nz = p > 0
        return float(np.sum(p[nz] * np.log2(p[nz] / m[nz])))

    for i in range(len(counters)):
        for j in range(i + 1, len(counters)):
            m = 0.5 * (P[i] + P[j])
            d = math.sqrt(max(0.0, 0.5 * kl(P[i], m) + 0.5 * kl(P[j], m)))
            D[i, j] = D[j, i] = d
    return D


def average_linkage(D, names):
    """agglomerative clustering; returns the merge list [(left, right, distance)] with nested-tuple cluster labels"""
    clusters = {i: (names[i],) for i in range(len(names))}
    dist = {(i, j): D[i, j] for i in range(len(names)) for j in range(i + 1, len(names))}
    merges = []
    nxt = len(names)
    while len(clusters) > 1:
        (i, j), d = min(dist.items(), key=lambda kv: kv[1])
        merges.append((clusters[i], clusters[j], d))
        new = clusters[i] + clusters[j]
        ni, nj = len(clusters[i]), len(clusters[j])
        del clusters[i], clusters[j]
        for k in clusters:
            a, b = dist.pop((min(i, k), max(i, k))), dist.pop((min(j, k), max(j, k)))
            dist[(k, nxt)] = (a * ni + b * nj) / (ni + nj)
        dist.pop((i, j), None)
        clusters[nxt] = new
        nxt += 1
    return merges


def flat_clusters(merges, names, cut):
    """clusters at distance `cut` (merges above it are undone), largest first"""
    groups = {n: {n} for n in names}
    for left, right, d in merges:
        if d > cut:
            break
        merged = set(left) | set(right)
        for n in merged:
            groups[n] = merged
    out, seen = [], set()
    for n in names:
        if n in seen:
            continue
        g = sorted(groups[n])
        seen |= set(g)
        out.append(g)
    return sorted(out, key=lambda g: (-len(g), g[0]))


def write_doc(res, names):
    k12 = res["vs_k12"]
    fl = res["floor"]
    W = [f"\n{HEAD}\n",
         "Built by `pool_similarity.py`; compares pools with each other (the stats table and the data screens describe each pool alone or "
         "against the seed split). `records in K12` = share of (prompt, proof) records identical to a K12 record; `classes in K12` = share of "
         "theorem renaming classes (curry-normal `canon_key`) present in K12; `JS` = Jensen–Shannon distance (base 2; 0 identical, 1 disjoint) "
         "between the pool's and K12's distributions of (rule, depth) trigrams of the pruned proofs, of rule bigrams, and of (goal connective, "
         "premises, length bin, depth) shape cells; `combined` = mean of the three. Noise floor, two disjoint halves of K12 (77,500 records) against each "
         f"other: JS tri {fl['half']['js_seq']:.3f}, JS bigram {fl['half']['js_bigram']:.3f}, JS shape {fl['half']['js_shape']:.3f}, combined "
         f"{fl['half']['combined']:.3f}; two disjoint {SMALL:,}-record samples (the size pools' scale): {fl['small']['js_seq']:.3f} / "
         f"{fl['small']['js_bigram']:.3f} / {fl['small']['js_shape']:.3f}, combined {fl['small']['combined']:.3f}. Distances at or below the floor mean "
         "'same distribution'. "
         "`nearest` = the three closest *other* pools by combined distance, the new pools (`new_pools.py`) marked †. No string matching: classes and skeletons only.\n",
         "| pool | records in K12 | classes in K12 | classes (distinct) | JS tri | JS bigram | JS shape | combined | nearest pools |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    mark = lambda n: f"{n}†" if n in NEW else n  # noqa: E731
    for name in names:
        v = k12[name]
        nn = ", ".join(f"{mark(m)} {d:.3f}" for m, d in res["nearest"][name])
        W.append(f"| {mark(name)} | {v['rec_in_k12']:.3f} | {v['class_in_k12']:.3f} | {v['n_classes']:,} | {v['js_seq']:.3f} | {v['js_bigram']:.3f} | "
                 f"{v['js_shape']:.3f} | {v['combined']:.3f} | {nn} |")
    W.append("")
    W.append(f"Average-linkage clusters on the combined distance, cut at {res['cut']}:\n")
    for g in res["clusters"]:
        W.append("- " + ", ".join(mark(n) for n in g))
    W.append("")
    W.append("Class-set Jaccard between the new pools and their nearest earlier pools (how many *theorems* they share):\n")
    W.append("| new pool | vs k12 | vs nearest earlier pool |")
    W.append("| --- | --- | --- |")
    for name in names:
        if name not in NEW:
            continue
        j = res["jaccard"][name]
        near = next((m for m, _ in res["nearest"][name] if m not in NEW), None)
        W.append(f"| {name} | {j.get('k12', float('nan')):.3f} | {near} {j.get(near, float('nan')):.3f} |" if near else f"| {name} | {j.get('k12', 0):.3f} | – |")
    W.append("")
    text = open(DOC).read()
    if HEAD in text:
        i = text.index(HEAD)
        j = text.find("\n## ", i + len(HEAD))
        text = text[:i].rstrip("\n") + "\n" + (text[j:] if j > 0 else "")
    with open(DOC, "w") as f:
        f.write(text.rstrip("\n") + "\n" + "\n".join(W) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="*", help="pool names (default: all)")
    ap.add_argument("--cut", type=float, default=0.10, help="cluster cut on the combined JS distance")
    ap.add_argument("--no_doc", action="store_true")
    ap.add_argument("--from_json", action="store_true", help="rewrite the doc section from the saved results, no recomputation")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.from_json:
        res = json.load(open(os.path.join(OUT, "pool_similarity.json")))
        res["nearest"] = {k: [tuple(x) for x in v] for k, v in res["nearest"].items()}
        write_doc(res, res["names"])
        print("wrote", DOC)
        return
    files = pool_files()
    names = a.pools or list(files)
    if "k12" not in names:
        names = ["k12"] + names
    feats = []
    floor = None
    for name in names:
        p = files[name]
        rows = PC.load_k12() if p is None else PC.read(p)
        feats.append(features(rows, name))
        if name == "k12":
            floor = noise_floor(rows)
            print("noise floor", floor, flush=True)
        del rows
    k = names.index("k12")
    D_seq = js_matrix([f["seqs"] for f in feats])
    D_big = js_matrix([f["bigrams"] for f in feats])
    D_shape = js_matrix([f["shapes"] for f in feats])
    D = (D_seq + D_big + D_shape) / 3
    vs_k12, nearest, jaccard = {}, {}, {}
    for i, f in enumerate(feats):
        vs_k12[f["label"]] = dict(
            n=f["n"], n_classes=len(f["classes"]),
            rec_in_k12=len(f["recs"] & feats[k]["recs"]) / max(1, len(f["recs"])),
            class_in_k12=len(f["classes"] & feats[k]["classes"]) / max(1, len(f["classes"])),
            js_seq=float(D_seq[i, k]), js_bigram=float(D_big[i, k]), js_shape=float(D_shape[i, k]), combined=float(D[i, k]))
        order = sorted((float(D[i, j]), names[j]) for j in range(len(names)) if j != i)
        nearest[f["label"]] = [(m, d) for d, m in order[:3]]
        jaccard[f["label"]] = {g["label"]: len(f["classes"] & g["classes"]) / max(1, len(f["classes"] | g["classes"]))
                               for g in feats if g is not f}
    merges = average_linkage(D, names)
    res = dict(names=names, vs_k12=vs_k12, nearest=nearest, jaccard=jaccard, cut=a.cut, floor=floor,
               clusters=flat_clusters(merges, names, a.cut),
               merges=[(list(l), list(r), round(d, 4)) for l, r, d in merges],
               combined=[[round(float(x), 4) for x in row] for row in D],
               js_seq=[[round(float(x), 4) for x in row] for row in D_seq],
               js_bigram=[[round(float(x), 4) for x in row] for row in D_big],
               js_shape=[[round(float(x), 4) for x in row] for row in D_shape])
    json.dump(res, open(os.path.join(OUT, "pool_similarity.json"), "w"), indent=1, ensure_ascii=False)
    for name in names:
        v = vs_k12[name]
        print(f"{name:26s} rec {v['rec_in_k12']:.3f} cls {v['class_in_k12']:.3f} seq {v['js_seq']:.3f} big {v['js_bigram']:.3f} "
              f"shape {v['js_shape']:.3f} comb {v['combined']:.3f}  nearest {nearest[name]}")
    print("clusters:", res["clusters"])
    if not a.no_doc:
        write_doc(res, names)
        print("wrote", DOC)


if __name__ == "__main__":
    main()
