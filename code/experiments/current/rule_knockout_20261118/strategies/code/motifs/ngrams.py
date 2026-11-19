# ruff: noqa
# Scratch analysis script written by a research agent on 2026-11-18; not reviewed.
"""Exploratory: dependency-graph rule bigrams/trigrams + box nesting, per corpus. Output ngrams.tsv."""

import collections, sys, json
from ndlib import *


def roles(p, ln):
    r = ln.rule
    c = ln.c
    if r == "IMPE":
        return [(c[0], "IMPE.maj"), (c[1], "IMPE.min")]
    if r == "NEGE":
        return [(c[0], "NEGE.pos"), (c[1], "NEGE.neg")]
    if r in ("IMPI", "NEGI"):
        return [(c[1], r + ".last")]
    if r == "ORE":
        return [(c[0], "ORE.disj"), (c[2], "ORE.b1last"), (c[4], "ORE.b2last")]
    return [(x, r) for x in c]


def boxtype(p, b):
    k = b["closer"]
    if k is None:
        return "open"
    ln = p.L[k]
    if ln.rule == "ORE":
        return "ORE.b1" if ln.c[1] == b["start"] else "ORE.b2"
    return ln.rule


def tags(p):
    t = set()
    for i in p.order:
        ln = p.L[i]
        if not ln.live or ln.rule in ("PR", "AS"):
            continue
        for x, role in roles(p, ln):
            xl = p.L[x]
            t.add(f"2: {xl.rule} -> {role}")
            if xl.rule in ("PR", "AS"):
                continue
            for y, role0 in roles(p, xl):
                t.add(f"3: {p.L[y].rule} -> {role0} -> {role}")
    for b in p.boxes:
        if b["closer"] is None or not p.L[b["closer"]].live:
            continue
        bt = boxtype(p, b)
        chain = [bt]
        par = b["parent"]
        while par is not None:
            chain.append(boxtype(p, p.boxes[par]))
            par = p.boxes[par]["parent"]
        if len(chain) >= 2:
            t.add("nest2: " + " < ".join(chain[:2]))
        if len(chain) >= 3:
            t.add("nest3: " + " < ".join(chain[:3]))
    return t


def corpus(it):
    c = collections.Counter()
    n = 0
    for pr in it:
        n += 1
        for x in tags(Proof(pr)):
            c[x] += 1
    return c, n


if __name__ == "__main__":
    pre, npre = corpus(r["proof"] for r in pretrain())
    rl, nrl = corpus(r["proof"] for r in jl(D + "rl.jsonl") if r.get("proof"))
    ctl, nctl = corpus(
        r["proof"]
        for s in ("full-ei_s0", "full-ei_s1")
        for r in jl(LIVE + s + "/found_stream.jsonl")
    )
    keys = set(pre) | set(rl) | set(ctl)
    rows = sorted(keys, key=lambda k: -(rl[k] / nrl + ctl[k] / nctl))
    with open("ngrams.tsv", "w") as f:
        f.write(f"tag\tpre_n\tpre_%\trl_%\tctl_%\n")
        for k in rows:
            f.write(
                f"{k}\t{pre[k]}\t{100 * pre[k] / npre:.2f}\t{100 * rl[k] / nrl:.1f}\t{100 * ctl[k] / nctl:.1f}\n"
            )
    print(npre, nrl, nctl, len(keys))
