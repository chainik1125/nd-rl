# ruff: noqa
# Scratch analysis script written by a research agent on 2026-11-18; not reviewed.
import collections, sys
from ndlib import *

# argument-order conventions and parse sanity
c = collections.Counter()
bad = collections.Counter()


def chk(p):
    for i in p.order:
        ln = p.L[i]
        r = ln.rule
        try:
            a = [p.L[x] for x in ln.c]
            if r == "IMPE":
                c[
                    "IMPE major-first"
                    if a[0].f == (">", a[1].f, ln.f)
                    else ("IMPE minor-first" if a[1].f == (">", a[0].f, ln.f) else "IMPE ??")
                ] += 1
            elif r == "NEGE":
                c[
                    "NEGE pos-first"
                    if a[1].f == ("~", a[0].f)
                    else ("NEGE neg-first" if a[0].f == ("~", a[1].f) else "NEGE ??")
                ] += 1
            elif r == "NEGI":
                c[
                    "NEGI ok"
                    if (a[0].rule == "AS" and a[1].f == "F" and ln.f == ("~", a[0].f))
                    else "NEGI ??"
                ] += 1
            elif r == "IMPI":
                c[
                    "IMPI ok"
                    if (a[0].rule == "AS" and ln.f == (">", a[0].f, a[1].f))
                    else "IMPI ??"
                ] += 1
            elif r == "ORE":
                ok = len(a) == 5 and a[0].f == ("v", a[1].f, a[3].f) and a[2].f == ln.f == a[4].f
                c["ORE ok" if ok else "ORE ??"] += 1
            elif r == "DN":
                c["DN ok" if a[0].f == ("~", ("~", ln.f)) else "DN ??"] += 1
            elif r == "BOTE":
                c["BOTE ok" if a[0].f == "F" else "BOTE ??"] += 1
            elif r == "ANDI":
                c["ANDI ok" if ln.f == ("&", a[0].f, a[1].f) else "ANDI ??"] += 1
            for b in p.box_of_closer(i):
                pass
        except Exception as e:
            bad[r + " " + type(e).__name__] += 1


n = 0
for r in pretrain():
    p = Proof(r["proof"])
    chk(p)
    n += 1
    assert p.body == r["body"], (p.body, r["body"])
print("pretrain", n, c, bad)
c.clear()
bad.clear()
nref = 0
for r in jl(D + "rl.jsonl"):
    if r.get("proof"):
        p = Proof(r["proof"])
        chk(p)
        nref += 1
        if p.body != r["body"]:
            bad["bodymismatch"] += 1
print("rl", nref, c, bad)
c.clear()
bad.clear()
k = 0
dead = 0
wp = collections.Counter()
for s in ("full-ei_s0", "full-ei_s1"):
    for r in jl(LIVE + s + "/found_stream.jsonl"):
        p = Proof(r["proof"])
        chk(p)
        k += 1
        dead += any(not p.L[i].live for i in p.order if p.L[i].rule != "PR")
        wp[(r["written"] == r["pruned"], p.body == r["written"], p.body == r["pruned"])] += 1
print("control", k, c, bad, "with dead body lines", dead, wp)
