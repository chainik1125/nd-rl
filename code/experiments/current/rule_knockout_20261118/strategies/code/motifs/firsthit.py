# ruff: noqa
# Scratch analysis script written by a research agent on 2026-11-18; not reviewed.
import collections, json
from ndlib import *
from motifs import motifs

C = [
    "mp_chain2",
    "bote_in_box",
    "depth>=3",
    "ore_any_work",
    "ore_b1_has_work",
    "mp_curried",
    "ore_in_impi",
    "negi_in_impi",
    "ore_on_derived",
    "disj_syllogism",
    "mp_chain3",
    "dilemma_mp_both",
    "mp_chain4",
    "ore_in_negi",
    "ore_in_ore",
    "ore_in_ore_b1",
    "ore_to_F_any",
    "depth>=4",
    "negi_in_negi",
]
pool = 18439
ref = collections.Counter()
ref14 = collections.Counter()
for r in jl(D + "rl.jsonl"):
    if r.get("proof"):
        ms = motifs(Proof(r["proof"]))
        for k in C:
            if k in ms:
                ref[k] += 1
                if r["body"] <= 14:
                    ref14[k] += 1
for s in ("full-ei_s0", "full-ei_s1"):
    cum = {}
    tot = 0
    mins = {}
    for r in jl(LIVE + s + "/rounds.jsonl"):
        cum[r["round"]] = tot
        tot += r["sampling"]["samples"]
        mins[r["round"]] = r["rl_secs_used"] / 60
    last = max(cum)
    first = {}
    by = collections.defaultdict(collections.Counter)
    n = 0
    for r in jl(LIVE + s + "/found_stream.jsonl"):
        ms = motifs(Proof(r["proof"]))
        n += 1
        for k in C:
            if k in ms:
                first.setdefault(k, (r["round"], r["name"], r["body"]))
                by[k][r["round"]] += 1
    print(f"\n{s}: {last} rounds, {tot:,} samples, {mins[last]:.0f} RL min, {n} accepted proofs")
    print(
        "motif\tref<=14\tfirst round\tsamples before that round\tRL min\t~attempts on reachable ref targets before\tfirst target(body)\tproofs by r10/r20/r40/now"
    )
    for k in C:
        if k in first:
            rd, nm, bd = first[k]
            cumc = lambda R: sum(v for q, v in by[k].items() if q <= R)
            print(
                f"{k}\t{ref14[k]}\t{rd}\t{cum[rd]:,}\t{mins[rd]:.0f}\t{cum[rd] * ref14[k] / pool:,.0f}\t{nm}({bd})\t{cumc(10)}/{cumc(20)}/{cumc(40)}/{cumc(10**9)}"
            )
        else:
            print(f"{k}\t{ref14[k]}\tnever\t>{tot:,}\t\t>{tot * ref14[k] / pool:,.0f}\t\t0")
