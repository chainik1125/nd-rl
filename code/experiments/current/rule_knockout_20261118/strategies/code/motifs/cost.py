# ruff: noqa
# Scratch analysis script written by a research agent on 2026-11-18; not reviewed.
"""First-hit cost proxy: for knockout K, the number of live lines in a motif-containing proof whose local signature
(rule, source rule per role, innermost box type, depth) never occurs in the K-filtered pretraining set, minus the
number that never occur in the full pretraining set ("extra novel lines")."""

import collections, pickle, statistics, sys
from ndlib import *
from motifs import motifs, roles, boxtype

CANDS = [
    "mp_chain2",
    "mp_curried",
    "mp_on_mp",
    "ande_then_mp_major",
    "assumption_cited_twice",
    "lemma_cited_twice",
    "refute_assumption_then_bote",
    "bote_in_box",
    "bote_closes_impi",
    "ore_on_derived",
    "ore_lemma",
    "ore_any_work",
    "ore_b1_has_work",
    "ore_both_work",
    "ore_branch_ends_mp",
    "ore_in_impi",
    "negi_in_impi",
    "mp_then_contradiction",
    "depth>=3",
    "impi_x3",
    "raa_classical",
    "lemma_box",
    "depth>=2",
    "impi_in_impi",
    "andi_then_mp",
    "negi_vacuous",
    "impi_vacuous",
    "disj_syllogism",
    "dilemma_mp_both",
    "mp_chain3",
    "ore_in_ore",
]


def sigs(p):
    """per live non-premise line: a pair of signatures. A = (rule, source rule per role), context-free.
    B = (rule, innermost box type, depth), source-free. A line is novel w.r.t. a corpus if either is unseen there."""
    out = []
    for i in p.order:
        ln = p.L[i]
        if not ln.live or ln.rule == "PR":
            continue
        ctx = boxtype(p, p.boxes[ln.box]) if ln.box is not None else "top"
        r = ln.rule.rstrip("12")
        if ln.rule == "AS":
            par = p.boxes[ln.box]["parent"]
            a = ("A", "AS", ctx)
            b = (
                "B",
                "AS",
                ctx,
                boxtype(p, p.boxes[par]) if par is not None else "top",
                min(ln.d, 4),
            )
        else:
            a = ("A", r, tuple((role, p.L[x].rule.rstrip("12")) for x, role in roles(ln)))
            b = ("B", r, ctx, min(ln.d, 4))
        out.append((a, b))
    return out


if __name__ == "__main__":
    tot = collections.Counter()
    withk = {k: collections.Counter() for k in CANDS}
    nk = collections.Counter()
    rule_recs = collections.Counter()
    rule_with = collections.defaultdict(collections.Counter)
    for r in pretrain():
        p = Proof(r["proof"])
        ms = motifs(p)
        ss = {x for ab in sigs(p) for x in ab}
        tot.update(ss)
        rules = {x for x in ms if x.startswith("RULE_")}
        rule_recs.update(rules)
        for k in CANDS:
            if k in ms:
                withk[k].update(ss)
                nk[k] += 1
                rule_with[k].update(rules)
    full = set(tot)
    filt = {k: {s for s in full if tot[s] > withk[k][s]} for k in CANDS}
    res = {k: {"ctl": [], "rl14": [], "rl": []} for k in CANDS}
    exs = {}

    def feed(corp, name, proof):
        p = Proof(proof)
        ms = motifs(p)
        ss = sigs(p)
        base = sum((a not in full or b not in full) for a, b in ss)
        for k in CANDS:
            if k in ms:
                extra = sum((a not in filt[k] or b not in filt[k]) for a, b in ss) - base
                res[k][corp].append(extra)
                if corp == "ctl":
                    e = exs.get(k)
                    if e is None or (extra, len(ss)) < e[:2]:
                        exs[k] = (
                            extra,
                            len(ss),
                            name,
                            proof,
                            sorted({x for ab in ss for x in ab if x not in filt[k] and x in full}),
                        )

    for s in ("full-ei_s0", "full-ei_s1"):
        for r in jl(LIVE + s + "/found_stream.jsonl"):
            feed("ctl", r["name"], r["proof"])
    for r in jl(D + "rl.jsonl"):
        if r.get("proof"):
            feed("rl", r["name"], r["proof"])
            if r["body"] <= 14:
                feed("rl14", r["name"], r["proof"])
    print(
        f"{'knockout':30s} {'del':>6s} {'sigs_lost':>9s}  extra novel lines: ctl min/med/mean   rl14 min/med   | share of each rule's pretraining records deleted"
    )
    for k in CANDS:

        def st(v):
            return f"{min(v)}/{statistics.median(v):.0f}/{statistics.mean(v):.1f}" if v else "-"

        share = ", ".join(
            f"{x[5:]} {100 * c / rule_recs[x]:.0f}%"
            for x, c in rule_with[k].most_common()
            if c / rule_recs[x] >= 0.1
        )
        print(
            f"{k:30s} {nk[k]:6d} {len(full) - len(filt[k]):9d}  {st(res[k]['ctl']):>16s} {st(res[k]['rl14']):>14s}   | {share}"
        )
    pickle.dump({"res": res, "exs": exs, "nk": dict(nk)}, open("cost.pkl", "wb"))
    print()
    for k in CANDS:
        if k in exs:
            e = exs[k]
            p = Proof(e[3])
            print(
                f"## {k}: cheapest control proof, extra novel={e[0]} ({e[2]})\n   "
                + " ; ".join(p.L[i].txt for i in p.order if p.L[i].live)
                + f"\n   lost sigs: {e[4]}"
            )
