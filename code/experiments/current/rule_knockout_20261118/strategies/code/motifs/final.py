# ruff: noqa
# Scratch analysis script written by a research agent on 2026-11-18; not reviewed.
import collections, pickle, json
from ndlib import *
from motifs import motifs

C = [
    "mp_chain2",
    "mp_curried",
    "mp_on_mp",
    "ande_then_mp_major",
    "assumption_cited_twice",
    "ore_on_derived",
    "ore_lemma",
    "refute_assumption_then_bote",
    "bote_in_box",
    "bote_closes_impi",
    "ore_any_work",
    "ore_b1_has_work",
    "ore_in_impi",
    "negi_in_impi",
    "mp_then_contradiction",
    "depth>=3",
    "raa_classical",
    "lemma_cited_twice",
    "disj_syllogism",
    "mp_chain3",
    "dilemma_mp_both",
    "mp_chain4",
    "ore_in_negi",
    "ore_in_ore",
    "ore_in_ore_b1",
    "ore_to_F_any",
    "depth>=4",
]
# pretraining: counts, overlap, rule share
n = 0
cnt = collections.Counter()
pair = collections.Counter()
rule = collections.Counter()
rulew = collections.defaultdict(collections.Counter)
for r in pretrain():
    ms = motifs(Proof(r["proof"]))
    n += 1
    hit = [k for k in C if k in ms]
    cnt.update(hit)
    rs = [x for x in ms if x.startswith("RULE_")]
    rule.update(rs)
    for k in hit:
        rulew[k].update(rs)
        for k2 in hit:
            pair[(k, k2)] += 1
# control: per target, has motif proof / has non-motif proof; round-1 counts; totals
tm = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
r1 = collections.Counter()
nr1 = 0
nctl = 0
cc = collections.Counter()
alltargets = set()
bandof = {}
for s in ("full-ei_s0", "full-ei_s1"):
    for r in jl(LIVE + s + "/found_stream.jsonl"):
        ms = motifs(Proof(r["proof"]))
        nctl += 1
        alltargets.add(r["name"])
        if r["round"] == 1:
            nr1 += 1
        for k in C:
            h = k in ms
            tm[k][r["name"]][0 if h else 1] += 1
            if h:
                cc[k] += 1
                if r["round"] == 1:
                    r1[k] += 1
# rl refs
rl = collections.Counter()
rl14 = collections.Counter()
nrl = nrl14 = 0
solved_ref = collections.Counter()
nsolved_ref = 0
for r in jl(D + "rl.jsonl"):
    if not r.get("proof"):
        continue
    ms = motifs(Proof(r["proof"]))
    nrl += 1
    hit = [k for k in C if k in ms]
    rl.update(hit)
    if r["body"] <= 14:
        nrl14 += 1
        rl14.update(hit)
    if r["name"] in alltargets:
        nsolved_ref += 1
        solved_ref.update(hit)
print(
    f"pretrain {n}; rl refs {nrl}; rl refs body<=14 {nrl14}; control proofs {nctl} on {len(alltargets)} distinct targets ({nsolved_ref} with a reference proof); control round-1 proofs {nr1}"
)
print(
    "motif\tpre_del\tpre_%\trule_share\trl_n\trl_%\trl14_n\trl14_%\tctl_proofs\tctl_%\tctl_targets\tctl_tgt_%\tavoidable_%\tr1_proofs\tsolved_ref_with"
)
for k in C:
    t = tm[k]
    withm = [x for x in t if t[x][0]]
    avoid = sum(1 for x in withm if t[x][1])
    rs = ",".join(
        f"{x[5:]} {100 * c / rule[x]:.0f}%"
        for x, c in rulew[k].most_common()
        if c / rule[x] >= 0.04 and x[5:] in ("IMPE", "ORE", "BOTE", "NEGI", "DN", "IMPI")
    )
    print(
        f"{k}\t{cnt[k]}\t{100 * cnt[k] / n:.2f}\t{rs}\t{rl[k]}\t{100 * rl[k] / nrl:.1f}\t{rl14[k]}\t{100 * rl14[k] / nrl14:.1f}\t{cc[k]}\t{100 * cc[k] / nctl:.1f}\t{len(withm)}\t{100 * len(withm) / len(alltargets):.1f}\t{100 * avoid / max(1, len(withm)):.0f}\t{r1[k]}\t{solved_ref[k]}"
    )
print("\noverlap of pretraining deletions (row & col):")
K = [
    "mp_chain2",
    "bote_in_box",
    "ore_any_work",
    "depth>=3",
    "mp_curried",
    "ore_in_impi",
    "negi_in_impi",
]
print("\t" + "\t".join(K))
for a in K:
    print(a + "\t" + "\t".join(str(pair[(a, b)]) for b in K))
