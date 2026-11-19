# ruff: noqa
# Scratch analysis script written by a research agent on 2026-11-18; not reviewed.
"""Motif detectors over parsed ND proofs. motifs(p) -> set of motif names. Only live lines (backward-reachable
from the final line) count, so padding in model-written proofs does not trigger a motif."""

from ndlib import *

ELIM_ROLES = {"IMPE.maj", "IMPE.min", "NEGE.pos", "NEGE.neg", "ANDE", "DN", "ORE.disj"}


def roles(ln):
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
    if r in ("ANDE1", "ANDE2"):
        return [(c[0], "ANDE")]
    return [(x, r) for x in c]


def boxtype(p, b):
    k = b["closer"]
    if k is None:
        return "open"
    ln = p.L[k]
    if ln.rule == "ORE":
        return "ORE.b1" if ln.c[1] == b["start"] else "ORE.b2"
    return ln.rule


def reach(p, s):
    """lines that transitively depend on line s"""
    seen = set()
    st = [s]
    while st:
        i = st.pop()
        for u in p.L[i].users:
            if u not in seen:
                seen.add(u)
                st.append(u)
    return seen


def motifs(p):
    L = p.L
    m = set()
    live = [L[i] for i in p.order if L[i].live]
    used_as = {}  # role uses of each line: line -> set of roles
    for ln in live:
        if ln.rule in ("PR", "AS"):
            continue
        for x, role in roles(ln):
            used_as.setdefault(x, set()).add(role)
    maxd = max((ln.d for ln in live), default=0)
    for k in (2, 3, 4):
        if maxd >= k:
            m.add(f"depth>={k}")
    chain = {}  # longest minor-premise IMPE chain ending at line
    for ln in live:
        r = ln.rule
        if r in ("PR", "AS"):
            continue
        a = [L[c] for c in ln.c]
        if r == "IMPE":
            maj, mn = a
            chain[ln.i] = 1 + (chain.get(mn.i, 0) if mn.rule == "IMPE" else 0)
            if maj.rule == "IMPE":
                m.add("mp_curried")
            if maj.rule == "IMPE" or mn.rule == "IMPE":
                m.add("mp_on_mp")
            if maj.rule in ("ANDE1", "ANDE2"):
                m.add("ande_then_mp_major")
            if maj.rule == "AS":
                m.add("mp_major_is_assumption")
            if maj.rule in ("IMPI", "NEGI", "ORE", "DN"):
                m.add("mp_major_other_derived")
            if mn.rule == "IMPE":
                m.add("mp_chain2")
                if chain[ln.i] >= 3:
                    m.add("mp_chain3")
                if chain[ln.i] >= 4:
                    m.add("mp_chain4")
            if mn.rule in ("ANDE1", "ANDE2"):
                m.add("ande_then_mp_minor")
            if mn.rule == "ANDI":
                m.add("andi_then_mp")
            if mn.rule in ("ORI1", "ORI2"):
                m.add("ori_then_mp")
            if mn.rule == "ORE":
                m.add("ore_then_mp")
            if mn.rule in ("IMPI", "NEGI"):
                m.add("box_then_mp_minor")
        elif r == "NEGE":
            pos, neg = a
            if pos.rule == "IMPE" or neg.rule == "IMPE":
                m.add("mp_then_contradiction")
            if neg.rule == "IMPE":
                m.add("mp_derives_negation_used")
            if pos.rule in ("ORI1", "ORI2"):
                m.add("ori_then_contradiction")
            if pos.rule == "ANDI":
                m.add("andi_then_contradiction")
            if pos.rule in ("IMPI", "NEGI") or neg.rule == "NEGI":
                m.add("box_then_contradiction")
        elif r == "DN":
            if a[0].rule == "NEGI":
                m.add("raa_classical")  # assume ~A ... F ; ~~A ; A
            else:
                m.add("dn_on_given")
        elif r == "BOTE":
            m.add("bote_any")
            if ln.d >= 1:
                m.add("bote_in_box")
            src = a[0]
            if src.rule == "NEGE" and (L[src.c[0]].rule == "AS" or L[src.c[1]].rule == "AS"):
                m.add("refute_assumption_then_bote")
        elif r in ("IMPI", "NEGI"):
            b = p.bystart[ln.c[0]]
            s, e = a[0], a[1]
            dep = reach(p, s.i)
            if e.i != s.i and e.i not in dep:
                m.add("impi_vacuous" if r == "IMPI" else "negi_vacuous")
            if e.i == s.i:
                m.add("impi_identity" if r == "IMPI" else "negi_of_F")
            if r == "IMPI":
                if e.rule == "NEGI":
                    m.add("impi_ends_in_negi")
                if e.rule == "BOTE":
                    m.add("bote_closes_impi")
                if e.rule == "ORE":
                    m.add("ore_closes_impi")
                if e.rule == "IMPE":
                    mn = L[e.c[1]]
                    if mn.rule == "IMPE" and L[mn.c[1]].i == s.i:
                        m.add("hyp_syllogism")  # A>B, B>C |- A>C core
                    if chain.get(e.i, 0) >= 2 and e.i in dep:
                        m.add("impi_over_mp_chain")
            else:
                if e.rule == "NEGE":
                    pos, neg = L[e.c[0]], L[e.c[1]]
                    if pos.rule == "IMPE" and pos.c[1] == s.i and not p.in_box(neg.i, b):
                        m.add("modus_tollens")
                    if pos.rule == "IMPE" and pos.i in dep:
                        m.add("negi_via_mp")
                    if pos.rule in ("ORI1", "ORI2") and pos.c[0] == s.i:
                        m.add("demorgan_ori")  # ~(AvB) |- ~A
                    if pos.rule == "ANDI" and s.i in pos.c:
                        m.add("demorgan_andi")  # ~(A&B), A |- ~B
                    if pos.i == s.i and neg.rule in ("PR",):
                        m.add("negi_direct_clash")
                    if neg.i == s.i:
                        m.add("negi_assume_neg")
                if e.rule == "ORE":
                    m.add("ore_to_F")
            roles_used = used_as.get(ln.i, set())
            if roles_used & ELIM_ROLES:
                m.add("lemma_box")
                m.add("lemma_box_" + r.lower())
        elif r == "ORE":
            dj = a[0]
            b1 = p.bystart[ln.c[1]]
            b2 = p.bystart[ln.c[3]]
            e1, e2 = a[2], a[4]
            if dj.rule not in ("PR", "AS"):
                m.add("ore_on_derived")
            if dj.rule == "IMPE":
                m.add("ore_on_mp_result")
            if dj.rule in ("ANDE1", "ANDE2"):
                m.add("ore_on_ande_result")
            if dj.rule == "AS":
                m.add("ore_on_assumption")
            w1 = e1.i != b1["start"]
            w2 = e2.i != b2["start"]
            if w1:
                m.add("ore_b1_has_work")
            if w2:
                m.add("ore_b2_has_work")
            if w1 and w2:
                m.add("ore_both_work")
            if w1 or w2:
                m.add("ore_any_work")
            if e1.rule == "IMPE" and e2.rule == "IMPE":
                m.add("dilemma_mp_both")
            if e1.rule == "IMPE" or e2.rule == "IMPE":
                m.add("ore_branch_ends_mp")
            if e1.rule == "IMPE":
                m.add("ore_b1_ends_mp")
            for e, nm in ((e1, "b1"), (e2, "b2")):
                if e.rule == "BOTE":
                    m.add("ore_branch_bote")
                    m.add(f"ore_{nm}_bote")
                    if L[e.c[0]].rule == "NEGE":
                        m.add("disj_syllogism")
                if e.rule in ("ORI1", "ORI2"):
                    m.add("ore_branch_ends_ori")
                if e.rule == "ORE":
                    m.add("ore_nested_as_last")
            if used_as.get(ln.i, set()) & ELIM_ROLES:
                m.add("ore_lemma")
            if ln.f == "F":
                m.add("ore_to_F_any")
            if ln.i != p.last and ln.d == 0:
                m.add("ore_top_not_goal")
    # nesting
    for b in p.boxes:
        k = b["closer"]
        if k is None or not L[k].live:
            continue
        bt = boxtype(p, b)
        par = b["parent"]
        if par is None:
            continue
        pt = boxtype(p, p.boxes[par])
        if bt.startswith("ORE") and pt == "ORE.b1":
            m.add("ore_in_ore_b1")
        if bt.startswith("ORE") and pt == "ORE.b2":
            m.add("ore_in_ore_b2")
        if bt.startswith("ORE") and pt.startswith("ORE"):
            m.add("ore_in_ore")
        if bt.startswith("ORE") and pt == "IMPI":
            m.add("ore_in_impi")
        if bt.startswith("ORE") and pt == "NEGI":
            m.add("ore_in_negi")
        if bt == "IMPI" and pt.startswith("ORE"):
            m.add("impi_in_ore")
        if bt == "NEGI" and pt.startswith("ORE"):
            m.add("negi_in_ore")
        if bt == "NEGI" and pt == "IMPI":
            m.add("negi_in_impi")
        if bt == "IMPI" and pt == "NEGI":
            m.add("impi_in_negi")
        if bt == "NEGI" and pt == "NEGI":
            m.add("negi_in_negi")
        if bt == "IMPI" and pt == "IMPI":
            m.add("impi_in_impi")
            gp = p.boxes[par]["parent"]
            if gp is not None and boxtype(p, p.boxes[gp]) == "IMPI":
                m.add("impi_x3")
    # reuse
    for ln in live:
        nu = len({u for u in ln.users if L[u].live})
        if ln.rule not in ("PR", "AS") and nu >= 2:
            m.add("lemma_cited_twice")
        if ln.rule == "AS":
            closer = p.bystart[ln.i]["closer"]
            if len({u for u in ln.users if L[u].live and u != closer}) >= 2:
                m.add("assumption_cited_twice")
    # rules (for "rule in disguise" checks)
    for ln in live:
        if ln.rule not in ("PR", "AS"):
            m.add("RULE_" + ln.rule.rstrip("12"))
    # goal connective x final rule
    m.add(f"GOAL_{conn(p.goal)}__LAST_{L[p.last].rule.rstrip('12')}")
    return m
