# ruff: noqa
# Scratch analysis script written by a research agent on 2026-11-18; not reviewed.
"""Proof-theoretic flags on Fitch proofs in the nd-rl text format (read-only analysis, scratch).

A proof is 'N1 <bars> <formula> : RULE refs ; ... QED'.  flags(prompt, proof) returns a dict of binary flags / counts.
Ref conventions (from long_rl_20261014/overlay/lean_tok.py): IMPE [imp, arg]; NEGE [pos, neg]; ANDI [a, b];
ANDE1/2 [a]; ORI1/2 [a]; BOTE [a]; DN [a]; R [a]; IMPI/NEGI [as, end]; ORE [disj, as1, end1, as2, end2].
"""

import re

INTRO = {"IMPI", "NEGI", "ANDI", "ORI1", "ORI2"}
BOXI = {"IMPI", "NEGI"}


def parse_formula(toks, i=0):
    t = toks[i]
    if t == "(":
        if toks[i + 1] == "~":
            a, j = parse_formula(toks, i + 2)
            assert toks[j] == ")"
            return ("not", a), j + 1
        a, j = parse_formula(toks, i + 1)
        op = {"&": "and", "v": "or", ">": "imp"}[toks[j]]
        b, k = parse_formula(toks, j + 1)
        assert toks[k] == ")"
        return (op, a, b), k + 1
    if t == "F":
        return ("bot",), i + 1
    return ("atom", t), i + 1


def parse_proof(proof):
    lines = {}
    order = []
    for seg in proof.split(" ; "):
        seg = seg.strip()
        if not seg or seg == "QED":
            continue
        head, tail = seg.rsplit(" : ", 1)
        ht = head.split()
        idx = int(ht[0][1:])
        d = 0
        while ht[1 + d] == "|":
            d += 1
        f, used = parse_formula(ht[1 + d :], 0)
        assert used == len(ht) - 1 - d
        tt = tail.replace(";", "").split()
        rule = tt[0]
        refs = [int(x[1:]) for x in tt[1:] if x != "QED"]
        lines[idx] = dict(idx=idx, depth=d, f=f, rule=rule, refs=refs)
        order.append(idx)
    return lines, order


def parse_prompt(prompt):
    body = prompt[len("THM") :].rsplit("PRF", 1)[0]
    prem_s, concl_s = body.split("SEQ")
    toks = prem_s.split()
    prem = []
    i = 0
    while i < len(toks):
        if toks[i] == ",":
            i += 1
            continue
        f, i = parse_formula(toks, i)
        prem.append(f)
    concl, _ = parse_formula(concl_s.split(), 0)
    return prem, concl


def subformulas(f, acc):
    acc.add(f)
    for g in f[1:]:
        if isinstance(g, tuple):
            subformulas(g, acc)
    return acc


def deps(ln):
    """lines a line's derivation consumes (box rules consume the box end line, not the assumption)"""
    r, refs = ln["rule"], ln["refs"]
    if r in BOXI:
        return [refs[1]]
    if r == "ORE":
        return [refs[0], refs[2], refs[4]]
    return list(refs)


def flags(prompt, proof):
    L, order = parse_proof(proof)
    final = order[-1]
    # live cone of the final line
    live = set()
    stack = [final]
    while stack:
        x = stack.pop()
        if x in live:
            continue
        live.add(x)
        stack.extend(deps(L[x]))
    cone_cache = {}

    def cone(x):
        if x not in cone_cache:
            s = set()
            st = [x]
            while st:
                y = st.pop()
                if y in s:
                    continue
                s.add(y)
                st.extend(deps(L[y]))
            cone_cache[x] = s
        return cone_cache[x]

    def origin(x):
        while L[x]["rule"] == "R":
            x = L[x]["refs"][0]
        return x

    out = dict(
        n_body=sum(1 for i in order if L[i]["rule"] != "PR"),
        dead=0,
        detour=0,
        detour_imp=0,
        detour_neg=0,
        detour_and=0,
        detour_or=0,
        perm=0,
        dn=0,
        raa=0,
        dne_proper=0,
        raa_k0=0,
        raa_k1=0,
        raa_k2=0,
        bote=0,
        vacuous=0,
        vac_ore=0,
        contr_hyp=0,
        contr_as=0,
        contr_pr=0,
        share=0,
        argbox=0,
        nonanalytic=0,
        weak_only=0,
        max_depth=max(L[i]["depth"] for i in order),
        reit=0,
        nonatomic_dn=0,
        nonatomic_bote=0,
        bote_then_elim=0,
        dn_then_elim=0,
    )
    out["dead"] = sum(1 for i in order if i not in live and L[i]["rule"] not in ("PR", "AS"))
    # citation counts within the live cone
    cites = {i: 0 for i in order}
    consumers = {i: [] for i in order}  # (consumer line, role)
    for i in order:
        if i not in live:
            continue
        ln = L[i]
        r, refs = ln["rule"], ln["refs"]
        if r == "R":
            out["reit"] += 1
        roles = []
        if r == "IMPE":
            roles = [(refs[0], "major"), (refs[1], "minor")]
        elif r == "NEGE":
            roles = [(refs[1], "major"), (refs[0], "minor")]
        elif r in ("ANDE1", "ANDE2"):
            roles = [(refs[0], "major")]
        elif r == "ORE":
            roles = [(refs[0], "major"), (refs[2], "pass"), (refs[4], "pass")]
        elif r in ("ANDI", "ORI1", "ORI2"):
            roles = [(x, "intro") for x in refs]
        elif r in BOXI:
            roles = [(refs[1], "pass")]
        elif r == "R":
            roles = [(refs[0], "pass")]
        elif r == "BOTE":
            roles = [(refs[0], "major")]
        elif r == "DN":
            roles = [(refs[0], "dn")]
        for x, role in roles:
            cites[x] += 1
            consumers[x].append((i, role))
    # detours / permutations: intro-derived (or DN/BOTE/ORE-derived) line as major premise of an elimination
    match = {
        "IMPE": ("IMPI",),
        "NEGE": ("NEGI",),
        "ANDE1": ("ANDI",),
        "ANDE2": ("ANDI",),
        "ORE": ("ORI1", "ORI2"),
    }
    for i in order:
        if i not in live:
            continue
        ln = L[i]
        r, refs = ln["rule"], ln["refs"]
        if r in match:
            mj = origin(refs[1] if r == "NEGE" else refs[0])
            mr = L[mj]["rule"]
            if mr in match[r]:
                out["detour"] += 1
                out[
                    "detour_"
                    + {"IMPE": "imp", "NEGE": "neg", "ANDE1": "and", "ANDE2": "and", "ORE": "or"}[r]
                ] += 1
            elif mr == "ORE":
                out["perm"] += 1
            elif mr == "BOTE":
                out["bote_then_elim"] += 1
            elif mr == "DN":
                out["dn_then_elim"] += 1
        if r == "BOTE":
            out["bote"] += 1
            if ln["f"][0] != "atom":
                out["nonatomic_bote"] += 1
        if r == "DN":
            out["dn"] += 1
            if ln["f"][0] != "atom":
                out["nonatomic_dn"] += 1
            src = origin(refs[0])
            if L[src]["rule"] == "NEGI":
                out["raa"] += 1
                s = L[src]["refs"][0]
                k = cites[s]
                out["raa_k0" if k == 0 else "raa_k1" if k == 1 else "raa_k2"] += 1
            else:
                out["dne_proper"] += 1
        if r in BOXI:
            s, e = refs
            if s not in cone(e):
                out["vacuous"] += 1
        if r == "ORE":
            if refs[1] not in cone(refs[2]) or refs[3] not in cone(refs[4]):
                out["vac_ore"] += 1
    for i in order:
        if i not in live:
            continue
        r = L[i]["rule"]
        if cites[i] >= 2:
            if r == "AS":
                out["contr_as"] += 1
                out["contr_hyp"] += 1
            elif r == "PR":
                out["contr_pr"] += 1
                out["contr_hyp"] += 1
            else:
                out["share"] += 1

    # argument boxes: an IMPI/NEGI line that reaches (through intros / R / box ends / ORE passes) a minor premise
    def reaches_minor(x, seen):
        for c, role in consumers[x]:
            if role == "minor":
                return True
            if role in ("intro", "pass") and c not in seen:
                seen.add(c)
                if L[c]["rule"] in BOXI:
                    continue  # became the body of another box: that box is judged on its own
                if reaches_minor(c, seen):
                    return True
        return False

    for i in order:
        if i in live and L[i]["rule"] in BOXI and reaches_minor(i, set()):
            out["argbox"] += 1
    # subformula property
    prem, concl = parse_prompt(prompt)
    S = set()
    for f in prem + [concl]:
        subformulas(f, S)
    S.add(("bot",))
    for i in order:
        if i not in live:
            continue
        f = L[i]["f"]
        if f in S:
            continue
        g = f
        n = 0
        while g[0] == "not" and g not in S:
            g = g[1]
            n += 1
        if g in S and n <= 2:
            out["weak_only"] += 1
        else:
            out["nonanalytic"] += 1
    out["rules"] = sorted({L[i]["rule"] for i in order if i in live})
    out["n_live_body"] = sum(1 for i in live if L[i]["rule"] != "PR")
    return out


if __name__ == "__main__":
    import sys, json, gzip, collections

    path = sys.argv[1]
    op = gzip.open if path.endswith(".gz") else open
    keys = [
        "dead",
        "detour",
        "detour_imp",
        "detour_neg",
        "detour_and",
        "detour_or",
        "perm",
        "bote_then_elim",
        "dn_then_elim",
        "dn",
        "raa",
        "dne_proper",
        "raa_k0",
        "raa_k1",
        "raa_k2",
        "nonatomic_dn",
        "bote",
        "nonatomic_bote",
        "vacuous",
        "vac_ore",
        "contr_hyp",
        "contr_as",
        "contr_pr",
        "share",
        "argbox",
        "nonanalytic",
        "weak_only",
        "reit",
    ]
    cnt = collections.Counter()
    n = 0
    bad = 0
    prompts = {}
    if len(sys.argv) > 2:  # found_stream has no prompt: take it from rl.jsonl by name
        for l in open(sys.argv[2]):
            r = json.loads(l)
            prompts[r["name"]] = r["prompt"]
    depth = collections.Counter()
    lin = collections.Counter()
    for l in op(path, "rt"):
        r = json.loads(l)
        if not r.get("proof"):
            continue
        pr = r.get("prompt") or prompts.get(r["name"])
        try:
            fl = flags(pr, r["proof"])
        except Exception as e:
            bad += 1
            if bad < 4:
                print("ERR", repr(e), r["proof"][:200])
            continue
        n += 1
        for k in keys:
            if fl[k]:
                cnt[k] += 1
        depth[fl["max_depth"]] += 1
        lin[
            ("K" if fl["vacuous"] or fl["vac_ore"] else "-") + ("W" if fl["contr_as"] else "-")
        ] += 1
    print(path, "n", n, "bad", bad)
    for k in keys:
        print(f"  {k:16s} {cnt[k]:7d}  {100 * cnt[k] / max(n, 1):6.2f}%")
    print("  max box depth", sorted(depth.items()))
    print(
        "  box-assumption structural class (K = vacuous discharge, W = assumption cited >=2):",
        sorted(lin.items()),
    )
