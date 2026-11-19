# ruff: noqa
# Scratch analysis script written by a research agent on 2026-11-18; not reviewed.
#!/usr/bin/env python3
"""Quick, UNVALIDATED census of the group's existing strategy definitions on Robbie's files.
Simplified re-implementation of predicates from Dan's TAXONOMY.md v1.2 / patterns.py / patterns2.py and the
long-RL note. Not Dan's strategies.py. Read-only. Usage: quick_census.py <file.jsonl[.gz]> ..."""

import sys, gzip, json, re, collections

LINE = re.compile(r"^N(\d+) ((?:\| )*)(.*) : ([A-Z]+[12]?)((?: N\d+)*)$")


def parse(proof):
    out = []
    for raw in proof.split(" ; "):
        raw = raw.strip()
        if raw in ("QED", ""):
            continue
        m = LINE.match(raw)
        if not m:
            return None
        out.append(
            {
                "idx": int(m.group(1)),
                "depth": m.group(2).count("|"),
                "f": m.group(3).strip(),
                "rule": m.group(4),
                "refs": [int(x[1:]) for x in m.group(5).split()],
            }
        )
    return out


def top(f):
    """-> (op, left, right) of a fully parenthesised formula string, or None for atom/F."""
    t = f.split()
    if t[0] != "(":
        return None
    if t[1] == "~":
        return ("~", " ".join(t[2:-1]), None)
    d = 0
    for i in range(1, len(t) - 1):
        if t[i] == "(":
            d += 1
        elif t[i] == ")":
            d -= 1
        elif d == 0 and t[i] in ("&", "v", ">"):
            return (t[i], " ".join(t[1:i]), " ".join(t[i + 1 : -1]))
    return None


def census(proof):
    L = parse(proof)
    if not L:
        return None
    by = {l["idx"]: l for l in L}
    # prune: PR + everything the last line transitively cites
    keep = {l["idx"] for l in L if l["rule"] == "PR"}
    st = [L[-1]["idx"]]
    while st:
        i = st.pop()
        if i in keep or i not in by:
            continue
        keep.add(i)
        st.extend(by[i]["refs"])
    dead = len(L) - len(keep)
    L = [l for l in L if l["idx"] in keep]

    def origin(i):
        while by[i]["rule"] == "R":
            i = by[i]["refs"][0]
        return i

    hy = {}
    for l in L:
        r, refs, i = l["rule"], l["refs"], l["idx"]
        if r == "AS":
            h = {i}
        elif r in ("IMPI", "NEGI"):
            h = hy[refs[1]] - {refs[0]}
        elif r == "ORE":
            h = hy[refs[0]] | (hy[refs[2]] - {refs[1]}) | (hy[refs[4]] - {refs[3]})
        else:
            h = set().union(*[hy[x] for x in refs]) if refs else set()
        hy[i] = h
    citers = collections.defaultdict(set)
    for l in L:
        if l["rule"] == "R":
            continue
        for x in l["refs"]:
            citers[origin(x)].add(l["idx"])
    root = origin(L[-1]["idx"])
    tags = set()
    ores = [l for l in L if l["rule"] == "ORE"]
    uses = lambda s, e: s == e or s in hy[e]

    def refuted(s, e):
        if by[s]["f"] == "F":
            return False
        o = by[origin(e)]
        return o["rule"] == "BOTE" and s <= o["idx"] <= e and s in hy[o["refs"][0]]

    def route(n):  # refutation route of a NEGI line, TAXONOMY section 2
        s, e = n["refs"]
        if s != e and s not in hy[e]:
            return "vacuous"
        f = by[origin(e)]
        if not (s <= f["idx"] <= e):
            return "other"
        if f["rule"] == "NEGE":
            a, b = f["refs"]
            da, db = s in hy[a], s in hy[b]
            if da and db:
                return "self"
            side = a if da else b
        elif f["rule"] == "IMPE":
            m, k = f["refs"]
            if s in hy[k] and s not in hy[m]:
                side = k
            else:
                return "mixed"
        elif f["rule"] == "ORE":
            return "cases"
        else:
            return "other"
        steps = []
        x = origin(side)
        while x != s:
            l = by[x]
            r = l["rule"]
            if r == "IMPE":
                mj, mn = l["refs"]
                dj, dn = s in hy[mj], s in hy[mn]
                if dn and not dj:
                    steps.append("mp")
                    x = origin(mn)
                elif dj and not dn:
                    steps.append("mp_major")
                    x = origin(mj)
                else:
                    steps.append("mixed")
                    break
            elif r in ("ANDE1", "ANDE2"):
                steps.append("ande")
                x = origin(l["refs"][0])
            elif r in ("ORI1", "ORI2"):
                steps.append("ori")
                x = origin(l["refs"][0])
            elif r == "ANDI":
                d = [y for y in l["refs"] if s in hy[y]]
                if len(set(d)) == 1 and len(set(l["refs"])) == 2:
                    steps.append("andi")
                    x = origin(d[0])
                else:
                    steps.append("mixed")
                    break
            elif r == "DN":
                steps.append("dn")
                x = origin(l["refs"][0])
            else:
                steps.append("mixed")
                break
        if not steps:
            return "immediate"
        if "mixed" in steps:
            return "mixed"
        if steps[-1] == "mp_major":
            return "imp_hyp"
        ss = set(steps)
        if ss <= {"mp", "ande"} and "mp" in ss:
            return "mt"
        if ss == {"ande"}:
            return "and_elim"
        if ss == {"ori"}:
            return "or_weaken"
        if ss == {"andi"}:
            return "and_intro"
        return "mixed"

    chain = {}
    for l in L:
        r, refs, i = l["rule"], l["refs"], l["idx"]
        tags.add("rule." + r)
        if r == "IMPE":
            chain[i] = 1 + max([chain[origin(x)] for x in refs if origin(x) in chain], default=0)
        if r == "IMPI":
            s, e = refs
            tags.add("cp.working" if uses(s, e) else "cp.vacuous")
            o = by[origin(e)]
            if o["rule"] == "IMPI" and s <= o["idx"] <= e:
                tags.add("cp.curried")
            if o["rule"] == "BOTE" and s <= o["idx"] <= e and s in hy[o["refs"][0]]:
                tags.add("cp.explosive")
            if any(x["rule"] == "ORE" and s <= x["idx"] <= e for x in L):
                tags.add("impi_ore")
            if any(x["rule"] == "NEGI" and s <= x["idx"] <= e for x in L):
                tags.add("negi_inside_impi")
        if r == "ORE":
            j, s1, e1, s2, e2 = refs
            g = uses(s1, e1) and uses(s2, e2)
            tags.add("cases.genuine" if g else "cases.redundant")
            oj = by[origin(j)]
            tags.add(
                {"PR": "cases.on_premise", "AS": "cases.on_hypothesis"}.get(
                    oj["rule"], "cases.on_derived"
                )
            )
            t = top(oj["f"])
            merge = bool(t and t[0] == "v" and t[1] == t[2])
            if merge:
                tags.add("cases.trivial_merge")
            if oj["rule"] not in ("PR", "AS") and not merge:
                tags.add("cases.derived_strict")
            if oj["rule"] in ("ORI1", "ORI2"):
                tags.add("cases.ori_ore")
            if l["f"] == "F":
                tags.add("cases.goal_bot")
            r1, r2 = refuted(s1, e1), refuted(s2, e2)
            if r1 and r2:
                tags.add("cases.all_refuted")
            if r1 != r2:
                so, eo = (s2, e2) if r1 else (s1, e1)
                if uses(so, eo) and by[so]["f"] != "F":
                    tags.add("disj_syll")
            for o in ores:
                if o is l:
                    continue
                _, a1, b1, a2, b2 = o["refs"]
                if a1 <= i <= b1:
                    tags.add("cases.nested")
                    tags.add("cases.nested_first_branch")
                if a2 <= i <= b2:
                    tags.add("cases.nested")
                    tags.add("cases.nested_second_branch")
        if r == "NEGI":
            s, e = refs
            rt = route(l)
            c = citers[i]
            only_dn = bool(c) and all(by[y]["rule"] == "DN" for y in c)
            if i == root or not only_dn:
                tags.add("neg_intro")
                tags.add("neg_intro." + rt)
                tags.add("neg_intro.working" if rt != "vacuous" else "neg_intro.vacuous_")
                if i != root:
                    tags.add("neg_intro.nonroot")
            if rt == "mt":
                tags.add("mod_tollens")
            if any(
                x["rule"] in ("ANDE1", "ANDE2") and x["refs"][0] == s and s <= x["idx"] <= e
                for x in L
            ):
                tags.add("negi_ande_hyp")
            if any(x["rule"] == "ORE" and s <= x["idx"] <= e for x in L):
                tags.add("ore_inside_negi")
        if r == "BOTE":
            b = refs[0]
            tags.add("ex_falso.inconsistent_premises" if not hy[b] else "ex_falso.under_assumption")
        if r == "DN":
            c = by[origin(refs[0])]
            if c["rule"] == "NEGI":
                tags.add("sigma")
                rt = route(c)
                tags.add("sigma.route_" + rt)
            if c["rule"] not in ("PR", "AS"):
                tags.add("sigma_plus")
            elif c["rule"] == "PR":
                tags.add("dn.on_premise")
            else:
                tags.add("dn.on_hypothesis")
        if r not in ("PR", "AS", "R") and len(citers[i]) >= 2:
            tags.add("lemma_reuse")
    if chain:
        m = max(chain.values())
        if m >= 2:
            tags.add("mp_chain2+")
        if m >= 3:
            tags.add("mp_chain3+")
        if m >= 4:
            tags.add("mp_chain4+")
    md = max(l["depth"] for l in L)
    tags.add(f"depth{min(md, 4)}{'+' if md >= 4 else ''}")
    if not any(l["rule"] == "AS" for l in L):
        tags.add("box_free")
    if dead:
        tags.add("has_dead_lines")
    return tags


for fn in sys.argv[1:]:
    op = gzip.open if fn.endswith(".gz") else open
    n = nproof = bad = 0
    c = collections.Counter()
    for line in op(fn, "rt"):
        r = json.loads(line)
        n += 1
        if not r.get("proof"):
            continue
        t = census(r["proof"])
        if t is None:
            bad += 1
            continue
        nproof += 1
        c.update(t)
    print(f"## {fn}\nrecords {n}, with proof {nproof}, unparsed {bad}")
    for k in sorted(c):
        print(f"{k}\t{c[k]}\t{100 * c[k] / nproof:.3f}")
