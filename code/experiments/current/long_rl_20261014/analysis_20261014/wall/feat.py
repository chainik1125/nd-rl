"""Structural features of an ND proof (gold or found) and theorem."""
import collections, re

def lines(proof):
    out = []
    for seg in proof.split(" ; "):
        seg = seg.strip()
        if not seg or seg == "QED":
            continue
        m = re.match(r"N(\d+) ((?:\| )*)(.*) : (\w+)((?: N\d+)*)$", seg)
        assert m, seg
        idx, bars, f, rule, refs = m.groups()
        out.append({"idx": int(idx), "depth": bars.count("|"), "f": f.strip(), "rule": rule,
                    "refs": [int(x[1:]) for x in refs.split()]})
    return out

def top(f):
    """main connective of a formula string"""
    f = f.strip()
    if not f.startswith("("):
        return "atom" if f != "F" else "F"
    inner = f[1:-1].strip()
    if inner.startswith("~"):
        return "~"
    d = 0
    toks = inner.split()
    for t in toks:
        if t == "(": d += 1
        elif t == ")": d -= 1
        elif d == 0 and t in (">", "&", "v", "<>", "="):
            return t
    return "?"

def feats(rec, proof=None):
    ls = lines(proof or rec["proof"])
    body = [l for l in ls if l["rule"] != "PR"]
    rules = collections.Counter(l["rule"] for l in body)
    by = {l["idx"]: l for l in ls}
    # needed closure
    need, todo = set(), [ls[-1]["idx"]]
    while todo:
        i = todo.pop()
        if i not in need:
            need.add(i); todo += by[i]["refs"]
    prem = [l for l in ls if l["rule"] == "PR"]
    used_prem = sum(1 for l in prem if l["idx"] in need)
    goal = ls[-1]["f"]
    d = {"body": len(body), "maxdepth": max((l["depth"] for l in ls), default=0), "n_prem": len(prem),
         "used_prem": used_prem, "goal": top(goal), "n_AS": rules["AS"]}
    for r in ("IMPI", "IMPE", "ANDI", "ANDE1", "ANDE2", "ORI1", "ORI2", "ORE", "NEGI", "NEGE", "R", "BOTE", "DN"):
        d[r] = rules[r]
    d["other_rules"] = sorted(k for k in rules if k not in d)
    d["has_F"] = any(l["f"] == "F" for l in body)
    # premise main connectives (used ones)
    d["prem_tops"] = collections.Counter(top(l["f"]) for l in prem if l["idx"] in need)
    return d
