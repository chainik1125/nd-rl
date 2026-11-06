"""Error localization on Lean-REJECTED samples (Lean is the judge; this only names the first suspicious line).
Categories: unbound/scope (LEANPARSE unbound), no-eos (rambling), other parse; for LEANREJ with decodable ND:
 fake_andi: ANDI citing the same line twice for a conjunction with distinct conjuncts
 bad_impe: IMPE whose major is not (minor > X) with X = line formula
 bad_nege: NEGE whose two refs are not {A, ~A}
 bad_bote/other."""
import json, sys, collections
sys.path.insert(0, ".")
from feat import parse_proof
def classify(s):
    nd = s["nd"]
    if s["ok"]: return "OK"
    if nd.startswith("LEANPARSE"):
        t = nd.split()[1] if len(nd.split()) > 1 else "?"
        return {"unbound": "scope/unbound name", "no-eos": "no-eos (ramble)"}.get(t, "parse-other")
    body = nd.split(" ", 1)[1] if nd.startswith("LEANREJ") else nd
    try:
        L = parse_proof(body)
    except Exception:
        return "rej-unparsed"
    by = {l["idx"]: l for l in L}
    for l in L:
        r, f = l["rule"], l["f"]
        refs = [by.get(i) for i in l["refs"]]
        if any(x is None for x in refs): return "rej-dangling-ref"
        if r == "ANDI":
            if l["refs"][0] == l["refs"][1] and f[1] != f[2]: return "fake ANDI (same line twice)"
            if (refs[0]["f"], refs[1]["f"]) != (f[1], f[2]): return "ANDI mismatch"
        if r == "IMPE":
            maj, mn = refs
            if not (isinstance(maj["f"], tuple) and maj["f"][0] == ">" and maj["f"][1] == mn["f"] and maj["f"][2] == f):
                if isinstance(mn["f"], tuple) and mn["f"][0] == ">" and mn["f"][1] == maj["f"] and mn["f"][2] == f: continue
                return "ill-typed IMPE"
        if r == "NEGE":
            a, b = refs[0]["f"], refs[1]["f"]
            if not (b == ("~", a) or a == ("~", b)): return "ill-typed NEGE"
        if r in ("ANDE1", "ANDE2"):
            g = refs[0]["f"]
            if not (isinstance(g, tuple) and g[0] == "&" and g[1 if r == "ANDE1" else 2] == f): return "ill-typed ANDE"
        if r == "R" and refs[0]["f"] != f: return "R mismatch"
    return "rej-other (scope/box/goal)"
for fn in sys.argv[1:]:
    res = json.load(open(fn))
    c = collections.Counter(classify(s) for r in res for s in r["samples"])
    n = sum(c.values())
    print(fn, n, {k: f"{v} ({v/n:.0%})" for k, v in c.most_common()})
