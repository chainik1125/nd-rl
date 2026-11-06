import json, sys, re, collections
def kind(raw):
    toks = raw.split(); depth = 0; openh = []  # (name, depth at :=)
    pending = None; seen = set(); res = None
    for i, t in enumerate(toks):
        if t == "(": depth += 1
        elif t == ")":
            depth -= 1
        if t == "have" and i + 1 < len(toks): pending = toks[i + 1]
        if t == ":=" and pending: openh.append((pending, depth)); pending = None
        if t == ";":
            openh = [(n, d) for n, d in openh if d < depth or (d == depth and False)]
        # close haves whose body depth exited
        openh = [(n, d) for n, d in openh if depth >= d]
        if re.fullmatch(r"n\d+", t) and toks[i - 1] != "have" and not (toks[i - 1] == "(" and toks[i + 1:i + 2] == [":"]):
            if any(n == t for n, _ in openh): return "self-reference (cites the have being proved)"
    return "other unbound (closed box / undefined)"
c = collections.Counter()
ex = {}
for fn in sys.argv[1:]:
    for r in json.load(open(fn)):
        for s in r["samples"]:
            if s["nd"].startswith("LEANPARSE unbound"):
                k = kind(s["raw"]); c[k] += 1; ex.setdefault(k, s["raw"][-400:])
print(c)
for k, v in ex.items(): print(k, "::", v)
