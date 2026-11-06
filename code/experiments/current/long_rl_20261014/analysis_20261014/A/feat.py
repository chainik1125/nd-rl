"""Structural features of ND proofs/theorems. Parse only (no checking). Output: A/feats.jsonl, one row per theorem
with pool (rl/dev/old), solved label for untied-ei s1, and features from the gold (or generator) proof."""
import json, hashlib, collections, sys

E = "/Users/robbie/repos/nd-rl/code/experiments/current/long_rl_20261014"
ART = "/Users/robbie/repos/nd-rl/artifacts/long_rl_20261014/v2/long-rl-20261014-v2_untied-ei_s1"
FORK = "/private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/fork51604b3"
OUT = "/private/tmp/claude-501/-Users-robbie-repos-nd-rl/a1d02fb4-ba43-48c4-838b-e227293b6c2a/scratchpad/probe/A/feats.jsonl"


def pf(toks, i):
    t = toks[i]
    if t == "(":
        if toks[i + 1] == "~":
            s, j = pf(toks, i + 2)
            return ("~", s), j + 1
        l, j = pf(toks, i + 1)
        op = toks[j]
        r, k = pf(toks, j + 1)
        return (op, l, r), k + 1
    return t, i + 1  # atom or F


def form(s):
    return pf(s.split(), 0)[0]


def show(f):
    if isinstance(f, str):
        return f
    if f[0] == "~":
        return f"( ~ {show(f[1])} )"
    return f"( {show(f[1])} {f[0]} {show(f[2])} )"


def parse_proof(p):
    lines = []
    for seg in p.split(" ; "):
        seg = seg.strip()
        if not seg or seg == "QED":
            continue
        head, rule = seg.rsplit(" : ", 1)
        toks = head.split()
        idx = int(toks[0][1:])
        d = 0
        while toks[1 + d] == "|":
            d += 1
        f = pf(toks[1 + d:], 0)[0]
        rt = rule.split()
        lines.append(dict(idx=idx, depth=d, f=f, rule=rt[0], refs=[int(x[1:]) for x in rt[1:]]))
    return lines


def parse_prompt(prompt):
    body = prompt.split("THM", 1)[1].rsplit("PRF", 1)[0]
    prem, goal = body.split(" SEQ ")
    prem = prem.strip()
    prems = []
    if prem:
        # split on top-level commas
        depth, cur = 0, []
        for t in prem.split():
            if t == "," and depth == 0:
                prems.append(" ".join(cur)); cur = []
                continue
            depth += t == "("
            depth -= t == ")"
            cur.append(t)
        prems.append(" ".join(cur))
    return [form(p) for p in prems], form(goal.strip())


def subformulas(f, acc=None):
    acc = set() if acc is None else acc
    acc.add(f)
    if not isinstance(f, str):
        for s in f[1:]:
            subformulas(s, acc)
    return acc


def atoms(f):
    return {s for s in subformulas(f) if isinstance(s, str) and s != "F"}


def kind(f):
    if isinstance(f, str):
        return "F" if f == "F" else "atom"
    return {"~": "neg", ">": "imp", "v": "or", "&": "and"}[f[0]]


def antecedent_closure(prems):
    """Formulas that are a 'wanted input' of some premise: antecedents of implications and (sub)parts thereof,
    negated formulas' bodies (to use ~X via NEGE you need X)."""
    out = set()
    for p in prems:
        for s in subformulas(p):
            if not isinstance(s, str) and s[0] == ">":
                out.add(s[1])
            if not isinstance(s, str) and s[0] == "~":
                out.add(s[1])
    return out


def features(prems, goal, proof):
    L = parse_proof(proof)
    byidx = {l["idx"]: l for l in L}
    body = [l for l in L if l["rule"] != "PR"]
    prem_idx = [l["idx"] for l in L if l["rule"] == "PR"]
    rc = collections.Counter(l["rule"] for l in body)
    gsub = subformulas(goal)
    psub = set().union(*[subformulas(p) for p in prems]) if prems else set()
    ante = antecedent_closure(prems)
    # citations used: walk back from last line
    used = set()
    stack = [L[-1]["idx"]]
    while stack:
        i = stack.pop()
        if i in used:
            continue
        used.add(i)
        l = byidx[i]
        rr = list(l["refs"])
        if l["rule"] in ("IMPI", "NEGI") and len(rr) == 2:
            rr = list(range(rr[0], rr[1] + 1))
        if l["rule"] == "ORE" and len(rr) == 5:
            rr = [rr[0]] + list(range(rr[1], rr[2] + 1)) + list(range(rr[3], rr[4] + 1))
        stack += [r for r in rr if r in byidx]
    pos = {l["idx"]: k for k, l in enumerate(body)}
    dists, fan = [], collections.Counter()
    for l in body:
        for r in l["refs"]:
            fan[r] += 1
            if r in pos and l["rule"] not in ("IMPI", "NEGI", "ORE"):
                dists.append(pos[l["idx"]] - pos[r])
    # ORE nesting: depth of ORE-boxes. Determine box structure from AS lines
    as_lines = [l for l in body if l["rule"] == "AS"]
    ore_as = set()
    for l in body:
        if l["rule"] == "ORE" and len(l["refs"]) == 5:
            ore_as |= {l["refs"][1], l["refs"][3]}
    # nesting of case splits: max number of enclosing ORE-boxes over all lines
    enclosing = []  # stack of (depth, is_ore)
    max_case_nest, cur = 0, []
    for l in body:
        while cur and (cur[-1][0] > l["depth"] or (l["rule"] == "AS" and cur[-1][0] >= l["depth"])):
            cur.pop()
        if l["rule"] == "AS":
            cur.append((l["depth"], l["idx"] in ore_as))
        max_case_nest = max(max_case_nest, sum(1 for c in cur if c[1]) // 2 + (sum(1 for c in cur if c[1]) % 2))
    # lemma-style introductions: intro rules whose output is NOT a subformula of the goal (so it must be built to feed
    # a premise / an elimination)
    lemma = collections.Counter()
    for l in body:
        if l["rule"] in ("IMPI", "NEGI", "ORI1", "ORI2", "ANDI") and l["f"] not in gsub:
            lemma[l["rule"][:3]] += 1
    # boxes opened for IMPI/NEGI vs ORE, and whether their hypothesis is goal-driven
    n_impi_box = rc["IMPI"]
    n_negi_box = rc["NEGI"]
    # reductio on non-negation: NEGI then DN
    raa = rc["DN"]
    # disjunction eliminated is derived (not a premise / premise conjunct) ?
    ore_derived = 0
    for l in body:
        if l["rule"] == "ORE" and len(l["refs"]) == 5:
            d = byidx.get(l["refs"][0])
            if d and d["rule"] not in ("PR",) and d["f"] not in psub:
                ore_derived += 1
    elim = sum(rc[r] for r in ("IMPE", "ANDE1", "ANDE2", "NEGE", "R"))
    prem_used = len([i for i in prem_idx if i in used])
    fe = dict(
        n_lines_gold=len(body), max_depth=max([l["depth"] for l in body] + [0]), n_boxes=len(as_lines),
        n_impe=rc["IMPE"], n_ande=rc["ANDE1"] + rc["ANDE2"], n_andi=rc["ANDI"], n_ori=rc["ORI1"] + rc["ORI2"],
        n_ore=rc["ORE"], n_impi=n_impi_box, n_negi=n_negi_box, n_nege=rc["NEGE"], n_bote=rc["BOTE"], n_dn=raa,
        n_r=rc["R"], case_nest=max_case_nest, ore_derived=ore_derived,
        lemma_impi=lemma["IMP"], lemma_negi=lemma["NEG"], lemma_ori=lemma["ORI"], lemma_andi=lemma["AND"],
        n_lemma=sum(lemma.values()),
        max_dist=max(dists) if dists else 0, mean_dist=sum(dists) / len(dists) if dists else 0,
        max_fan=max(fan.values()) if fan else 0,
        elim_frac=elim / max(1, len(body)),
        prem_used=prem_used, n_distract=len(prems) - prem_used,
    )
    return fe


def stmt_features(prems, goal):
    at = set().union(atoms(goal), *[atoms(p) for p in prems])
    gk = kind(goal)
    neg_in_goal = sum(1 for s in subformulas(goal) if kind(s) == "neg")
    neg_in_prem = sum(1 for p in prems for s in subformulas(p) if kind(s) == "neg")
    or_in_prem = sum(1 for p in prems for s in subformulas(p) if kind(s) == "or")
    goal_atoms_in_prem = len(atoms(goal) & set().union(*[atoms(p) for p in prems])) if prems else 0
    # goal decomposition depth: strip > as long as possible (IMPI chain), count
    g, impi_chain = goal, 0
    while kind(g) == "imp":
        impi_chain += 1
        g = g[2]
    return dict(n_prem=len(prems), atoms=len(at), goal_kind=gk, goal_core=kind(g), impi_chain=impi_chain,
                neg_goal=neg_in_goal, neg_prem=neg_in_prem, or_prem=or_in_prem,
                goal_size=len(subformulas(goal)), prem_size=sum(len(subformulas(p)) for p in prems),
                goal_in_prem=int(any(goal in subformulas(p) for p in prems)), goal_atoms_in_prem=goal_atoms_in_prem)


def main():
    read = lambda f: [json.loads(l) for l in open(f)]
    found = collections.defaultdict(list)
    for r in read(f"{ART}/found_targets.jsonl"):
        found[r["name"]].append(r)
    ev = {r["name"]: r for r in read(f"{ART}/eval_dev.jsonl")}
    evo = {r["name"]: r for r in read(f"{ART}/eval_old_dev.jsonl")}
    lad = {r["prompt"]: r for r in read(f"{FORK}/data/ladder/rl_targets.jsonl")}
    out = []
    for pool, recs in (("rl", read(f"{E}/data/rl.jsonl")), ("dev", read(f"{E}/data/dev.jsonl"))):
        for r in recs:
            prems, goal = parse_prompt(r["prompt"])
            row = dict(pool=pool, name=r["name"], body=r["body"], band=r["band"], src=r["src"].split(":")[0],
                       prompt=r["prompt"], body_exact=r["body_exact"])
            row.update(stmt_features(prems, goal))
            proof = r.get("proof")
            if proof is None and r["prompt"] in lad:
                lr = lad[r["prompt"]]
                row["schema"] = lr.get("schema") or lr["source"]
                proof = lr.get("gen_proof")
                if proof:  # generator proof uses its own atom naming (thm) -- we only use structure, but prompt
                    pass   # atoms differ; structure features are naming-invariant except lemma checks vs goal
            row["gold"] = proof
            if proof:
                try:
                    if r["src"].startswith("ladder"):
                        lp, lg = parse_prompt("THM " + lad[r["prompt"]]["thm"].replace(" |- ", " SEQ ") + " PRF")
                        row.update(features(lp, lg, proof))
                    else:
                        row.update(features(prems, goal, proof))
                except Exception as e:  # noqa
                    print("parse fail", r["name"], e, file=sys.stderr)
            if pool == "rl":
                fs = found.get(r["name"], [])
                row["solved"] = int(bool(fs))
                row["first_round"] = min((f["round"] for f in fs), default=None)
                row["n_found"] = len(fs)
                row["found_pruned_min"] = min((f["pruned"] for f in fs), default=None)
                row["found_proof"] = min(fs, key=lambda f: f["pruned"])["proof"] if fs else None
            else:
                e = ev[r["name"]]
                row["solved"] = int(e["solved"])
                row["n_ok"] = e["n_ok"]
                row["found_proof"] = e["proofs"][0] if e["proofs"] else None
                row["found_pruned_min"] = min(e["pruned"]) if e["pruned"] else None
            out.append(row)
    old = [r for r in read(f"{FORK}/data/ladder/transfer.jsonl")
           if int(hashlib.sha1(r["key"].encode()).hexdigest(), 16) % 2 == 0]
    for r in old:
        prems, goal = parse_prompt(r["prompt"])
        row = dict(pool="old", name=r["name"], body=r["n_lines"] - r["n_prem"], src=r["source"],
                   schema=r.get("schema") or r["source"], prompt=r["prompt"], gold=r.get("gen_proof"))
        row.update(stmt_features(prems, goal))
        if row["gold"]:
            lp, lg = parse_prompt("THM " + r["thm"].replace(" |- ", " SEQ ") + " PRF")
            row.update(features(lp, lg, row["gold"]))
        e = evo[r["name"]]
        row["solved"] = int(e["solved"])
        row["n_ok"] = e["n_ok"]
        row["found_proof"] = e["proofs"][0] if e["proofs"] else None
        row["found_pruned_min"] = min(e["pruned"]) if e["pruned"] else None
        out.append(row)
    with open(OUT, "w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    print(len(out), collections.Counter(r["pool"] for r in out), sum("n_boxes" in r for r in out))


if __name__ == "__main__":
    main()
