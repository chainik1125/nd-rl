import json, gzip, collections, sys
sys.path.insert(0, ".")
from feat import features, parse_prompt, parse_proof, subformulas, kind
E = "/Users/robbie/repos/nd-rl/code/experiments/current/long_rl_20261014"
def extra(prems, goal, proof):
    """pattern flags"""
    L = parse_proof(proof); by = {l["idx"]: l for l in L}
    g = subformulas(goal)
    flags = collections.Counter()
    for l in L:
        if l["rule"] == "ANDI" and l["f"] not in g:
            # used where?
            users = [u for u in L if l["idx"] in u["refs"]]
            for u in users:
                if u["rule"] == "IMPE": flags["andi_to_impe"] += 1
                if u["rule"] == "NEGE": flags["andi_to_nege"] += 1
        if l["rule"] == "NEGI" and l["f"] not in g and not any(u["rule"] == "DN" and l["idx"] in u["refs"] for u in L): flags["lemma_negi"] += 1
        if l["rule"] == "IMPI" and l["f"] not in g: flags["lemma_impi"] += 1
        if l["rule"] in ("ORI1", "ORI2") and l["f"] not in g: flags["lemma_ori"] += 1
        if l["rule"] == "ORE" and l["f"] == "F": flags["ore_to_F"] += 1
        if l["rule"] == "DN":
            src = by.get(l["refs"][0])
            if src and src["rule"] == "NEGI": flags["raa"] += 1
        if l["rule"] == "ORE" and kind(l["f"]) not in ("atom", "F"): flags["ore_nonatomic"] += 1
    return flags
if __name__ == "__main__":
    n = 0; c = collections.Counter(); cb = collections.defaultdict(collections.Counter); nb = collections.Counter()
    for line in gzip.open(f"{E}/data/pretrain.jsonl.gz", "rt"):
        r = json.loads(line)
        prems, goal = parse_prompt(r["prompt"])
        f = extra(prems, goal, r["proof"])
        n += 1; nb[r["body"]] += 1
        for k, v in f.items():
            if v: c[k] += 1; cb[r["body"]][k] += 1
    print("pretrain n", n, {k: f"{v} ({v/n:.3%})" for k, v in c.items()})
    for b in sorted(nb): print(b, nb[b], {k: v for k, v in cb[b].items()})
