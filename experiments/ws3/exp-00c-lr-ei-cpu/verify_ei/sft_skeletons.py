"""Top SFT skeletons, max box depth in SFT data, and the frozen model's rate of depth-3 boxes on eval."""
import os, sys
from collections import Counter, defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from verify import rd, norm, body_lines, canon_proof, E
sk = Counter(); depth = Counter()
seen = set()
for k in (1, 2, 3):
    for r in rd(os.path.join(E, f"run_ei/sft_data/round{k}.jsonl")):
        key = (norm(r["prompt"]), canon_proof(r["completion"]))
        if key in seen: continue
        seen.add(key); ls = body_lines(r["completion"])
        sk[" ".join(l["rule"] + "|" * l["depth"] for l in ls)] += 1
        depth[max(l["depth"] for l in ls)] += 1
print("SFT distinct proofs by max box depth", dict(sorted(depth.items())))
for s, n in sk.most_common(15): print(f"{n:4d} {s}")
for name in ("baseline", "evaluate"):
    rows = rd(os.path.join(E, f"run_ei/{name}/dumps/long_pool.jsonl"))
    d3 = sum(1 for r in rows if r["ok"] and max(l["depth"] for l in body_lines(r["completion"])) >= 3)
    d3a = 0
    for r in rows:
        ls = body_lines(r["completion"])
        if ls and max(l["depth"] for l in ls) >= 3: d3a += 1
    print(name, "accepted with depth>=3:", d3, " any parseable sample with depth>=3:", d3a)
