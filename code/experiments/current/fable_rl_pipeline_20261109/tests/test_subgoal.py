"""CPU check of common.subgoal_variant on x proofs: every variant replays and Lean accepts it."""
import gzip, json, os, random, sys, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fr.common import *
from fr.common import subgoal_variant, lean_check_nd

rows = [json.loads(l) for l in gzip.open(sys.argv[1], "rt")]
random.Random(1).shuffle(rows)
rows = rows[: int(sys.argv[2]) if len(sys.argv) > 2 else 3000]
items, nds, nprem, fails = [], [], collections.Counter(), 0
for r in rows:
    p, nd = r["prompt"], r["proof"]
    depth = 0
    while True:
        v = subgoal_variant(p, nd)
        if v is None:
            break
        p, nd = v
        try:
            pairs, text = nd_pairs(p, nd)
        except Exception as e:
            fails += 1
            break
        items.append((p, text)); nds.append((p, nd)); nprem[len(parse_prompt(p)[0])] += 1
        depth += 1
print("variants", len(items), "from", len(rows), "proofs; premises:", sorted(nprem.items()), "replay failures", fails)
ok = lean_check(items); ok2 = lean_check_nd(nds)
print("lean (text)", sum(ok), "/", len(ok), " lean (nd route)", sum(ok2), "/", len(ok2))
assert all(ok) and all(ok2) and fails == 0
print(items[0]); print("TEST_OK")
