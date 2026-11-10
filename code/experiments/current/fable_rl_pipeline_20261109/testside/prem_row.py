"""Extra row from TEST_PLAN.md item 6: pass@1 / pass@5 for theorems with more than 3 premises (x has
at most 3) and for the rest. Prints counts and rates only, never statements.

    python prem_row.py --labels L1,L2,L3 [--sets t832,tb72,t1320,t2130]
"""
import argparse, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib import SET_TITLE, load_set, pass_at_k, per_theorem_from_attempts, read_jsonl
from score import load_refs, boot, rates


def n_prem(thm):
    left = thm.split("|-")[0].strip()
    if not left:
        return 0
    d, n = 0, 1
    for tok in left.split():
        if tok == "(":
            d += 1
        elif tok == ")":
            d -= 1
        elif tok == "," and d == 0:
            n += 1
    return n


ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--sets", default="t832,tb72,t1320,t2130")
a = ap.parse_args()
labels = a.labels.split(",")
for sid in a.sets.split(","):
    files = [HERE / "results" / lab / f"{sid}.rechecked.jsonl" for lab in labels]
    if not all(f.exists() for f in files):
        print(f"## {sid}: not run"); continue
    st = load_set(sid)
    names = [r["name"] for r in st]
    K = json.load(open(str(files[0]) + ".meta.json"))["k"]
    pers = [per_theorem_from_attempts(read_jsonl(f), names, K)[0] for f in files]
    refs = load_refs(sid, names)
    np_ = {r["name"]: n_prem(r["thm"]) for r in st}
    dist = {}
    for v in np_.values():
        dist[v] = dist.get(v, 0) + 1
    print(f"## {SET_TITLE[sid]} ({sid}), k {K}, labels {labels}; premise counts {sorted(dist.items())}")
    for g, gn in (("<=3 premises", [n for n in names if np_[n] <= 3]), (">3 premises", [n for n in names if np_[n] > 3])):
        if not gn:
            print(f"  {g}: n 0"); continue
        line = f"  {g}: n {len(gn)} | this "
        mine = {k: rates(pers, gn, k) for k in (1, 5)}
        line += " / ".join(f"{100 * sum(mine[k].values()) / len(gn):.1f}" for k in (1, 5))
        for title, rp in refs.items():
            rr = {k: rates(rp, gn, k) for k in (1, 5)}
            d1 = boot([mine[1][n] - rr[1][n] for n in gn], 4000)
            line += f" | {title[:28]}: " + " / ".join(f"{100 * sum(rr[k].values()) / len(gn):.1f}" for k in (1, 5))
            line += f" (d p@1 {d1[0]:+.1f} [{d1[1]:+.1f}, {d1[2]:+.1f}])"
        print(line)
