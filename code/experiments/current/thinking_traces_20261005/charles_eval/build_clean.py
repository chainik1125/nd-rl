"""Charles's textbook release 2026-10-02 as a held-out evaluation set, minus problems that share a class with
our training pools.

    nice python3 code/experiments/current/thinking_traces_20261005/charles_eval/build_clean.py

A problem's class is its renaming class (atoms P-S permuted, premises as a set) and that of its curried normal
form (leading antecedents of the goal moved into the premises), Charles's audit rule of 2026-09-30. Pools
checked: the K12 Stage-1 theorems, rl_hard12 and the whole hard712 pool (its samples a and b trained the
round-2 and round-3 models). Writes `charles_clean.jsonl` (candidate_v0 + candidate_v1 + batch3, overlaps
removed) and `charles_clean.report.json` next to the release copy. Eval only: never train on these.
"""

import itertools
import json
import sys

R = "/scratch/cowifornia/nd-rl/think/runs"
ATOMS = ["P", "Q", "R", "S"]


def parse(prompt):
    t = prompt.split()
    assert t[0] == "THM" and t[-1] == "PRF", prompt
    i = t.index("SEQ")
    prem, goal = t[1:i], t[i + 1 : -1]
    # split premises on top-level commas
    ps, cur, depth = [], [], 0
    for tok in prem:
        if tok == "," and depth == 0:
            ps.append(cur)
            cur = []
            continue
        depth += tok == "("
        depth -= tok == ")"
        cur.append(tok)
    if cur:
        ps.append(cur)
    return [tuple(p) for p in ps], tuple(goal)


def split_imp(f):
    """( A > B ) at top level -> (A, B) else None."""
    if len(f) < 5 or f[0] != "(" or f[-1] != ")":
        return None
    depth = 0
    for j, tok in enumerate(f[1:-1], 1):
        depth += tok == "("
        depth -= tok == ")"
        if tok == ">" and depth == 0:
            return f[1:j], f[j + 1 : -1]
    return None


def curry(ps, g):
    ps = list(ps)
    while (s := split_imp(g)) is not None:
        a, b = s
        ps.append(tuple(a))
        g = tuple(b)
    return ps, g


def canon(ps, g):
    best = None
    for perm in itertools.permutations(ATOMS):
        m = dict(zip(ATOMS, perm))
        ren = lambda f: tuple(m.get(x, x) for x in f)
        key = (tuple(sorted(" ".join(ren(p)) for p in set(ps))), " ".join(ren(g)))
        best = key if best is None or key < best else best
    return best


def classes(prompt):
    ps, g = parse(prompt)
    return {canon(ps, g), canon(*curry(ps, g))}


def load(path, key="prompt"):
    with open(path) as f:
        for line in f:
            yield json.loads(line)[key]


charles = {}
for f in ["candidate_v0", "candidate_v1", "batch3", "hand_proved_14"]:
    for r in map(json.loads, open(f"{R}/pools/charles_1002/{f}.jsonl")):
        charles[(f, r["name"])] = classes(r["prompt"])
pools = {
    "K12 (Stage 1)": "/scratch/cowifornia/nd-rl/bench/dan_ckpts/data/train_k12.jsonl",
    "rl_hard12": f"{R}/pools/rl_hard12.jsonl",
    "hard712 pool (all)": f"{R}/pools/hard712/pool.jsonl",
}
for extra in sys.argv[1:]:
    pools[extra] = extra
report = {}
hit_any = set()
for name, path in pools.items():
    cls = set()
    for p in load(path):
        try:
            cls |= classes(p)
        except Exception:
            pass
    hits = sorted(k for k, c in charles.items() if c & cls)
    by = {}
    for f, _ in hits:
        by[f] = by.get(f, 0) + 1
    print(f"{name:22s} overlapping Charles problems: {len(hits)} {by}")
    report[name] = {"n": len(hits), "by_file": by, "names": [n for _, n in hits]}
    hit_any |= set(hits)
out = []
for f in ["candidate_v0", "candidate_v1", "batch3"]:
    for r in map(json.loads, open(f"{R}/pools/charles_1002/{f}.jsonl")):
        if (f, r["name"]) not in hit_any:
            r["charles_file"] = f
            out.append(r)
with open(f"{R}/pools/charles_1002/charles_clean.jsonl", "w") as g:
    g.writelines(json.dumps(r) + "\n" for r in out)
report["kept"] = len(out)
report["removed"] = len({k for k in hit_any if k[0] != "hand_proved_14"})
json.dump(report, open(f"{R}/pools/charles_1002/charles_clean.report.json", "w"), indent=1)
print(f"kept {len(out)} of 173 (candidate_v0 + candidate_v1 + batch3)")
