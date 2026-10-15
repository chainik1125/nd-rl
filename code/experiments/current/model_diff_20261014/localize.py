# ruff: noqa: E501,E741,SIM115  (research scripts: long result lines, bare open(), short loop names)
"""Where, token by token, do the GRPO-wall model (g0_fin) and the no-wall model (g1_fin) differ on depth-3 targets?

Texts: Lean-accepted proofs of ladder targets whose goal has a 3-deep implication spine (A → (B → (C → D))), as found by
EI s1 (a third model, so neither GRPO run wrote them) and by GRPO s0. Control: 2-deep spine targets.
Per token: log p under each checkpoint; grouped by role (box-open `(` at new depth d, other tokens by depth/kind).
python localize.py [n_targets]  -> out/localize.json + printed tables
"""

import collections
import json
import os
import random
import sys

import nd
import torch

torch.set_num_threads(4)
A = "/Users/robbie/repos/nd-rl/artifacts/long_rl_20261014"
sys.path.insert(0, f"{nd.E}/analysis_20261014/wall")
import spine as S  # noqa: E402  (spine(), goal())

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
R = {r["name"]: r for r in map(json.loads, open(f"{nd.E}/data/rl.jsonl"))}


def found(fn):
    d = {}
    for r in map(json.loads, open(fn)):
        if r["name"] not in d or r["written"] < d[r["name"]]["written"]:
            d[r["name"]] = r
    return d


SRC = {
    "ei1": found(f"{A}/v2/long-rl-20261014-v2_untied-ei_s1/found_targets.jsonl"),
    "g0": found(f"{A}/v3/long-rl-20261014-v3_untied-grpo_s0/found_targets.jsonl"),
}
lad = [r for r in R.values() if r["src"].startswith("ladder")]
rng = random.Random(0)
sets = {}
for sp in (2, 3):
    names = sorted(
        r["name"]
        for r in lad
        if S.spine(S.goal(r)) == sp and all(r["name"] in s for s in SRC.values())
    )
    sets[sp] = rng.sample(names, min(N, len(names)))
    print(f"spine {sp}: {len(names)} targets solved by both sources; using {len(sets[sp])}")

MODELS = ["g0_pre", "g0_fin", "g1_pre", "g1_fin", "e1_pre", "e1_fin"]


def role(l, prev_depth):
    if l["tok"] == "(" and l["depth"] > prev_depth:
        return f"box_open d{l['depth']}"
    return f"{l['kind']} d{min(l['depth'], 3)}"


recs = [(sp, src, n) for sp, names in sets.items() for src in SRC for n in names]
models = {}
for mname in MODELS:
    models[mname] = nd.load(nd.CKPT[mname])
tok = models["g0_pre"][1]
out = collections.defaultdict(lambda: collections.defaultdict(list))
worst = collections.defaultdict(collections.Counter)
for sp, src, n in recs:
    f = SRC[src][n]
    ids, n_p = nd.encode(tok, f["prompt"], f["proof"], 0)
    L = nd.labels(tok, ids, n_p)
    x = torch.tensor([ids])
    lps = {}
    for mname, (m, _) in models.items():
        with torch.no_grad():
            lp = torch.log_softmax(m(x[:, :-1])[0].float(), -1)
        lps[mname] = lp[torch.arange(n_p - 1, len(ids) - 1), x[0, n_p:]].tolist()
    prev = 0
    roles = []
    for j, l in enumerate(L):
        r = role(l, prev)
        prev = l["depth"]
        roles.append(r)
        for mname in MODELS:
            out[(sp, src, mname)][r].append(lps[mname][j])
    for mname in MODELS:
        j = min(range(len(L)), key=lambda k: lps[mname][k])
        worst[(sp, src, mname)][roles[j]] += 1

os.makedirs("out", exist_ok=True)
summary = {}
for sp in sets:
    for src in SRC:
        print(
            f"\n== spine {sp}, texts from {src}: mean log p by role (n tokens) | share with p<0.2"
        )
        allroles = sorted(
            out[(sp, src, MODELS[0])], key=lambda r: -len(out[(sp, src, MODELS[0])][r])
        )
        print(f"{'role':>16} {'n':>6} " + " ".join(f"{m:>15}" for m in MODELS))
        for r in allroles:
            v = {m: out[(sp, src, m)][r] for m in MODELS}
            n = len(v[MODELS[0]])
            if n < 20 and not r.startswith("box"):
                continue
            cells = []
            for m in MODELS:
                a = torch.tensor(v[m])
                cells.append(f"{a.mean():7.2f} {float((a.exp() < 0.2).float().mean()):5.0%}  ")
                summary[f"s{sp}|{src}|{m}|{r}"] = [
                    float(a.mean()),
                    float((a.exp() < 0.2).float().mean()),
                    n,
                ]
            print(f"{r:>16} {n:>6} " + " ".join(cells))
        print(
            "worst token role per proof:", {m: worst[(sp, src, m)].most_common(3) for m in MODELS}
        )
json.dump(summary, open("out/localize.json", "w"), indent=1)
