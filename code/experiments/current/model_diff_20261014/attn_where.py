# ruff: noqa: B023,E501,E741,SIM115  (research scripts: long result lines, bare open(), short loop names)
"""Where does attention from the pop `exact` go? Aggregate attention by the attended token's structural role relative
to the box stack: for each pop at depth k (returning to box k after box k+1 closed), roles of key positions:
  open_k+1:<tok>  tokens of the just-closed box's header `( fun ( nB : ... ) => by` and the `:=`, name before it
  open_k:<tok>    the same for the enclosing box (the 'up' one)
  close:<tok>     the `exact nE )` that just closed box k+1
  other
python attn_where.py [n] [models]"""

import collections
import contextlib
import io
import json
import random
import sys

import nd
import torch

torch.set_num_threads(6)
A = "/Users/robbie/repos/nd-rl/artifacts/long_rl_20261014"
sys.path.insert(0, f"{nd.E}/analysis_20261014/wall")
with contextlib.redirect_stdout(io.StringIO()):
    import spine as S
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
MS = sys.argv[2:] or ["g0_pre", "g0_fin", "g1_fin"]
R = {r["name"]: r for r in map(json.loads, open(f"{nd.E}/data/rl.jsonl"))}
F = {}
for r in map(json.loads, open(f"{A}/v2/long-rl-20261014-v2_untied-ei_s1/found_targets.jsonl")):
    if r["name"] not in F or r["written"] < F[r["name"]]["written"]:
        F[r["name"]] = r
names = sorted(
    n for n, r in R.items() if r["src"].startswith("ladder") and S.spine(S.goal(r)) == 3 and n in F
)
names = random.Random(0).sample(names, N)
HDR = ["have", "name", ":", ":=", "(", "fun", "(b", "binder"]


def analyse(tok, ids, n_p):
    s = [tok.itos[x] for x in ids]
    ev, stack, opener = [], [], []
    hv = {}  # depth -> last `have` seen at that depth
    for i in range(n_p, len(s)):
        t = s[i]
        if t == "have":
            hv[len(opener)] = i
        if t == "(":
            box = s[i + 1] == "fun" and s[i + 2] == "("
            stack.append(box)
            if box:
                opener.append((hv[len(opener)], i))  # have position, '(' position
        elif t == ")" and stack and stack.pop():
            o = opener.pop()
            if s[i + 1] == ";" and s[i + 2] == "exact" and opener:
                k = len(opener)
                role = {}

                def hdr(tag, hv, lp):
                    role[hv] = f"{tag}:have"
                    role[hv + 1] = f"{tag}:name"
                    for q in range(hv + 2, lp):
                        role.setdefault(q, f"{tag}:type")
                    role[lp - 1] = f"{tag}:assign"
                    role[lp] = f"{tag}:("
                    role[lp + 1] = f"{tag}:fun"
                    role[lp + 3] = f"{tag}:binder"

                hdr("closed", *o)
                hdr("up", *opener[-1])
                role[i] = "close:)"
                role[i - 1] = "close:exact_name"
                role[i - 2] = "close:exact"
                role[i + 1] = "close:;"
                ev.append((i + 2, k, role))
    return ev


tok = nd.load(nd.CKPT[MS[0]])[1]
for m in MS:
    model = nd.load(nd.CKPT[m])[0]
    agg = collections.defaultdict(lambda: torch.zeros(6, 8))
    n_ev = collections.Counter()
    for n in names:
        f = F[n]
        ids, n_p = nd.encode(tok, f["prompt"], f["proof"], 0)
        att = nd.attn_maps(model, torch.tensor([ids]))
        for j, k, role in analyse(tok, ids, n_p):
            n_ev[k] += 1
            seen = torch.zeros(6, 8)
            for q, r in role.items():
                agg[(k, r)] += att[:, :, j, q]
                seen += att[:, :, j, q]
            agg[(k, "self")] += att[:, :, j, j]
            agg[(k, "other")] += 1 - seen - att[:, :, j, j]
    print(f"\n== {m}")
    for k in (1, 2):
        print(
            f" k={k} n={n_ev[k]}: heads with >=0.15 mean attention to a role (layer.head role weight)"
        )
        rows = []
        for (kk, r), v in agg.items():
            if kk != k:
                continue
            v = v / n_ev[k]
            for l in range(6):
                for h in range(8):
                    if v[l, h] >= 0.15 and r not in ("other",):
                        rows.append((l, h, r, float(v[l, h])))
        rows.sort()
        print("   " + "  ".join(f"{l}.{h} {r} {w:.2f}" for l, h, r, w in rows))
