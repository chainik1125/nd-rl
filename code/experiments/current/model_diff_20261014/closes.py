# ruff: noqa: E501,SIM115  (research scripts: long result lines, bare open(), short loop names)
"""Box-closing stack test. For every box close in Lean-accepted proofs of ladder targets (goal spine 1-4), score:
  pop    the name after `) ; exact` at depth k: must cite the `have` whose `:=` opened the box that just closed.
         Error 'up' = predicts the opener of the box one level further out (an off-by-one stack pop).
  close  the `)` after that name, closing box k (k >= 1). Error 'eos' = predicts end of proof.
python closes.py [n per spine] [models...] -> table; out/closes.json
"""

import collections
import contextlib
import io
import json
import random
import sys

import nd
import torch

torch.set_num_threads(4)
A = "/Users/robbie/repos/nd-rl/artifacts/long_rl_20261014"
sys.path.insert(0, f"{nd.E}/analysis_20261014/wall")
with contextlib.redirect_stdout(io.StringIO()):
    import spine as S
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
MS = sys.argv[2:] or [
    "g0_pre",
    "g0_fin",
    "g1_pre",
    "g1_fin",
    "e0_pre",
    "e0_fin",
    "e1_pre",
    "e1_fin",
]
R = {r["name"]: r for r in map(json.loads, open(f"{nd.E}/data/rl.jsonl"))}
F = {}
for r in map(json.loads, open(f"{A}/v2/long-rl-20261014-v2_untied-ei_s1/found_targets.jsonl")):
    if r["name"] not in F or r["written"] < F[r["name"]]["written"]:
        F[r["name"]] = r
rng = random.Random(0)
by = collections.defaultdict(list)
for n, r in R.items():
    if r["src"].startswith("ladder") and n in F:
        by[min(S.spine(S.goal(r)), 4)].append(n)
texts = [(sp, n) for sp in (1, 2, 3, 4) for n in rng.sample(sorted(by[sp]), min(N, len(by[sp])))]


def events(tok, ids, n_p):
    """[(kind, depth, position j in ids predicting ids[j+1], correct id, 'up' id or None)]"""
    s = [tok.itos[x] for x in ids]
    ev, stack, opener = (
        [],
        [],
        [],
    )  # stack of parens (True = box); opener: have-name id that owns each open box
    hv = {}  # depth -> last `have` seen at that depth
    for i in range(n_p, len(s)):
        t = s[i]
        if t == "have":
            hv[len(opener)] = ids[i + 1]
        if t == "(":
            box = s[i + 1] == "fun" and s[i + 2] == "("
            stack.append(box)
            if box:
                opener.append(hv[len(opener)])
        elif t == ")":
            if stack and stack.pop():
                opener.pop()
                if s[i + 1] == ";" and s[i + 2] == "exact":
                    k = len(opener)
                    up = opener[-1] if opener else None
                    ev.append(("pop", k, i + 2, ids[i + 3], up))
                    if k >= 1:
                        ev.append(("close", k, i + 3, ids[i + 4], None))
    return ev


M = {m: nd.load(nd.CKPT[m])[0] for m in MS}
tok = nd.load(nd.CKPT[MS[0]])[1]
EOS = tok.eos
st = collections.defaultdict(lambda: collections.defaultdict(list))
for sp, n in texts:
    f = F[n]
    ids, n_p = nd.encode(tok, f["prompt"], f["proof"], 0)
    ev = events(tok, ids, n_p)
    x = torch.tensor([ids])
    for m, mod in M.items():
        with torch.no_grad():
            lp = torch.log_softmax(mod(x)[0].float(), -1)
        for kind, k, j, good, up in ev:
            a = int(lp[j].argmax())
            key = (sp, kind, k)
            st[key][m].append(
                (float(lp[j, good]), a == good, a == up if kind == "pop" else a == EOS)
            )
print("spine = goal implication depth of the target; k = depth of the box being returned to")
print(f"{'spine':>5} {'event':>6} {'k':>2} {'n':>5}  " + "  ".join(f"{m:>18}" for m in MS))
print(f"{'':>22}" + "  ".join(f"{'logp acc err':>18}" for m in MS))
J = {}
for key in sorted(st):
    sp, kind, k = key
    n = len(st[key][MS[0]])
    if n < 10:
        continue
    cells = []
    for m in MS:
        v = st[key][m]
        lp = sum(a for a, _, _ in v) / n
        acc = sum(b for _, b, _ in v) / n
        err = sum(c for _, _, c in v) / n
        cells.append(f"{lp:6.2f} {acc:4.0%} {err:4.0%}")
        J[f"{sp}|{kind}|{k}|{m}"] = [lp, acc, err, n]
    print(f"{sp:>5} {kind:>6} {k:>2} {n:>5}  " + "  ".join(f"{c:>18}" for c in cells))
print("err: pop -> argmax is the opener one level further out; close -> argmax is <eos>")
json.dump(J, open("out/closes.json", "w"), indent=1)
